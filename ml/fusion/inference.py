"""
Inference utilities for the trained Optical-SAR Dual Encoder.

Provides high-level functions to:
    - Load a trained model from checkpoint
    - Embed SAR and optical images
    - Compute cross-modal similarity scores
    - Classify terrain from SAR-optical pairs
    - Retrieve the most similar optical image for a given SAR image (and vice versa)

Usage:
    from ml.fusion.inference import FusionModel

    model = FusionModel.from_checkpoint("checkpoints/fusion/best.pt")

    # Embed and compare
    similarity = model.compare("path/to/sar.png", "path/to/optical.png")

    # Classify terrain
    terrain, confidence = model.classify_terrain("sar.png", "optical.png")

    # Batch embed for retrieval
    embeddings = model.embed_folder("path/to/sar_images/", modality="sar")
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    from PIL import Image
except ImportError:
    Image = None

from ml.fusion.model import ContrastiveLoss, DualEncoder, TerrainClassifier
from ml.fusion.transforms import normalize_optical, normalize_sar

# Default image size (should match training config)
DEFAULT_IMAGE_SIZE = 224


class FusionModel:
    """High-level inference wrapper for the trained dual encoder.

    Args:
        model: The DualEncoder model.
        terrain_head: Optional terrain classifier.
        device: Device to run inference on.
        image_size: Expected input image size.
    """

    TERRAIN_CLASSES = ["agri", "barrenland", "grassland", "urban"]

    def __init__(
        self,
        model: DualEncoder,
        terrain_head: TerrainClassifier | None = None,
        device: str = "cpu",
        image_size: int = DEFAULT_IMAGE_SIZE,
    ) -> None:
        self.model = model.to(device).eval()
        self.terrain_head = terrain_head
        if self.terrain_head is not None:
            self.terrain_head = self.terrain_head.to(device).eval()
        self.device = device
        self.image_size = image_size

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: str = "auto") -> "FusionModel":
        """Load a trained model from a checkpoint file.

        Args:
            checkpoint_path: Path to the .pt checkpoint file.
            device: Device to load onto ('auto', 'cuda', 'cpu').

        Returns:
            FusionModel ready for inference.
        """
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        cfg = ckpt.get("model_config", {})

        model = DualEncoder(
            backbone=cfg.get("backbone", "resnet18"),
            pretrained=False,
            embed_dim=cfg.get("embed_dim", 256),
            projection_hidden=cfg.get("projection_hidden", 512),
        )
        model.load_state_dict(ckpt["model"])

        terrain_head = None
        if "terrain_head" in ckpt:
            feat_dim = 512 if cfg.get("backbone", "resnet18") in ("resnet18", "resnet34") else 2048
            terrain_head = TerrainClassifier(feature_dim=feat_dim, num_classes=4)
            terrain_head.load_state_dict(ckpt["terrain_head"])

        print(f"Loaded fusion model from {checkpoint_path}")
        print(f"  Backbone: {cfg.get('backbone', 'resnet18')}, Embed dim: {cfg.get('embed_dim', 256)}")
        if "epoch" in ckpt:
            print(f"  Trained for {ckpt['epoch']} epochs")

        return cls(model=model, terrain_head=terrain_head, device=device)

    # ── Image loading ────────────────────────────────────────────────────

    def _load_and_preprocess(
        self,
        image_path: str | Path,
        modality: str,
    ) -> torch.Tensor:
        """Load an image and preprocess it for the model.

        Args:
            image_path: Path to the image file.
            modality: 'sar' or 'optical'.

        Returns:
            Preprocessed tensor (1, C, H, W) on the model's device.
        """
        import torchvision.transforms.functional as TF

        if Image is None:
            raise ImportError("Pillow is required for image loading: pip install Pillow")

        grayscale = modality == "sar"
        mode = "L" if grayscale else "RGB"
        img = Image.open(image_path).convert(mode)

        # To tensor (0-1 range)
        arr = np.array(img, dtype=np.float32) / 255.0
        if arr.ndim == 2:
            tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)
        else:
            tensor = torch.from_numpy(arr.transpose(2, 0, 1))  # (3, H, W)

        # Resize
        tensor = TF.resize(tensor, [self.image_size, self.image_size], antialias=True)

        # Normalize
        if modality == "sar":
            tensor = normalize_sar(tensor)
        else:
            tensor = normalize_optical(tensor)

        return tensor.unsqueeze(0).to(self.device)  # (1, C, H, W)

    def _prepare_tensor(self, image, modality: str) -> torch.Tensor:
        """Accept either a file path or a pre-loaded tensor."""
        if isinstance(image, (str, Path)):
            return self._load_and_preprocess(image, modality)
        elif isinstance(image, torch.Tensor):
            # Assume already preprocessed, just add batch dim if needed
            if image.dim() == 3:
                image = image.unsqueeze(0)
            return image.to(self.device)
        else:
            raise TypeError(f"Expected str, Path, or Tensor, got {type(image)}")

    # ── Core operations ──────────────────────────────────────────────────

    @torch.no_grad()
    def embed_sar(self, image) -> np.ndarray:
        """Embed a SAR image. Returns (embed_dim,) numpy array."""
        tensor = self._prepare_tensor(image, "sar")
        embedding = self.model.encode_sar(tensor)
        return embedding.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def embed_optical(self, image) -> np.ndarray:
        """Embed an optical image. Returns (embed_dim,) numpy array."""
        tensor = self._prepare_tensor(image, "optical")
        embedding = self.model.encode_optical(tensor)
        return embedding.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def compare(self, sar_image, optical_image) -> float:
        """Compute cosine similarity between a SAR-optical pair.

        Returns:
            Similarity score in [-1, 1]. Higher = more likely the same location.
        """
        sar_tensor = self._prepare_tensor(sar_image, "sar")
        opt_tensor = self._prepare_tensor(optical_image, "optical")

        sar_emb = self.model.encode_sar(sar_tensor)
        opt_emb = self.model.encode_optical(opt_tensor)

        similarity = F.cosine_similarity(sar_emb, opt_emb).item()
        return similarity

    @torch.no_grad()
    def classify_terrain(
        self,
        sar_image,
        optical_image,
    ) -> tuple[str, float, dict[str, float]]:
        """Classify the terrain type from a SAR-optical pair.

        Returns:
            (predicted_class, confidence, all_probabilities)
        """
        if self.terrain_head is None:
            raise RuntimeError("No terrain classification head was loaded. "
                               "Train with --use-terrain-head or load a checkpoint that has one.")

        sar_tensor = self._prepare_tensor(sar_image, "sar")
        opt_tensor = self._prepare_tensor(optical_image, "optical")

        sar_feat, opt_feat = self.model.get_backbone_features(sar_tensor, opt_tensor)
        logits = self.terrain_head(sar_feat, opt_feat)
        probs = F.softmax(logits, dim=-1).squeeze(0)

        pred_idx = probs.argmax().item()
        pred_class = self.TERRAIN_CLASSES[pred_idx]
        confidence = probs[pred_idx].item()

        all_probs = {cls: probs[i].item() for i, cls in enumerate(self.TERRAIN_CLASSES)}

        return pred_class, confidence, all_probs

    @torch.no_grad()
    def embed_batch(
        self,
        image_paths: list[str | Path],
        modality: str,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Embed a batch of images. Returns (N, embed_dim) numpy array.

        Args:
            image_paths: List of image file paths.
            modality: 'sar' or 'optical'.
            batch_size: Processing batch size.
        """
        all_embeddings = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            tensors = torch.cat(
                [self._load_and_preprocess(p, modality) for p in batch_paths],
                dim=0,
            )

            if modality == "sar":
                embeddings = self.model.encode_sar(tensors)
            else:
                embeddings = self.model.encode_optical(tensors)

            all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0).numpy()

    @torch.no_grad()
    def cross_modal_retrieval(
        self,
        query_image,
        query_modality: str,
        gallery_embeddings: np.ndarray,
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        """Retrieve the top-K most similar images from the other modality.

        Args:
            query_image: Path or tensor of the query image.
            query_modality: 'sar' or 'optical' (modality of the query).
            gallery_embeddings: (N, embed_dim) precomputed embeddings of the gallery
                                (should be the OTHER modality).
            top_k: Number of results to return.

        Returns:
            List of (gallery_index, similarity_score) tuples, sorted by similarity.
        """
        if query_modality == "sar":
            query_emb = self.embed_sar(query_image)
        else:
            query_emb = self.embed_optical(query_image)

        # Cosine similarity
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        gallery_norm = gallery_embeddings / (
            np.linalg.norm(gallery_embeddings, axis=1, keepdims=True) + 1e-8
        )
        similarities = gallery_norm @ query_norm

        # Top-K
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = [(int(idx), float(similarities[idx])) for idx in top_indices]

        return results

    def get_analysis(
        self,
        sar_image,
        optical_image,
    ) -> dict:
        """Full analysis of a SAR-optical pair.

        Returns a dict with similarity, embeddings, and terrain classification.
        This is the main method to call from the backend/controller.
        """
        result = {
            "similarity": self.compare(sar_image, optical_image),
            "sar_embedding": self.embed_sar(sar_image).tolist(),
            "optical_embedding": self.embed_optical(optical_image).tolist(),
        }

        if self.terrain_head is not None:
            terrain, confidence, all_probs = self.classify_terrain(sar_image, optical_image)
            result["terrain"] = {
                "predicted": terrain,
                "confidence": confidence,
                "probabilities": all_probs,
            }

        return result


# ── CLI ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run inference with the fusion model")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--sar", required=True, help="Path to SAR image")
    parser.add_argument("--optical", required=True, help="Path to optical image")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    model = FusionModel.from_checkpoint(args.checkpoint, device=args.device)
    result = model.get_analysis(args.sar, args.optical)

    print(f"\nSimilarity:  {result['similarity']:.4f}")
    if "terrain" in result:
        t = result["terrain"]
        print(f"Terrain:     {t['predicted']} ({t['confidence']:.1%})")
        for cls, prob in t["probabilities"].items():
            bar = "#" * int(prob * 30)
            print(f"  {cls:12s} {prob:6.1%}  {bar}")
