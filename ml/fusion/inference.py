"""
Inference utilities for the trained Optical-SAR Dual Encoder with Physics Pre-processing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

from fusion.model import ContrastiveLoss, DualEncoder, TerrainClassifier
from fusion.sar_physics import process_sar_physics, refined_lee_filter
from fusion.transforms import (
    compute_ndwi,
    compute_sar_water_mask,
    normalize_optical,
    normalize_sar,
)

DEFAULT_IMAGE_SIZE = 224


class FusionModel:
    TERRAIN_CLASSES = ["agri", "barrenland", "grassland", "urban", "water"]

    def __init__(
        self,
        model: DualEncoder,
        terrain_head: TerrainClassifier | None = None,
        device: str = "cpu",
        image_size: int = DEFAULT_IMAGE_SIZE,
    ) -> None:
        self.model = model.to(device).eval()
        self.terrain_head = terrain_head.to(device).eval() if terrain_head is not None else None
        self.device = device
        self.image_size = image_size

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: str = "auto") -> "FusionModel":
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
            backbone=cfg.get("backbone", "resnet50"),
            pretrained=False,
            embed_dim=cfg.get("embed_dim", 256),
            projection_hidden=cfg.get("projection_hidden", 512),
            use_attention=cfg.get("use_attention", True),
            attn_dim=cfg.get("attn_dim", 512),
            attn_heads=cfg.get("attn_heads", 8),
            attn_layers=cfg.get("attn_layers", 2),
            sar_in_channels=cfg.get("sar_in_channels", 1),
        )
        model.load_state_dict(ckpt["model"])

        terrain_head = None
        if "terrain_head" in ckpt:
            backbone_name = cfg.get("backbone", "resnet50")
            feat_dim = 768 if backbone_name == "convnext_tiny" else (512 if backbone_name in ("resnet18", "resnet34") else 2048)
            num_classes = ckpt["terrain_head"]["classifier.3.weight"].shape[0]
            use_gate = any(k.startswith("sar_gate") for k in ckpt["terrain_head"])
            terrain_head = TerrainClassifier(feature_dim=feat_dim, num_classes=num_classes, use_cross_gate=use_gate)
            terrain_head.load_state_dict(ckpt["terrain_head"])

        instance = cls(model=model, terrain_head=terrain_head, device=device)
        if terrain_head is not None:
            instance.TERRAIN_CLASSES = cls.TERRAIN_CLASSES[:num_classes]
        return instance

    def _to_raw_tensor(self, image: str | Path | Image.Image | torch.Tensor, modality: str) -> torch.Tensor:
        """Convert path, PIL.Image, or Tensor to an unnormalized [0, 1] resized tensor (C, H, W)."""
        if isinstance(image, torch.Tensor):
            t = image.detach().cpu()
            if t.dim() == 4:
                t = t.squeeze(0)
            return TF.resize(t, [self.image_size, self.image_size], antialias=True)

        pil_img = image if isinstance(image, Image.Image) else Image.open(image)
        if modality == "sar":
            if getattr(self.model, "sar_in_channels", 1) == 3:
                arr_raw = np.array(pil_img)
                vv = arr_raw[:, :, 0] if arr_raw.ndim == 3 else arr_raw
                vh = arr_raw[:, :, 1] if (arr_raw.ndim == 3 and arr_raw.shape[2] >= 2) else None
                phys = process_sar_physics(vv, vh)
                arr = phys.pseudo_rgb_uint8.astype(np.float32) / 255.0
                tensor = torch.from_numpy(arr.transpose(2, 0, 1))
            else:
                gray = np.array(pil_img.convert("L"), dtype=np.float32) / 255.0
                # Apply Refined Lee filter to suppress speckle while preserving edges
                gray_filtered = refined_lee_filter(gray, window_size=5, enl=4.4)
                tensor = torch.from_numpy(np.clip(gray_filtered, 0.0, 1.0)).unsqueeze(0)
        else:
            arr = np.array(pil_img.convert("RGB"), dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr.transpose(2, 0, 1))

        return TF.resize(tensor, [self.image_size, self.image_size], antialias=True)

    def _load_and_preprocess(self, image: str | Path | Image.Image, modality: str) -> torch.Tensor:
        raw_tensor = self._to_raw_tensor(image, modality)
        norm_tensor = normalize_sar(raw_tensor) if modality == "sar" else normalize_optical(raw_tensor)
        return norm_tensor.unsqueeze(0).to(self.device)

    def _prepare_tensor(self, image, modality: str) -> torch.Tensor:
        if isinstance(image, (str, Path, Image.Image)):
            return self._load_and_preprocess(image, modality)
        if isinstance(image, torch.Tensor):
            return (image.unsqueeze(0) if image.dim() == 3 else image).to(self.device)
        raise TypeError(f"Expected str, Path, PIL Image, or Tensor, got {type(image)}")

    @torch.no_grad()
    def embed_sar(self, image) -> np.ndarray:
        return self.model.encode_sar(self._prepare_tensor(image, "sar")).squeeze(0).cpu().numpy()

    @torch.no_grad()
    def embed_optical(self, image) -> np.ndarray:
        return self.model.encode_optical(self._prepare_tensor(image, "optical")).squeeze(0).cpu().numpy()

    @torch.no_grad()
    def compare(self, sar_image, optical_image) -> float:
        sar_emb = self.model.encode_sar(self._prepare_tensor(sar_image, "sar"))
        opt_emb = self.model.encode_optical(self._prepare_tensor(optical_image, "optical"))
        return F.cosine_similarity(sar_emb, opt_emb).item()

    @torch.no_grad()
    def classify_terrain(self, sar_image, optical_image) -> tuple[str, float, dict[str, float]]:
        if self.terrain_head is None:
            raise RuntimeError("No terrain classification head was loaded.")

        sar_tensor = self._prepare_tensor(sar_image, "sar")
        opt_tensor = self._prepare_tensor(optical_image, "optical")
        sar_feat, opt_feat = self.model.get_backbone_features(sar_tensor, opt_tensor)
        probs = F.softmax(self.terrain_head(sar_feat, opt_feat), dim=-1).squeeze(0)

        pred_idx = int(probs.argmax().item())
        return (
            self.TERRAIN_CLASSES[pred_idx],
            float(probs[pred_idx].item()),
            {cls: float(probs[i].item()) for i, cls in enumerate(self.TERRAIN_CLASSES)},
        )

    @torch.no_grad()
    def embed_batch(self, image_paths: list[str | Path], modality: str, batch_size: int = 32) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(image_paths), batch_size):
            batch = torch.cat([self._load_and_preprocess(p, modality) for p in image_paths[i : i + batch_size]], dim=0)
            emb = self.model.encode_sar(batch) if modality == "sar" else self.model.encode_optical(batch)
            all_embeddings.append(emb.cpu())
        return torch.cat(all_embeddings, dim=0).numpy()

    @torch.no_grad()
    def cross_modal_retrieval(
        self,
        query_image,
        query_modality: str,
        gallery_embeddings: np.ndarray,
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        query_emb = self.embed_sar(query_image) if query_modality == "sar" else self.embed_optical(query_image)
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        gallery_norm = gallery_embeddings / (np.linalg.norm(gallery_embeddings, axis=1, keepdims=True) + 1e-8)
        sims = gallery_norm @ query_norm
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [(int(idx), float(sims[idx])) for idx in top_indices]

    def get_analysis(self, sar_image, optical_image) -> dict:
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

        # Always compute NDWI and SAR water masks on unnormalized [0, 1] tensors
        raw_opt = self._to_raw_tensor(optical_image, "optical")
        raw_sar = self._to_raw_tensor(sar_image, "sar")

        ndwi_map = compute_ndwi(raw_opt)
        ndwi_mean = float(ndwi_map.mean().item())
        water_pixel_ratio = float((ndwi_map > 0.0).float().mean().item())

        sar_water = compute_sar_water_mask(raw_sar)
        sar_water_ratio = float(sar_water.mean().item())

        result["water"] = {
            "ndwi_mean": round(ndwi_mean, 4),
            "ndwi_water_pixel_ratio": round(water_pixel_ratio, 4),
            "sar_water_pixel_ratio": round(sar_water_ratio, 4),
        }
        return result