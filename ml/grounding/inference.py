"""
Inference utilities for the trained GroundingDINO grounding model.

Provides high-level functions to:
    - Load a fine-tuned model from checkpoint or a pre-trained model
    - Ground text queries in satellite images (return bounding boxes)
    - Batch inference
    - Full analysis dict for the agentic controller

Usage:
    from ml.grounding.inference import GroundingInference

    # From fine-tuned checkpoint
    model = GroundingInference.from_checkpoint("checkpoints/grounding/best.pt")

    # Or zero-shot pre-trained
    model = GroundingInference.from_pretrained()

    # Ground a query
    results = model.ground("path/to/satellite.png", "water body near bridge")
    # [{"box": [x1,y1,x2,y2], "score": 0.87, "label": "water body near bridge"}]

    # Full analysis for the controller
    analysis = model.get_analysis("satellite.png", "buildings and roads")
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None

from ml.grounding.config import ModelConfig
from ml.grounding.model import GroundingModel


class GroundingInference:
    """High-level inference wrapper for the GroundingDINO grounding model.

    Args:
        grounding: The GroundingModel instance.
        device: Device to run inference on.
    """

    def __init__(
        self,
        grounding: GroundingModel,
        device: str = "cpu",
    ) -> None:
        self.grounding = grounding
        self.grounding.model.to(device)
        self.grounding.model.eval()
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "IDEA-Research/grounding-dino-tiny",
        device: str = "auto",
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
    ) -> "GroundingInference":
        """Load the pre-trained (zero-shot) GroundingDINO model.

        Args:
            model_id: HuggingFace model ID.
            device: Target device.
            box_threshold: Min box confidence.
            text_threshold: Min text grounding score.

        Returns:
            GroundingInference ready for zero-shot grounding.
        """
        device = _resolve_device(device)

        config = ModelConfig(
            model_id=model_id,
            freeze_backbone=False,
            freeze_text_encoder=False,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        grounding = GroundingModel(config)

        print(f"Loaded pre-trained GroundingDINO: {model_id}")
        return cls(grounding=grounding, device=device)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "auto",
    ) -> "GroundingInference":
        """Load a fine-tuned model from a training checkpoint.

        Args:
            checkpoint_path: Path to the .pt checkpoint file.
            device: Target device.

        Returns:
            GroundingInference ready for inference.
        """
        device = _resolve_device(device)

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        cfg_dict = ckpt.get("model_config", {})

        config = ModelConfig(
            model_id=cfg_dict.get("model_id", "IDEA-Research/grounding-dino-tiny"),
            freeze_backbone=False,  # No freezing at inference
            freeze_text_encoder=False,
            box_threshold=cfg_dict.get("box_threshold", 0.25),
            text_threshold=cfg_dict.get("text_threshold", 0.25),
        )
        grounding = GroundingModel(config)
        grounding.model.load_state_dict(ckpt["model"])

        print(f"Loaded fine-tuned GroundingDINO from {checkpoint_path}")
        if "epoch" in ckpt:
            print(f"  Trained for {ckpt['epoch']} epochs")
        if "metrics" in ckpt:
            loss = ckpt["metrics"].get("val_loss", ckpt["metrics"].get("loss", "N/A"))
            print(f"  Checkpoint loss: {loss}")

        return cls(grounding=grounding, device=device)

    # ── Core inference ───────────────────────────────────────────────────

    @torch.no_grad()
    def ground(
        self,
        image,
        text_query: str,
        box_threshold: float | None = None,
        text_threshold: float | None = None,
    ) -> list[dict]:
        """Ground a text query in an image, returning bounding boxes.

        Args:
            image: PIL Image, file path, or numpy array.
            text_query: Natural-language expression (e.g. "water body near bridge").
                        For multiple classes, separate with periods:
                        "buildings . roads . water bodies ."
            box_threshold: Override default box confidence threshold.
            text_threshold: Override default text grounding threshold.

        Returns:
            List of dicts, each with:
                'box': [x1, y1, x2, y2] in pixel coordinates
                'score': float confidence
                'label': str matched text span
        """
        pil_image = self._load_image(image)
        w, h = pil_image.size

        # Ensure the query ends with a period (GroundingDINO convention)
        if not text_query.strip().endswith("."):
            text_query = text_query.strip() + " ."

        # Preprocess
        inputs = self.grounding.preprocess(pil_image, text_query)
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        # Forward
        outputs = self.grounding(**inputs)

        # Post-process
        target_sizes = torch.tensor([[h, w]], device=self.device)
        box_thr = box_threshold or self.grounding.config.box_threshold
        text_thr = text_threshold or self.grounding.config.text_threshold

        results = self.grounding.processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs["input_ids"],
            box_threshold=box_thr,
            text_threshold=text_thr,
            target_sizes=target_sizes,
        )

        # Format output
        detections = []
        if results and len(results) > 0:
            r = results[0]
            boxes = r["boxes"].cpu().numpy()       # (N, 4) xyxy
            scores = r["scores"].cpu().numpy()     # (N,)
            labels = r["labels"]                   # list[str]

            for box, score, label in zip(boxes, scores, labels):
                detections.append({
                    "box": [round(float(c), 2) for c in box],
                    "score": round(float(score), 4),
                    "label": label.strip(),
                })

        return detections

    @torch.no_grad()
    def ground_batch(
        self,
        images: list,
        text_queries: list[str],
        box_threshold: float | None = None,
        text_threshold: float | None = None,
    ) -> list[list[dict]]:
        """Ground text queries in a batch of images.

        Args:
            images: List of PIL Images, file paths, or numpy arrays.
            text_queries: List of queries (one per image).
            box_threshold: Override box confidence threshold.
            text_threshold: Override text grounding threshold.

        Returns:
            List of detection lists (one list per image).
        """
        # Process one at a time (GroundingDINO text lengths can vary)
        all_results = []
        for img, query in zip(images, text_queries):
            result = self.ground(img, query, box_threshold, text_threshold)
            all_results.append(result)
        return all_results

    def get_analysis(
        self,
        image,
        text_query: str,
        box_threshold: float | None = None,
        text_threshold: float | None = None,
    ) -> dict:
        """Full analysis of a grounding query.

        Returns a dict suitable for the agentic controller / backend.

        Args:
            image: PIL Image, file path, or numpy array.
            text_query: Natural-language grounding expression.

        Returns:
            dict with:
                'query': str
                'detections': list of box dicts
                'num_detections': int
                'image_size': [width, height]
                'model_id': str
        """
        pil_image = self._load_image(image)
        w, h = pil_image.size

        detections = self.ground(
            pil_image, text_query, box_threshold, text_threshold,
        )

        return {
            "query": text_query,
            "detections": detections,
            "num_detections": len(detections),
            "image_size": [w, h],
            "model_id": self.grounding.config.model_id,
        }

    # ── Visualization ────────────────────────────────────────────────────

    def visualize(
        self,
        image,
        text_query: str,
        output_path: str | None = None,
        box_threshold: float | None = None,
        text_threshold: float | None = None,
    ) -> "Image.Image":
        """Ground a query and draw bounding boxes on the image.

        Args:
            image: PIL Image or file path.
            text_query: Grounding expression.
            output_path: If provided, save the annotated image here.
            box_threshold: Override box confidence.
            text_threshold: Override text confidence.

        Returns:
            PIL Image with drawn bounding boxes and labels.
        """
        if Image is None or ImageDraw is None:
            raise ImportError("Pillow is required for visualization: pip install Pillow")

        pil_image = self._load_image(image).copy()
        detections = self.ground(pil_image, text_query, box_threshold, text_threshold)

        draw = ImageDraw.Draw(pil_image)

        # Color palette for different labels
        colors = [
            "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF",
            "#00FFFF", "#FF8000", "#8000FF", "#0080FF", "#FF0080",
        ]

        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det["box"]
            color = colors[i % len(colors)]
            label_text = f"{det['label']} ({det['score']:.2f})"

            # Draw box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            # Draw label background
            text_bbox = draw.textbbox((x1, y1 - 15), label_text)
            draw.rectangle(text_bbox, fill=color)
            draw.text((x1, y1 - 15), label_text, fill="white")

        if output_path:
            pil_image.save(output_path)
            print(f"Annotated image saved: {output_path}")

        return pil_image

    # ── Helpers ───────────────────────────────────────────────────────────

    def _load_image(self, image) -> "Image.Image":
        """Load and convert an image to PIL RGB."""
        if Image is None:
            raise ImportError("Pillow is required: pip install Pillow")

        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            return Image.fromarray(image).convert("RGB")
        elif hasattr(image, "convert"):
            return image.convert("RGB")
        else:
            raise TypeError(f"Expected PIL Image, path, or numpy array, got {type(image)}")


def _resolve_device(device: str) -> str:
    """Resolve 'auto' to the best available device."""
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


# ── CLI ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run grounding inference")
    parser.add_argument("--checkpoint", default=None, help="Path to fine-tuned .pt checkpoint")
    parser.add_argument("--pretrained", action="store_true", help="Use zero-shot pre-trained model")
    parser.add_argument("--image", required=True, help="Path to satellite image")
    parser.add_argument("--query", required=True, help="Text query (e.g. 'buildings near road')")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--output", default=None, help="Path to save annotated image")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    # Load model
    if args.checkpoint:
        model = GroundingInference.from_checkpoint(args.checkpoint, device=args.device)
    elif args.pretrained:
        model = GroundingInference.from_pretrained(device=args.device)
    else:
        parser.error("Specify either --checkpoint or --pretrained")

    # Run inference
    analysis = model.get_analysis(
        args.image,
        args.query,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )

    print(f"\nQuery: {analysis['query']}")
    print(f"Image size: {analysis['image_size']}")
    print(f"Detections: {analysis['num_detections']}")

    for i, det in enumerate(analysis["detections"]):
        print(f"  [{i+1}] {det['label']:30s}  score={det['score']:.4f}  box={det['box']}")

    # Visualize if requested
    if args.output:
        model.visualize(
            args.image, args.query, output_path=args.output,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
        )
