"""
GroundingDINO model wrapper for text-guided region grounding.

Wraps Hugging Face's ``GroundingDinoForObjectDetection`` with helpers for:
    - Loading pre-trained weights
    - Selective layer freezing (backbone / text encoder)
    - Post-processing raw outputs into usable bounding-box dicts

Architecture (simplified):
    ┌──────────────┐   ┌──────────────┐
    │  Satellite   │   │  Text query  │
    │  image       │   │  (e.g.       │
    │  (H × W × 3)│   │  "buildings")│
    └──────┬───────┘   └──────┬───────┘
           │                  │
    ┌──────▼───────┐   ┌──────▼───────┐
    │  Swin-T      │   │  BERT text   │
    │  backbone    │   │  encoder     │
    └──────┬───────┘   └──────┬───────┘
           │                  │
           └──────┬───────────┘
                  │
           ┌──────▼───────┐
           │  Cross-modal │
           │  decoder     │
           │  + det heads │
           └──────┬───────┘
                  │
           ┌──────▼───────┐
           │  Bounding    │
           │  boxes +     │
           │  scores      │
           └──────────────┘
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
)

from ml.grounding.config import ModelConfig


class GroundingModel(nn.Module):
    """Wrapper around HF GroundingDINO for remote-sensing grounding.

    Handles model loading, selective freezing, and output post-processing.

    Args:
        config: Model configuration with model_id, freeze flags, thresholds.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.config.model_id,
        )
        self.processor = AutoProcessor.from_pretrained(self.config.model_id)

        # Apply freezing strategy
        if self.config.freeze_backbone:
            self._freeze_backbone()
        if self.config.freeze_text_encoder:
            self._freeze_text_encoder()

    # ── Freezing helpers ─────────────────────────────────────────────────

    def _freeze_backbone(self) -> None:
        """Freeze the Swin-T vision backbone parameters."""
        frozen = 0
        for name, param in self.model.named_parameters():
            if "backbone" in name or "input_proj" in name:
                param.requires_grad = False
                frozen += 1
        print(f"  Froze {frozen} backbone parameters")

    def _freeze_text_encoder(self) -> None:
        """Freeze the BERT text encoder parameters."""
        frozen = 0
        for name, param in self.model.named_parameters():
            if "text_backbone" in name or "bert" in name.lower():
                param.requires_grad = False
                frozen += 1
        print(f"  Froze {frozen} text encoder parameters")

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters (for full fine-tuning)."""
        for param in self.model.parameters():
            param.requires_grad = True

    def get_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.model.parameters())

    # ── Forward ──────────────────────────────────────────────────────────

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        labels: list[dict] | None = None,
        **kwargs,
    ):
        """Forward pass through GroundingDINO.

        During training (labels provided): returns a dict with losses.
        During inference (no labels): returns raw model outputs.

        Args:
            pixel_values: (B, 3, H, W) preprocessed images.
            input_ids: (B, seq_len) tokenized text queries.
            attention_mask: (B, seq_len) attention mask for text.
            token_type_ids: (B, seq_len) token type IDs.
            labels: List of dicts with 'class_labels' (Tensor) and
                    'boxes' (Tensor, cx/cy/w/h normalized 0–1). Only for training.

        Returns:
            GroundingDinoObjectDetectionOutput with loss (if training)
            or logits + pred_boxes (if inference).
        """
        return self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            **kwargs,
        )

    # ── Post-processing ──────────────────────────────────────────────────

    @torch.no_grad()
    def post_process(
        self,
        outputs,
        target_sizes: torch.Tensor,
        box_threshold: float | None = None,
        text_threshold: float | None = None,
    ) -> list[dict]:
        """Convert raw model outputs to lists of detected boxes.

        Args:
            outputs: Raw GroundingDINO output from forward().
            target_sizes: (B, 2) tensor of (height, width) for each image.
            box_threshold: Min box confidence (default from config).
            text_threshold: Min text score (default from config).

        Returns:
            List of dicts per image, each with:
                'boxes': Tensor (N, 4) in xyxy pixel coords
                'scores': Tensor (N,)
                'labels': list[str] matched text spans
        """
        box_thr = box_threshold or self.config.box_threshold
        text_thr = text_threshold or self.config.text_threshold

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            input_ids=None,  # Will use stored input_ids if available
            box_threshold=box_thr,
            text_threshold=text_thr,
            target_sizes=target_sizes,
        )

        return results

    # ── Convenience ──────────────────────────────────────────────────────

    def preprocess(
        self,
        images,
        text: str | list[str],
        return_tensors: str = "pt",
    ) -> dict:
        """Preprocess images and text using the HF processor.

        Args:
            images: PIL Image(s) or tensor(s).
            text: Text query or list of queries.
            return_tensors: 'pt' for PyTorch tensors.

        Returns:
            Dict with 'pixel_values', 'input_ids', 'attention_mask', etc.
        """
        return self.processor(
            images=images,
            text=text,
            return_tensors=return_tensors,
        )


# ── Quick test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing GroundingModel...")
    print("Loading pre-trained GroundingDINO-Tiny...")

    config = ModelConfig(freeze_backbone=True, freeze_text_encoder=False)
    model = GroundingModel(config)

    total = model.get_total_params() / 1e6
    trainable = model.get_trainable_params() / 1e6
    print(f"  Total params:     {total:.1f}M")
    print(f"  Trainable params: {trainable:.1f}M")
    print(f"  Frozen params:    {total - trainable:.1f}M")

    # Test forward with dummy input
    from PIL import Image
    import requests
    from io import BytesIO

    # Use a small test image
    dummy_image = Image.new("RGB", (800, 800), color=(100, 150, 200))
    text = "buildings . roads ."

    inputs = model.preprocess(dummy_image, text)
    print(f"  pixel_values shape: {inputs['pixel_values'].shape}")
    print(f"  input_ids shape:    {inputs['input_ids'].shape}")

    # Forward (inference mode)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    print(f"  pred_boxes shape:   {outputs.pred_boxes.shape}")
    print(f"  logits shape:       {outputs.logits.shape}")

    # Post-process
    target_sizes = torch.tensor([[800, 800]])
    results = model.post_process(outputs, target_sizes)
    print(f"  Detected {len(results[0]['boxes'])} objects")

    print("\nAll tests passed.")
