"""
Configuration for the GroundingDINO grounding model and training.

All hyperparameters are centralized here so training runs are reproducible
and easy to sweep over.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    """Architecture and inference hyperparameters."""

    # Pre-trained model
    model_id: str = "IDEA-Research/grounding-dino-tiny"

    # Fine-tuning strategy
    freeze_backbone: bool = True        # Freeze Swin-T vision backbone
    freeze_text_encoder: bool = False   # Keep BERT text encoder trainable

    # Inference thresholds
    box_threshold: float = 0.25         # Min confidence to keep a predicted box
    text_threshold: float = 0.25        # Min text-grounding score


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    # Data
    data_name: str = "xiang709/VRSBench"   # HuggingFace dataset name
    data_subset: str = "VRSBench"          # HuggingFace dataset config name
    data_cache_dir: str | None = None      # Local cache dir for HF datasets
    num_workers: int = 2
    pin_memory: bool = True

    # Image preprocessing
    image_size: int = 800               # GroundingDINO default input size

    # Augmentation
    augment: bool = True                # Enable training augmentations

    # Optimization
    batch_size: int = 4                 # Small batches (GroundingDINO is ~172M params)
    epochs: int = 20
    lr: float = 1e-5                    # Low LR for fine-tuning
    backbone_lr: float = 1e-6           # Even lower LR if backbone is unfrozen
    weight_decay: float = 0.01
    warmup_epochs: int = 2
    min_lr: float = 1e-7
    max_grad_norm: float = 0.1          # GroundingDINO uses tight gradient clipping

    # Checkpointing
    checkpoint_dir: str = "checkpoints/grounding"
    save_every: int = 5                 # Save checkpoint every N epochs
    resume_from: str | None = None      # Path to checkpoint to resume from

    # Logging
    log_every: int = 50                 # Print metrics every N steps
    eval_every: int = 1                 # Evaluate on val set every N epochs

    # Device
    device: str = "auto"               # auto | cuda | cpu

    @property
    def checkpoint_path(self) -> Path:
        return Path(self.checkpoint_dir)

    def resolve_device(self) -> str:
        """Determine the actual device to use."""
        if self.device == "auto":
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return self.device
