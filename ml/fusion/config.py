"""
Configuration for the Optical-SAR Fusion model and training.

All hyperparameters are centralized here so training runs are reproducible
and easy to sweep over.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    """Architecture hyperparameters."""

    # Backbone
    backbone: str = "resnet18"          # resnet18 | resnet34 | resnet50
    pretrained: bool = True             # ImageNet pretrained backbone

    # Projection head
    embed_dim: int = 256                # Shared embedding dimensionality
    projection_hidden: int = 512        # Hidden layer in projection MLP

    # Contrastive loss
    temperature: float = 0.07           # NT-Xent temperature (learnable if learn_temperature=True)
    learn_temperature: bool = True      # Make temperature a learnable parameter


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    # Data
    data_root: str = "data/sen12/raw"
    terrains: list[str] = field(default_factory=lambda: ["agri", "barrenland", "grassland", "urban"])
    num_workers: int = 4
    pin_memory: bool = True

    # Image preprocessing
    image_size: int = 224               # Resize/crop to this size
    augment: bool = True                # Enable data augmentation

    # Optimization
    batch_size: int = 128
    epochs: int = 50
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    min_lr: float = 1e-6

    # Terrain classification (multi-task)
    use_terrain_head: bool = True       # Train terrain classifier alongside contrastive
    terrain_loss_weight: float = 0.3    # Weight of terrain classification loss

    # Checkpointing
    checkpoint_dir: str = "checkpoints/fusion"
    save_every: int = 5                 # Save checkpoint every N epochs
    resume_from: str | None = None      # Path to checkpoint to resume from

    # Logging
    log_every: int = 20                 # Print metrics every N steps
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
