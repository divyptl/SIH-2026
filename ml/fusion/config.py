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
    backbone: str = "convnext_tiny"     # Switched from resnet50 to convnext_tiny for better representation
    pretrained: bool = True             # ImageNet pretrained backbone

    # Projection head
    embed_dim: int = 256                # Shared embedding dimensionality
    projection_hidden: int = 1024       # Increased from 512. Wider MLP improves contrastive learning (SimCLR trick)

    # Self-Attention Transformer
    use_attention: bool = True           # Enable Self-Attention blocks after backbone
    attn_dim: int = 512                  # Internal attention dimension (projected from backbone)
    attn_heads: int = 8                  # Number of attention heads
    attn_layers: int = 2                 # Number of Transformer encoder layers

    # Contrastive loss
    temperature: float = 0.07           # NT-Xent temperature (learnable if learn_temperature=True)
    learn_temperature: bool = True      # Enabled (like CLIP) to dynamically scale hard/easy negatives


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    # Data
    dataset: str = "both"               # sen12 | qxs | both
    data_root: str = "data/sen12/raw"
    qxs_root: str = "data/QXSLAB_SAROPT/QXSLAB_SAROPT"
    terrains: list[str] = field(default_factory=lambda: ["agri", "barrenland", "grassland", "urban"])
    num_workers: int = 16
    pin_memory: bool = True

    # Image preprocessing
    image_size: int = 224               # Resize/crop to this size
    augment: bool = True                # Enable data augmentation

    # Optimization
    batch_size: int = 512
    epochs: int = 50
    lr: float = 2e-4
    weight_decay: float = 5e-2          # Increased from 1e-2 to combat overfitting
    warmup_epochs: int = 5
    min_lr: float = 1e-7

    # Terrain classification (multi-task)
    use_terrain_head: bool = True       # Train terrain classifier alongside contrastive
    terrain_loss_weight: float = 0.1    # Lowered from 0.3 so it doesn't overpower the contrastive alignment

    # Checkpointing
    checkpoint_dir: str = "checkpoints/fusion/v3"
    save_every: int = 1                 # Save checkpoint every N epochs
    resume_from: str | None = None      # Path to checkpoint to resume from

    # Logging
    log_every: int = 20                 # Print metrics every N steps
    eval_every: int = 1                 # Evaluate on val set every N epochs

    # Device
    device: str = "auto"               # auto | cuda | cpu
    use_amp: bool = True                # Mixed-precision training (halves VRAM usage)

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
