"""
Configuration for the Siamese Change Detection / Change-VQA specialist model.

All architectural and training hyperparameters are centralized here for
reproducibility, ease of hyperparameter sweeps, and clean separation of concerns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    """Architecture hyperparameters for Siamese Vision Encoder + VLM Head."""

    # Vision backbone (shared weights for T1 and T2 images)
    backbone: str = "resnet18"          # resnet18 | resnet34 | resnet50
    pretrained: bool = True             # ImageNet pretrained backbone
    in_channels: int = 3                # Number of channels (3 for RGB optical)

    # Feature dimensions
    visual_feature_dim: int = 256       # Projected visual feature dimension per token
    spatial_token_resolution: int = 7   # Spatial grid resolution (7x7 = 49 visual tokens)

    # Text encoder & VLM cross-modal fusion
    vocab_size: int = 2000              # Max vocabulary size for question/answer tokenizer
    max_question_length: int = 32       # Max token length for query questions
    text_embed_dim: int = 256           # Text token embedding dimension
    num_cross_attention_heads: int = 8  # Number of attention heads in VLM fusion
    cross_attention_layers: int = 2     # Number of cross-modal transformer layers
    feedforward_dim: int = 512          # Dimension of feedforward network in VLM
    dropout: float = 0.1                # Dropout probability

    # Change segmentation mask head
    mask_hidden_dim: int = 128          # Hidden channels in change mask decoder
    mask_threshold: float = 0.5         # Probability threshold for binary change mask

    # Answer classification & generation
    num_classes: int = 64               # Number of canonical answer categories in CDVQA
    answer_hidden_dim: int = 256        # Hidden layer in VQA classification head


@dataclass
class TrainConfig:
    """Training hyperparameters for multi-task Change-VQA."""

    # Data
    data_root: str = "data/cdvqa"
    split_train: str = "train"
    split_val: str = "val"
    num_workers: int = 2
    pin_memory: bool = True

    # Image preprocessing
    image_size: int = 256               # Input image resolution (H=W=256)
    augment: bool = True                # Enable paired bi-temporal data augmentations

    # Optimization
    batch_size: int = 16
    epochs: int = 30
    lr: float = 1e-4
    backbone_lr: float = 2e-5           # Lower learning rate for pretrained backbone
    weight_decay: float = 1e-4
    warmup_epochs: int = 3
    min_lr: float = 1e-6

    # Multi-task loss weights
    vqa_loss_weight: float = 1.0        # Cross-entropy loss weight for VQA answers
    mask_bce_weight: float = 0.5        # Binary cross entropy loss weight for change mask
    mask_dice_weight: float = 0.5       # Soft Dice loss weight for change mask

    # Checkpointing
    checkpoint_dir: str = "checkpoints/change_detection"
    save_every: int = 5                 # Save checkpoint every N epochs
    resume_from: str | None = None      # Path to checkpoint to resume from

    # Logging & Evaluation
    log_every: int = 10                 # Print metrics every N steps
    eval_every: int = 1                 # Evaluate on val set every N epochs

    # Device
    device: str = "auto"               # auto | cuda | cpu

    @property
    def checkpoint_path(self) -> Path:
        return Path(self.checkpoint_dir)

    def resolve_device(self) -> str:
        """Determine the actual torch device to use."""
        if self.device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    return "mps"
            except ImportError:
                pass
            return "cpu"
        return self.device
