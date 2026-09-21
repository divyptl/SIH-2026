"""
Configuration for the GroundingDINO grounding model and training.

All hyperparameters are centralized here so training runs are reproducible
and easy to sweep over.
"""

from __future__ import annotations
from ml.grounding import dataset

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    """Architecture and inference hyperparameters."""

    # Pre-trained model
    model_id: str = "IDEA-Research/grounding-dino-tiny"

    # Fine-tuning strategy
    freeze_backbone: bool = True        # Freeze Swin-T vision backbone
    freeze_text_encoder: bool = True    # Freeze BERT text encoder (saves ~30-40% backward pass time & VRAM)

    # Inference thresholds
    box_threshold: float = 0.25         # Min confidence to keep a predicted box
    text_threshold: float = 0.25        # Min text-grounding score


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    # Data
    data_name: str = "xiang709/VRSBench"   # HuggingFace repo holding VRSBench
    data_cache_dir: str = "data/vrsbench"  # Local cache dir for HF downloads
    image_dir: str | None = None           # Extracted VRSBench images; skips the
                                           # multi-GB archive download when set
    extracted_image_dir: str = "data/vrsbench/extracted_images"  # Auto-extraction destination
    auto_extract_zip: bool = True          # Auto-extract downloaded zip to disk once to avoid on-the-fly zip decompression
    download_images: bool = True           # Fetch Images_*.zip from the Hub
    num_workers: int = 6                   # High-throughput workers for i9 multi-core
    pin_memory: bool = True

    # Image preprocessing
    image_size: int = 512               # VRSBench tiles are natively 512x512;
                                        # the processor default (800) upscales
                                        # them and inflates activation memory

    # Augmentation
    augment: bool = True                # Enable training augmentations

    # Loss weighting. Applied to GroundingDINO's individual loss terms in place
    # of the hardcoded weights in HF's `loss_grounding_dino`.
    #
    # `loss_ce_enc` is zeroed. On this checkpoint the two-stage encoder's
    # classification logits are uncalibrated against the text tokens — they span
    # -140..+66 and score 65% of (query, token) pairs above 0.5, where the
    # decoder's span -8..0 and score 0.01% above. Focal loss over that many
    # confident-wrong predictions comes out ~5,000x the decoder's term, which at
    # HF's default weight of 2.0 is 99.98% of the total loss and leaves box
    # regression with no effective gradient. The encoder's box terms are kept.
    loss_weights: dict[str, float] = field(default_factory=lambda: {
        "loss_ce": 2.0,
        "loss_bbox": 5.0,
        "loss_giou": 2.0,
        "loss_ce_enc": 0.0,
        "loss_bbox_enc": 5.0,
        "loss_giou_enc": 2.0,
    })

    # Memory / throughput
    amp: bool = True                    # Mixed precision (bf16 on CUDA)
    grad_accum_steps: int = 2           # Optimizer step every N batches; raises
                                        # effective batch to batch_size * grad_accum_steps (16)

    # Optimization
    batch_size: int = 8                 # Saturated batch size for 20GB VRAM (RTX A4000)
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

    # Logging & Validation
    log_every: int = 50                 # Print metrics every N steps
    eval_every: int = 5                 # Evaluate on val set every N epochs (avoids 30+ min eval delay each epoch)
    eval_max_samples: int | None = 1000 # Subsample val set during intermediate epochs for fast verification

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
