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
    # Freeze BERT text encoder (saves ~30-40% backward pass time & VRAM)
    freeze_text_encoder: bool = True

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
    # Auto-extraction destination
    extracted_image_dir: str = "data/vrsbench/extracted_images"
    # Auto-extract downloaded zip to disk once to avoid on-the-fly zip decompression
    auto_extract_zip: bool = True
    # Files downloaded by hand. Train and validation use different files, so
    # each split has its own; anything left None is fetched from the Hub.
    image_zip: str | None = None           # Images_train.zip, read without unpacking
    val_image_zip: str | None = None       # Images_val.zip
    annotations_file: str | None = None    # VRSBench_train.json
    val_annotations_file: str | None = None  # VRSBench_EVAL_referring.json
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
    # Saturated batch size for 20GB VRAM (RTX A4000)
    batch_size: int = 8
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
    # Evaluate on val set every N epochs (avoids 30+ min eval delay each epoch)
    eval_every: int = 5
    # Subsample val set during intermediate epochs for fast verification
    eval_max_samples: int | None = 1000

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


@dataclass
class RerankConfig:
    """Second-stage candidate re-ranker (see ml/grounding/rerank.py).

    GroundingDINO finds the referred object among its top-10 boxes ~85% of the
    time but ranks it first only ~50% of the time, so a model that compares the
    candidates against the full expression recovers most of that gap.
    """

    # Candidate extraction from the frozen GroundingDINO
    cache_k: int = 20                   # Candidates stored per expression
    num_candidates: int = 10            # Candidates the re-ranker chooses among
    nms_iou: float = 0.5                # Collapse near-duplicate queries
    max_text_tokens: int = 48           # Text features stored per expression
    cache_dir: str = "data/vrsbench/rerank_cache"

    # Architecture
    d_model: int = 256
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    # Off by default: it helps on held-out train images but hurts Acc@0.7 on
    # the VRSBench eval file, whose boxes are drawn a little differently.
    refine_boxes: bool = False          # Also regress a correction to the chosen box
    vocab_size: int = 30522             # GroundingDINO's BERT tokenizer
    word_dim: int | None = None         # Word-embedding width; 768 allows BERT init
    init_words_from_bert: bool = False  # Start word embeddings from BERT's vocabulary
    text_layers: int = 0                # Self-attention layers over the expression

    # Optimization (on cached features, so an epoch takes seconds)
    epochs: int = 40
    batch_size: int = 256
    lr: float = 4e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 1
    min_iou: float = 0.5                # Candidate must reach this IoU to be a target
    flip_prob: float = 0.5              # Mirror boxes + swap left/right, top/bottom words
    dev_fraction: float = 0.05          # Train images held out for model selection
    checkpoint: str = "checkpoints/grounding/reranker.pt"
