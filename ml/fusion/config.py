"""
Configuration for the Optical-SAR Fusion model and training.

Supports dynamic hardware auto-scaling (laptop -> RTX 5090 -> Multi-GPU Cloud)
as well as environment variable and CLI argument overrides.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_int(key: str, default: int) -> int:
    val = os.getenv(key)
    return int(val) if val is not None else default


def _env_float(key: str, default: float) -> float:
    val = os.getenv(key)
    return float(val) if val is not None else default


def _env_bool(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class ModelConfig:
    """Architecture hyperparameters."""

    # Backbone
    backbone: str = os.getenv("SATQUERY_BACKBONE", "convnext_tiny")
    pretrained: bool = _env_bool("SATQUERY_PRETRAINED", True)
    sar_in_channels: int = _env_int("SATQUERY_SAR_CHANNELS", 1)

    # Projection head
    embed_dim: int = _env_int("SATQUERY_EMBED_DIM", 256)
    projection_hidden: int = _env_int("SATQUERY_PROJ_HIDDEN", 1024)

    # Self-Attention Transformer
    use_attention: bool = _env_bool("SATQUERY_USE_ATTENTION", False)
    attn_dim: int = _env_int("SATQUERY_ATTN_DIM", 512)
    attn_heads: int = _env_int("SATQUERY_ATTN_HEADS", 8)
    attn_layers: int = _env_int("SATQUERY_ATTN_LAYERS", 2)

    # Cross-Modal Gating in TerrainClassifier
    use_cross_gate: bool = _env_bool("SATQUERY_USE_CROSS_GATE", True)

    # Contrastive loss & Hard-Negative Mining
    temperature: float = _env_float("SATQUERY_TEMPERATURE", 0.07)
    learn_temperature: bool = _env_bool("SATQUERY_LEARN_TEMP", True)
    hard_negative_weight: float = _env_float("SATQUERY_HARD_NEG_WEIGHT", 1.25)


@dataclass
class TrainConfig:
    """Training hyperparameters with dynamic hardware & CLI overrides."""

    # Data
    dataset: str = os.getenv("SATQUERY_DATASET", "both")  # sen12 | qxs | both
    data_root: str = os.getenv("SATQUERY_DATA_ROOT", "data/sen12/raw")
    qxs_root: str = os.getenv("SATQUERY_QXS_ROOT", "data/QXSLAB_SAROPT/QXSLAB_SAROPT")
    terrains: list[str] = field(
        default_factory=lambda: ["agri", "barrenland", "grassland", "urban", "water"]
    )
    num_workers: int = _env_int("SATQUERY_NUM_WORKERS", 16)
    pin_memory: bool = _env_bool("SATQUERY_PIN_MEMORY", True)

    # Image & SAR Physics preprocessing
    image_size: int = _env_int("SATQUERY_IMAGE_SIZE", 224)
    augment: bool = _env_bool("SATQUERY_AUGMENT", True)
    apply_lee: bool = _env_bool("SATQUERY_APPLY_LEE", False)

    # Optimization
    batch_size: int = _env_int("SATQUERY_BATCH_SIZE", 512)
    epochs: int = _env_int("SATQUERY_EPOCHS", 50)
    lr: float = _env_float("SATQUERY_LR", 2e-4)
    weight_decay: float = _env_float("SATQUERY_WEIGHT_DECAY", 5e-2)
    warmup_epochs: int = _env_int("SATQUERY_WARMUP_EPOCHS", 5)
    min_lr: float = _env_float("SATQUERY_MIN_LR", 1e-7)

    # Multi-task Auxiliary Objectives (Terrain + 2D Fourier Phase Alignment)
    use_terrain_head: bool = _env_bool("SATQUERY_USE_TERRAIN_HEAD", True)
    terrain_loss_weight: float = _env_float("SATQUERY_TERRAIN_WEIGHT", 0.1)
    phase_loss_weight: float = _env_float("SATQUERY_PHASE_WEIGHT", 0.05)

    # Checkpointing
    checkpoint_dir: str = os.getenv("SATQUERY_CKPT_DIR", "checkpoints/fusion/v3")
    save_every: int = _env_int("SATQUERY_SAVE_EVERY", 1)
    resume_from: str | None = os.getenv("SATQUERY_RESUME_FROM", None)

    # Logging
    log_every: int = _env_int("SATQUERY_LOG_EVERY", 20)
    eval_every: int = _env_int("SATQUERY_EVAL_EVERY", 1)

    # Device
    device: str = os.getenv("SATQUERY_DEVICE", "auto")
    use_amp: bool = _env_bool("SATQUERY_USE_AMP", True)

    @property
    def checkpoint_path(self) -> Path:
        return Path(self.checkpoint_dir)

    def resolve_device(self) -> str:
        """Determine the actual device to use."""
        if self.device == "auto":
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return self.device

    def auto_scale_hardware(self) -> None:
        """
        Inspects GPU VRAM (e.g., 32GB on RTX 5090 vs 24GB on 4090 vs 6GB laptop)
        and CPU core count to set optimal batch_size and num_workers if not overridden.
        """
        import torch

        if "SATQUERY_NUM_WORKERS" not in os.environ:
            cpu_count = os.cpu_count() or 8
            self.num_workers = min(16, max(2, cpu_count - 2))

        if "SATQUERY_BATCH_SIZE" not in os.environ and torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            if vram_gb >= 28.0:       # RTX 5090 (32GB) / A100 (40GB/80GB)
                self.batch_size = 512
            elif vram_gb >= 20.0:     # RTX 3090 / 4090 (24GB) / L4
                self.batch_size = 384
            elif vram_gb >= 11.0:     # RTX 3080 / 4070 Ti / T4 (12-16GB)
                self.batch_size = 192
            else:                     # 6-8GB GPUs
                self.batch_size = 64