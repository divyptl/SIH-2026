"""
Training script for GroundingDINO fine-tuning on VRSBench.

Fine-tunes a pre-trained GroundingDINO model on the VRSBench grounding
subset for remote-sensing text-guided region grounding.

Usage:
    # From project root, using the ml venv:
    python -m ml.grounding.train

    # With overrides:
    python -m ml.grounding.train --epochs 30 --batch-size 2 --lr 5e-6

    # Resume from checkpoint:
    python -m ml.grounding.train --resume checkpoints/grounding/epoch_10.pt

    # Reuse images you already extracted instead of downloading the archives:
    python -m ml.grounding.train --image-dir /data/VRSBench/Images_train

    # VRSBench + DIOR-RSVG, starting from an existing detector's weights with a
    # fresh learning-rate schedule (--resume would continue the old schedule):
    python -m ml.grounding.train --datasets vrsbench dior_rsvg \
        --init-from checkpoints/grounding/kaggle/last.pt --epochs 6

    # Several GPUs on one machine (e.g. Kaggle 2x T4): one process per GPU.
    # --batch-size is per GPU; the effective batch is
    # batch-size x grad-accum x number of GPUs.
    torchrun --nproc_per_node=2 -m ml.grounding.train --batch-size 4 --grad-accum 4
"""

from __future__ import annotations

import argparse
import builtins
import json
import math
import platform
import sys
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import os
from pathlib import Path

# Set Hugging Face cache directory to the project's data folder to avoid C drive
HF_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "hf_cache"
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HF_DATASETS_CACHE", str(HF_CACHE_DIR / "datasets"))
# Reduces allocator fragmentation on long runs. Must be set before the CUDA
# allocator initialises. Not supported on Windows, where setting it only emits
# a warning on every run.
if platform.system() != "Windows":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

# Add project root to path
# Windows consoles default to cp1252, which cannot encode the box-drawing and
# arrow characters in this module's progress output.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.grounding.config import ModelConfig, TrainConfig
from ml.grounding.dataset import collate_fn
from ml.grounding.model import GroundingModel
from ml.grounding.sources import DATASETS, load_split, load_training_set
from ml.grounding.transforms import GroundingAugmentation, prepare_training_batch


# ── Learning rate scheduler ─────────────────────────────────────────────

def get_cosine_schedule(optimizer, warmup_epochs, total_epochs, min_lr, steps_per_epoch):
    """Cosine annealing with linear warmup."""
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(min_lr / optimizer.defaults["lr"], 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Loss weighting ──────────────────────────────────────────────────────

def weighted_loss(
    loss_dict: dict, weights: dict[str, float],
) -> torch.Tensor | None:
    """Recombine GroundingDINO's loss terms under our own weights.

    HF hardcodes the weights inside `loss_grounding_dino`, so the only way to
    reweight is to rebuild the total from `loss_dict` — whose entries are still
    attached to the graph. Returns None if nothing matched, so callers can fall
    back to the model's own total.
    """
    total = None
    for key, weight in weights.items():
        term = loss_dict.get(key)
        if term is None or weight == 0 or not isinstance(term, torch.Tensor):
            continue
        scaled = term * weight
        total = scaled if total is None else total + scaled
    return total


def term(loss_dict: dict, key: str) -> float:
    """Read one loss term out of the dict as a plain float."""
    value = loss_dict.get(key, 0.0)
    return value.item() if isinstance(value, torch.Tensor) else float(value)


# ── Mixed precision ─────────────────────────────────────────────────────

def resolve_amp(device: str, enabled: bool) -> tuple[bool, torch.dtype]:
    """Pick the autocast dtype for this device.

    bf16 is preferred over fp16: it has the same exponent range as fp32, so it
    needs no loss scaling — which matters here because GroundingDINO's encoder
    loss runs to five figures and would overflow fp16.

    Emulated bf16 is excluded: pre-Ampere GPUs such as Colab's T4 report bf16 as
    supported but run it without native kernels, far slower than fp16.
    """
    if not enabled or device == "cpu":
        return False, torch.float32
    if device == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False):
        return True, torch.bfloat16
    return True, torch.float16


# ── Multi-GPU ───────────────────────────────────────────────────────────

def setup_distributed() -> tuple[bool, int, int]:
    """Join the process group when launched by torchrun.

    Returns (distributed, rank, world_size). Each process drives the GPU
    matching its LOCAL_RANK; it is made the current device, so the rest of the
    script can keep addressing it as plain "cuda".
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 1
    local_rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank % torch.cuda.device_count())
    # NCCL is the fast GPU backend but is Linux-only; gloo lets the same code
    # run (slowly) on Windows for testing.
    backend = "gloo" if platform.system() == "Windows" else "nccl"
    # Rank 0 alone downloads data, evaluates and saves checkpoints while the
    # others wait, which can take far longer than the default 10-minute timeout.
    dist.init_process_group(backend, timeout=timedelta(hours=3))
    return True, dist.get_rank(), world_size


def barrier(distributed: bool) -> None:
    if distributed:
        dist.barrier()


def average_across_ranks(metrics: dict[str, float], distributed: bool) -> dict[str, float]:
    """Mean of each metric over all processes, so logs reflect the full batch."""
    if not distributed:
        return metrics
    keys = sorted(metrics)
    device = "cuda" if dist.get_backend() == "nccl" else "cpu"   # gloo reduces on CPU
    values = torch.tensor([metrics[k] for k in keys], device=device, dtype=torch.float64)
    dist.all_reduce(values)
    values /= dist.get_world_size()
    return dict(zip(keys, values.tolist()))


# ── Training loop ────────────────────────────────────────────────────────

def train_one_epoch(
    grounding: GroundingModel,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    augmentation: GroundingAugmentation,
    device: str,
    epoch: int,
    cfg: TrainConfig,
    scaler: torch.amp.GradScaler | None = None,
    amp_dtype: torch.dtype = torch.float32,
    ddp_model: DistributedDataParallel | None = None,
) -> dict[str, float]:
    """Train for one epoch. Returns average metrics.

    With `ddp_model`, the forward pass goes through the DistributedDataParallel
    wrapper so gradients are averaged across GPUs.
    """
    grounding.model.train()

    use_amp = amp_dtype != torch.float32
    accum = max(cfg.grad_accum_steps, 1)
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    total_loss_ce = 0.0
    total_loss_bbox = 0.0
    total_loss_giou = 0.0
    num_batches = 0

    forward = ddp_model if ddp_model is not None else grounding

    for step, batch in enumerate(train_loader):
        # Prepare batch (augmentation + processor)
        inputs, labels = prepare_training_batch(
            batch, grounding.processor, augmentation, device, cfg.image_size,
        )
        stepping = (step + 1) % accum == 0 or (step + 1) == len(train_loader)

        # Across GPUs, gradients only need averaging on the batch that ends an
        # accumulation window; syncing on every batch just adds traffic.
        sync = ddp_model.no_sync() if ddp_model is not None and not stepping else nullcontext()
        with sync:
            # Forward pass under autocast — GroundingDINO returns losses when labels are provided
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                outputs = forward(
                    pixel_values=inputs["pixel_values"],
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    token_type_ids=inputs.get("token_type_ids"),
                    labels=labels,
                )

            loss_dict = outputs.loss_dict or {}
            loss = weighted_loss(loss_dict, cfg.loss_weights)
            if loss is None:
                loss = outputs.loss

            # Scaled backward pass (fp16 only). Divide by accum so accumulated
            # grads average rather than sum.
            if scaler is not None:
                scaler.scale(loss / accum).backward()
            else:
                (loss / accum).backward()

        # Step only once per accumulation window
        if stepping:
            if scaler is not None:
                scaler.unscale_(optimizer)

            # Gradient clipping (GroundingDINO benefits from tight clipping)
            torch.nn.utils.clip_grad_norm_(
                grounding.model.parameters(), max_norm=cfg.max_grad_norm,
            )

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        # Accumulate metrics
        total_loss += loss.item()
        total_loss_ce += term(loss_dict, "loss_ce")
        total_loss_bbox += term(loss_dict, "loss_bbox")
        total_loss_giou += term(loss_dict, "loss_giou")
        num_batches += 1

        # Logging
        if (step + 1) % cfg.log_every == 0:
            lr = optimizer.param_groups[0]["lr"]
            avg_loss = total_loss / num_batches
            print(
                f"  [{epoch}][{step+1}/{len(train_loader)}] "
                f"loss={loss.item():.4f}  "
                f"avg_loss={avg_loss:.4f}  "
                f"lr={lr:.2e}"
            )

    n = max(num_batches, 1)
    return {
        "loss": total_loss / n,
        "loss_ce": total_loss_ce / n,
        "loss_bbox": total_loss_bbox / n,
        "loss_giou": total_loss_giou / n,
    }


@torch.no_grad()
def evaluate(
    grounding: GroundingModel,
    val_loader: DataLoader,
    device: str,
    image_size: int | None = None,
    amp_dtype: torch.dtype = torch.float32,
    loss_weights: dict[str, float] | None = None,
    max_samples: int | None = None,
) -> dict[str, float]:
    """Evaluate on the validation set. Returns average loss metrics."""
    grounding.model.eval()

    total_loss = 0.0
    total_loss_ce = 0.0
    total_loss_bbox = 0.0
    total_loss_giou = 0.0
    num_batches = 0
    samples_seen = 0

    for batch in val_loader:
        inputs, labels = prepare_training_batch(
            batch, grounding.processor, augmentation=None, device=device,
            image_size=image_size,
        )

        with torch.autocast(device_type=device, dtype=amp_dtype,
                            enabled=amp_dtype != torch.float32):
            outputs = grounding(
                pixel_values=inputs["pixel_values"],
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                token_type_ids=inputs.get("token_type_ids"),
                labels=labels,
            )

        loss_dict = outputs.loss_dict or {}
        loss = weighted_loss(loss_dict, loss_weights) if loss_weights else None
        total_loss += (loss if loss is not None else outputs.loss).item()
        total_loss_ce += term(loss_dict, "loss_ce")
        total_loss_bbox += term(loss_dict, "loss_bbox")
        total_loss_giou += term(loss_dict, "loss_giou")
        num_batches += 1
        samples_seen += len(batch["images"])

        if max_samples is not None and samples_seen >= max_samples:
            break

    n = max(num_batches, 1)
    return {
        "val_loss": total_loss / n,
        "val_loss_ce": total_loss_ce / n,
        "val_loss_bbox": total_loss_bbox / n,
        "val_loss_giou": total_loss_giou / n,
    }


# ── Checkpointing ───────────────────────────────────────────────────────

def save_checkpoint(
    path: Path,
    epoch: int,
    grounding: GroundingModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    metrics: dict,
    model_cfg: ModelConfig,
    scaler: torch.amp.GradScaler | None = None,
    history: list[dict] | None = None,
    best_val_loss: float | None = None,
) -> None:
    """Save a training checkpoint.

    `history` and `best_val_loss` travel with the checkpoint so that a run
    resumed in a new session (e.g. the next Kaggle session) keeps its full
    history and does not overwrite best.pt with a worse epoch.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model": grounding.model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "metrics": metrics,
        "model_config": {
            "model_id": model_cfg.model_id,
            "freeze_backbone": model_cfg.freeze_backbone,
            "freeze_text_encoder": model_cfg.freeze_text_encoder,
            "box_threshold": model_cfg.box_threshold,
            "text_threshold": model_cfg.text_threshold,
        },
    }
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    if history is not None:
        state["history"] = history
    if best_val_loss is not None:
        state["best_val_loss"] = best_val_loss
    torch.save(state, path)
    print(f"  Checkpoint saved: {path}")


def load_checkpoint(
    path: str,
    grounding: GroundingModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: str,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[int, list[dict], float]:
    """Load a checkpoint. Returns (last completed epoch, history, best val loss)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    grounding.model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    print(f"  Resumed from checkpoint: {path} (epoch {ckpt['epoch']})")
    return ckpt["epoch"], ckpt.get("history", []), ckpt.get("best_val_loss", float("inf"))


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    distributed, rank, world_size = setup_distributed()
    is_main = rank == 0
    if not is_main:
        # One set of logs is enough; errors still reach stderr.
        builtins.print = lambda *args, **kwargs: None

    # High-performance hardware acceleration (TF32 on Ampere/Ada like RTX A4000, cuDNN benchmark)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser(description="Fine-tune GroundingDINO on VRSBench")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--data-name", type=str, default=None)
    parser.add_argument("--image-dir", type=str, default=None,
                        help="Directory of extracted VRSBench images; avoids "
                             "downloading the multi-GB image archives")
    parser.add_argument("--extracted-image-dir", type=str, default=None,
                        help="Directory to extract/locate uncompressed VRSBench images")
    parser.add_argument("--no-extract-zip", action="store_true",
                        help="Do not auto-extract image zip files to disk")
    parser.add_argument("--image-zip", type=str, default=None,
                        help="Images_train.zip you downloaded yourself; read "
                             "directly, no unpacking needed")
    parser.add_argument("--val-image-zip", type=str, default=None,
                        help="Images_val.zip you downloaded yourself")
    parser.add_argument("--annotations", type=str, default=None,
                        help="VRSBench_train.json you downloaded yourself; "
                             "skips the annotation download for the train split")
    parser.add_argument("--val-annotations", type=str, default=None,
                        help="VRSBench_EVAL_referring.json you downloaded "
                             "yourself; used for the validation split")
    parser.add_argument("--no-download-images", action="store_true",
                        help="Fail instead of downloading VRSBench image archives")
    parser.add_argument("--datasets", nargs="+", default=["vrsbench"], choices=DATASETS,
                        help="Training data. Several are concatenated, with each one's "
                             "photos that sit in the other benchmark's test split removed "
                             "(see ml/grounding/sources.py). Validation stays on VRSBench.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Continue an interrupted run: weights, optimizer, LR "
                             "schedule and epoch counter")
    parser.add_argument("--init-from", type=str, default=None,
                        help="Start from a checkpoint's weights only, with a fresh "
                             "optimizer and LR schedule (e.g. to fine-tune on new data)")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Where checkpoints and history.json go "
                             "(default checkpoints/grounding)")
    parser.add_argument("--save-every", type=int, default=None,
                        help="Keep a numbered epoch_N.pt every N epochs (default 5)")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--no-freeze-backbone", action="store_true",
                        help="Do NOT freeze the vision backbone")
    parser.add_argument("--freeze-text", action="store_true",
                        help="Freeze the text encoder (default: True)")
    parser.add_argument("--unfreeze-text", action="store_true",
                        help="Unfreeze the text encoder to train BERT")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable training augmentations")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable mixed precision (use full fp32)")
    parser.add_argument("--grad-accum", type=int, default=None,
                        help="Accumulate gradients over N batches before stepping")
    parser.add_argument("--eval-every", type=int, default=None,
                        help="Run validation every N epochs (default 5)")
    parser.add_argument("--eval-max-samples", type=int, default=None,
                        help="Subsample validation set during intermediate epochs (default: 1000)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit dataset size (for debugging)")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    # Build config with CLI overrides
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()

    if args.model_id:
        model_cfg.model_id = args.model_id
    if args.no_freeze_backbone:
        model_cfg.freeze_backbone = False
    if args.freeze_text:
        model_cfg.freeze_text_encoder = True
    if args.unfreeze_text:
        model_cfg.freeze_text_encoder = False
    if args.epochs:
        train_cfg.epochs = args.epochs
    if args.batch_size:
        train_cfg.batch_size = args.batch_size
    if args.lr:
        train_cfg.lr = args.lr
    if args.image_size:
        train_cfg.image_size = args.image_size
    if args.data_name:
        train_cfg.data_name = args.data_name
    if args.image_dir:
        train_cfg.image_dir = args.image_dir
    if args.extracted_image_dir:
        train_cfg.extracted_image_dir = args.extracted_image_dir
    if args.no_extract_zip:
        train_cfg.auto_extract_zip = False
    if args.image_zip:
        train_cfg.image_zip = args.image_zip
    if args.val_image_zip:
        train_cfg.val_image_zip = args.val_image_zip
    if args.annotations:
        train_cfg.annotations_file = args.annotations
    if args.val_annotations:
        train_cfg.val_annotations_file = args.val_annotations
    if args.no_download_images:
        train_cfg.download_images = False
    if args.resume:
        train_cfg.resume_from = args.resume
    if args.checkpoint_dir:
        train_cfg.checkpoint_dir = args.checkpoint_dir
    if args.save_every:
        train_cfg.save_every = args.save_every
    if args.no_augment:
        train_cfg.augment = False
    if args.no_amp:
        train_cfg.amp = False
    if args.grad_accum:
        train_cfg.grad_accum_steps = args.grad_accum
    if args.eval_every:
        train_cfg.eval_every = args.eval_every
    if args.eval_max_samples is not None:
        train_cfg.eval_max_samples = args.eval_max_samples
    if args.num_workers is not None:
        train_cfg.num_workers = args.num_workers
    if args.device:
        train_cfg.device = args.device

    device = train_cfg.resolve_device()
    use_amp, amp_dtype = resolve_amp(device, train_cfg.amp)

    # pin_memory only does anything with an accelerator; leaving it on for a
    # CPU run just emits a warning every epoch.
    if device == "cpu":
        train_cfg.pin_memory = False

    print("=" * 70)
    print("  GroundingDINO Fine-Tuning on VRSBench")
    print("=" * 70)
    print(f"  Model:           {model_cfg.model_id}")
    print(f"  Freeze backbone: {model_cfg.freeze_backbone}")
    print(f"  Freeze text enc: {model_cfg.freeze_text_encoder}")
    print(f"  Batch size:      {train_cfg.batch_size} per GPU x {world_size} GPU(s)")
    print(f"  Epochs:          {train_cfg.epochs}")
    print(f"  LR:              {train_cfg.lr}")
    print(f"  Device:          {device}")
    print(f"  Datasets:        {', '.join(args.datasets)} (validation: vrsbench)")
    print(f"  Workers:         {train_cfg.num_workers}")
    print(f"  Augmentation:    {train_cfg.augment}")
    print(f"  Mixed precision: {amp_dtype if use_amp else 'off (fp32)'}")
    print(f"  Grad accum:      {train_cfg.grad_accum_steps} (effective batch "
          f"{train_cfg.batch_size * train_cfg.grad_accum_steps * world_size})")
    print(f"  Eval cadence:    Every {train_cfg.eval_every} epochs "
          f"(intermediate cap: {train_cfg.eval_max_samples or 'all'})")
    print("=" * 70)

    # ── Data ──
    print("\nLoading datasets...")
    # Rank 0 downloads and extracts VRSBench; the others wait, then load the
    # files it wrote instead of racing it for the same archives.
    if not is_main:
        barrier(distributed)
    train_ds = load_training_set(args.datasets, train_cfg, max_samples=args.max_samples)
    val_ds = load_split(
        "vrsbench", "validation", train_cfg,
        max_samples=args.max_samples // 5 if args.max_samples else None,
    )

    if is_main:
        barrier(distributed)

    print(f"  Train: {len(train_ds):,} samples")
    print(f"  Val:   {len(val_ds):,} samples")

    loader_kwargs = {
        "batch_size": train_cfg.batch_size,
        "num_workers": train_cfg.num_workers,
        "pin_memory": train_cfg.pin_memory,
        "collate_fn": collate_fn,
    }
    if train_cfg.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    # Each GPU trains on its own disjoint share of every epoch
    train_sampler = (
        DistributedSampler(train_ds, shuffle=True, drop_last=True) if distributed else None
    )
    train_loader = DataLoader(
        train_ds,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=True,
        **loader_kwargs,
    )

    val_workers = min(2, train_cfg.num_workers)
    val_loader_kwargs = {
        "batch_size": train_cfg.batch_size,
        "num_workers": val_workers,
        "pin_memory": train_cfg.pin_memory,
        "collate_fn": collate_fn,
    }
    if val_workers > 0:
        val_loader_kwargs["persistent_workers"] = True
        val_loader_kwargs["prefetch_factor"] = 2

    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        **val_loader_kwargs,
    )

    # ── Model ──
    print("\nLoading pre-trained GroundingDINO...")
    if not is_main:
        barrier(distributed)
    grounding = GroundingModel(model_cfg)
    if is_main:
        barrier(distributed)
    grounding.model.to(device)

    total_params = grounding.get_total_params() / 1e6
    trainable_params = grounding.get_trainable_params() / 1e6
    print(f"  Total params:     {total_params:.1f}M")
    print(f"  Trainable params: {trainable_params:.1f}M")
    print(f"  Frozen params:    {total_params - trainable_params:.1f}M")

    # ── Optimizer ──
    # Use different LR for backbone vs decoder (if backbone is unfrozen)
    if not model_cfg.freeze_backbone:
        backbone_params = []
        other_params = []
        for name, param in grounding.model.named_parameters():
            if not param.requires_grad:
                continue
            if "backbone" in name or "input_proj" in name:
                backbone_params.append(param)
            else:
                other_params.append(param)
        param_groups = [
            {"params": other_params, "lr": train_cfg.lr},
            {"params": backbone_params, "lr": train_cfg.backbone_lr},
        ]
    else:
        param_groups = [
            {"params": [p for p in grounding.model.parameters() if p.requires_grad]},
        ]

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )

    # The scheduler advances once per optimizer step, not once per batch. The
    # loop also steps on a partial final window, so round up — rounding down
    # runs the cosine past its end, where the LR climbs back up.
    accum = max(train_cfg.grad_accum_steps, 1)
    steps_per_epoch = max(math.ceil(len(train_loader) / accum), 1)
    scheduler = get_cosine_schedule(
        optimizer,
        train_cfg.warmup_epochs,
        train_cfg.epochs,
        train_cfg.min_lr,
        steps_per_epoch,
    )

    # ── Mixed precision ──
    # Create the grad scaler before the training loop. fp16 needs loss
    # scaling to avoid gradient underflow; bf16 does not.
    scaler = (
        torch.amp.GradScaler(device)
        if use_amp and amp_dtype == torch.float16
        else None
    )

    # ── Augmentation ──
    augmentation = GroundingAugmentation(augment=train_cfg.augment)

    # ── Resume ──
    start_epoch = 1
    best_val_loss = float("inf")
    history: list[dict] = []
    if train_cfg.resume_from and args.init_from:
        parser.error("--resume and --init-from are exclusive: resume continues a run, "
                     "init-from starts a new one from its weights")
    if train_cfg.resume_from:
        last_epoch, history, best_val_loss = load_checkpoint(
            train_cfg.resume_from, grounding, optimizer, scheduler, device, scaler,
        )
        start_epoch = last_epoch + 1
    elif args.init_from:
        ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        grounding.model.load_state_dict(ckpt["model"])
        print(f"  Initialized weights from {args.init_from} (epoch {ckpt.get('epoch', '?')}); "
              f"optimizer and LR schedule start fresh")
        del ckpt

    # ── Multi-GPU ──
    # Wrapped after resuming, so every process starts from the same weights.
    # find_unused_parameters: heads whose loss is weighted to zero (loss_ce_enc)
    # get no gradient, which DDP otherwise treats as an error.
    ddp_model = None
    if distributed:
        ddp_model = DistributedDataParallel(
            grounding.model,
            device_ids=[torch.cuda.current_device()] if device == "cuda" else None,
            find_unused_parameters=True,
        )

    # ── Training loop ──
    print(f"\nStarting training from epoch {start_epoch}...\n")

    for epoch in range(start_epoch, train_cfg.epochs + 1):
        t0 = time.time()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)   # a different shuffle every epoch

        # Train
        train_metrics = train_one_epoch(
            grounding, train_loader, optimizer, scheduler,
            augmentation, device, epoch, train_cfg, scaler, amp_dtype, ddp_model,
        )
        train_metrics = average_across_ranks(train_metrics, distributed)

        # Evaluation, checkpoints and logs are rank 0's job; the other ranks
        # wait at the barrier at the end of the epoch.
        if not is_main:
            barrier(distributed)
            continue

        # Evaluate
        val_metrics = {}
        is_final_epoch = (epoch == train_cfg.epochs)
        if epoch % train_cfg.eval_every == 0 or is_final_epoch:
            max_eval = None if is_final_epoch else train_cfg.eval_max_samples
            val_metrics = evaluate(
                grounding, val_loader, device, train_cfg.image_size, amp_dtype,
                train_cfg.loss_weights, max_samples=max_eval,
            )

        elapsed = time.time() - t0
        all_metrics = {**train_metrics, **val_metrics, "epoch": epoch, "time": elapsed}
        history.append(all_metrics)

        # Epoch summary
        val_str = ""
        if val_metrics:
            val_str = f"  val_loss={val_metrics['val_loss']:.4f}"
        print(
            f"Epoch {epoch}/{train_cfg.epochs}  "
            f"loss={train_metrics['loss']:.4f}  "
            f"ce={train_metrics['loss_ce']:.4f}  "
            f"bbox={train_metrics['loss_bbox']:.4f}  "
            f"giou={train_metrics['loss_giou']:.4f}"
            f"{val_str}  "
            f"[{elapsed:.1f}s]"
        )

        # Rolling checkpoint, overwritten every epoch, so an interrupted session
        # (e.g. a Colab disconnect) loses at most one epoch
        # Decide "best" first, so the checkpoints saved below all agree on it
        is_best = bool(val_metrics) and val_metrics["val_loss"] < best_val_loss
        if is_best:
            best_val_loss = val_metrics["val_loss"]
        extra = {"history": history, "best_val_loss": best_val_loss}

        save_checkpoint(
            train_cfg.checkpoint_path / "last.pt",
            epoch, grounding, optimizer, scheduler, all_metrics, model_cfg, scaler, **extra,
        )

        # Save checkpoint
        if epoch % train_cfg.save_every == 0:
            save_checkpoint(
                train_cfg.checkpoint_path / f"epoch_{epoch}.pt",
                epoch, grounding, optimizer, scheduler, all_metrics, model_cfg, scaler, **extra,
            )

        # Save best model
        if is_best:
            save_checkpoint(
                train_cfg.checkpoint_path / "best.pt",
                epoch, grounding, optimizer, scheduler, all_metrics, model_cfg, scaler, **extra,
            )
            print(f"  >> New best model! val_loss={best_val_loss:.4f}")

        # Written every epoch, so a session cut short still leaves the history
        history_path = train_cfg.checkpoint_path / "history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        barrier(distributed)

    # ── Save final model + history ──
    if is_main:
        save_checkpoint(
            train_cfg.checkpoint_path / "final.pt",
            train_cfg.epochs, grounding, optimizer, scheduler,
            history[-1] if history else {}, model_cfg, scaler,
            history=history, best_val_loss=best_val_loss,
        )

        history_path = train_cfg.checkpoint_path / "history.json"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"\nTraining history saved: {history_path}")
        print("Training complete.")

    barrier(distributed)
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
