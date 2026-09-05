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
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.grounding.config import ModelConfig, TrainConfig
from ml.grounding.dataset import VRSBenchGroundingDataset, collate_fn
from ml.grounding.model import GroundingModel
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
) -> dict[str, float]:
    """Train for one epoch. Returns average metrics."""
    grounding.model.train()

    total_loss = 0.0
    total_loss_ce = 0.0
    total_loss_bbox = 0.0
    total_loss_giou = 0.0
    num_batches = 0

    for step, batch in enumerate(train_loader):
        # Prepare batch (augmentation + processor)
        inputs, labels = prepare_training_batch(
            batch, grounding.processor, augmentation, device,
        )

        # Forward pass — GroundingDINO returns losses when labels are provided
        outputs = grounding(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            token_type_ids=inputs.get("token_type_ids"),
            labels=labels,
        )

        loss = outputs.loss

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (GroundingDINO benefits from tight clipping)
        torch.nn.utils.clip_grad_norm_(
            grounding.model.parameters(), max_norm=cfg.max_grad_norm,
        )

        optimizer.step()
        scheduler.step()

        # Accumulate metrics
        loss_dict = outputs.loss_dict if hasattr(outputs, "loss_dict") and outputs.loss_dict else {}
        total_loss += loss.item()
        total_loss_ce += loss_dict.get("loss_ce", torch.tensor(0.0)).item() if isinstance(loss_dict.get("loss_ce"), torch.Tensor) else loss_dict.get("loss_ce", 0.0)
        total_loss_bbox += loss_dict.get("loss_bbox", torch.tensor(0.0)).item() if isinstance(loss_dict.get("loss_bbox"), torch.Tensor) else loss_dict.get("loss_bbox", 0.0)
        total_loss_giou += loss_dict.get("loss_giou", torch.tensor(0.0)).item() if isinstance(loss_dict.get("loss_giou"), torch.Tensor) else loss_dict.get("loss_giou", 0.0)
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
) -> dict[str, float]:
    """Evaluate on the validation set. Returns average loss metrics."""
    grounding.model.eval()

    total_loss = 0.0
    total_loss_ce = 0.0
    total_loss_bbox = 0.0
    total_loss_giou = 0.0
    num_batches = 0

    for batch in val_loader:
        inputs, labels = prepare_training_batch(
            batch, grounding.processor, augmentation=None, device=device,
        )

        outputs = grounding(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            token_type_ids=inputs.get("token_type_ids"),
            labels=labels,
        )

        loss_dict = outputs.loss_dict if hasattr(outputs, "loss_dict") and outputs.loss_dict else {}
        total_loss += outputs.loss.item()
        total_loss_ce += loss_dict.get("loss_ce", torch.tensor(0.0)).item() if isinstance(loss_dict.get("loss_ce"), torch.Tensor) else loss_dict.get("loss_ce", 0.0)
        total_loss_bbox += loss_dict.get("loss_bbox", torch.tensor(0.0)).item() if isinstance(loss_dict.get("loss_bbox"), torch.Tensor) else loss_dict.get("loss_bbox", 0.0)
        total_loss_giou += loss_dict.get("loss_giou", torch.tensor(0.0)).item() if isinstance(loss_dict.get("loss_giou"), torch.Tensor) else loss_dict.get("loss_giou", 0.0)
        num_batches += 1

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
) -> None:
    """Save a training checkpoint."""
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
    torch.save(state, path)
    print(f"  Checkpoint saved: {path}")


def load_checkpoint(
    path: str,
    grounding: GroundingModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: str,
) -> int:
    """Load a checkpoint. Returns the epoch to resume from."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    grounding.model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    print(f"  Resumed from checkpoint: {path} (epoch {ckpt['epoch']})")
    return ckpt["epoch"]


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune GroundingDINO on VRSBench")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--data-name", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--no-freeze-backbone", action="store_true",
                        help="Do NOT freeze the vision backbone")
    parser.add_argument("--freeze-text", action="store_true",
                        help="Also freeze the text encoder")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable training augmentations")
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
    if args.resume:
        train_cfg.resume_from = args.resume
    if args.no_augment:
        train_cfg.augment = False
    if args.num_workers is not None:
        train_cfg.num_workers = args.num_workers
    elif platform.system() == "Windows":
        train_cfg.num_workers = 0
    if args.device:
        train_cfg.device = args.device

    device = train_cfg.resolve_device()

    print("=" * 70)
    print("  GroundingDINO Fine-Tuning on VRSBench")
    print("=" * 70)
    print(f"  Model:           {model_cfg.model_id}")
    print(f"  Freeze backbone: {model_cfg.freeze_backbone}")
    print(f"  Freeze text enc: {model_cfg.freeze_text_encoder}")
    print(f"  Batch size:      {train_cfg.batch_size}")
    print(f"  Epochs:          {train_cfg.epochs}")
    print(f"  LR:              {train_cfg.lr}")
    print(f"  Device:          {device}")
    print(f"  Dataset:         {train_cfg.data_name}")
    print(f"  Augmentation:    {train_cfg.augment}")
    print("=" * 70)

    # ── Data ──
    print("\nLoading datasets...")
    train_ds = VRSBenchGroundingDataset(
        data_name=train_cfg.data_name,
        data_subset=train_cfg.data_subset,
        split="train",
        cache_dir=train_cfg.data_cache_dir,
        max_samples=args.max_samples,
    )
    val_ds = VRSBenchGroundingDataset(
        data_name=train_cfg.data_name,
        data_subset=train_cfg.data_subset,
        split="validation",
        cache_dir=train_cfg.data_cache_dir,
        max_samples=args.max_samples // 5 if args.max_samples else None,
    )

    print(f"  Train: {len(train_ds):,} samples")
    print(f"  Val:   {len(val_ds):,} samples")

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        pin_memory=train_cfg.pin_memory,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        pin_memory=train_cfg.pin_memory,
        collate_fn=collate_fn,
    )

    # ── Model ──
    print("\nLoading pre-trained GroundingDINO...")
    grounding = GroundingModel(model_cfg)
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

    steps_per_epoch = len(train_loader)
    scheduler = get_cosine_schedule(
        optimizer,
        train_cfg.warmup_epochs,
        train_cfg.epochs,
        train_cfg.min_lr,
        steps_per_epoch,
    )

    # ── Augmentation ──
    augmentation = GroundingAugmentation(augment=train_cfg.augment)

    # ── Resume ──
    start_epoch = 1
    if train_cfg.resume_from:
        start_epoch = load_checkpoint(
            train_cfg.resume_from, grounding, optimizer, scheduler, device,
        ) + 1

    # ── Training loop ──
    print(f"\nStarting training from epoch {start_epoch}...\n")
    best_val_loss = float("inf")
    history: list[dict] = []

    for epoch in range(start_epoch, train_cfg.epochs + 1):
        t0 = time.time()

        # Train
        train_metrics = train_one_epoch(
            grounding, train_loader, optimizer, scheduler,
            augmentation, device, epoch, train_cfg,
        )

        # Evaluate
        val_metrics = {}
        if epoch % train_cfg.eval_every == 0:
            val_metrics = evaluate(grounding, val_loader, device)

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

        # Save checkpoint
        if epoch % train_cfg.save_every == 0:
            save_checkpoint(
                train_cfg.checkpoint_path / f"epoch_{epoch}.pt",
                epoch, grounding, optimizer, scheduler, all_metrics, model_cfg,
            )

        # Save best model
        if val_metrics and val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            save_checkpoint(
                train_cfg.checkpoint_path / "best.pt",
                epoch, grounding, optimizer, scheduler, all_metrics, model_cfg,
            )
            print(f"  >> New best model! val_loss={best_val_loss:.4f}")

    # ── Save final model + history ──
    save_checkpoint(
        train_cfg.checkpoint_path / "final.pt",
        train_cfg.epochs, grounding, optimizer, scheduler,
        history[-1] if history else {}, model_cfg,
    )

    history_path = train_cfg.checkpoint_path / "history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved: {history_path}")
    print("Training complete.")


if __name__ == "__main__":
    main()
