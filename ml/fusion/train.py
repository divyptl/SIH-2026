"""
Training script for the Optical-SAR Dual Encoder.

Trains a CLIP-style dual encoder on SEN1-2 SAR-optical image pairs using
NT-Xent contrastive loss with optional terrain classification as an
auxiliary multi-task objective.

Usage:
    # From project root, using the ml venv:
    python -m ml.fusion.train

    # With overrides:
    python -m ml.fusion.train --epochs 100 --batch-size 64 --backbone resnet50

    # Resume from checkpoint:
    python -m ml.fusion.train --resume checkpoints/fusion/epoch_25.pt
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
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.datasets.sen12 import SEN12Dataset
from ml.fusion.config import ModelConfig, TrainConfig
from ml.fusion.model import ContrastiveLoss, DualEncoder, TerrainClassifier
from ml.fusion.transforms import PairedTransform, normalize_optical, normalize_sar


# ── Terrain label mapping ───────────────────────────────────────────────

TERRAIN_TO_IDX = {"agri": 0, "barrenland": 1, "grassland": 2, "urban": 3}


def collate_with_terrain(batch):
    """Custom collate that extracts terrain labels from metadata."""
    sars, opticals, metas = zip(*batch)
    sar_batch = torch.stack(sars)
    opt_batch = torch.stack(opticals)
    terrain_labels = torch.tensor(
        [TERRAIN_TO_IDX[m["terrain"]] for m in metas],
        dtype=torch.long,
    )
    return sar_batch, opt_batch, terrain_labels


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
    model: DualEncoder,
    loss_fn: ContrastiveLoss,
    terrain_head: TerrainClassifier | None,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: str,
    epoch: int,
    cfg: TrainConfig,
) -> dict[str, float]:
    """Train for one epoch. Returns average metrics."""
    model.train()
    loss_fn.train()
    if terrain_head is not None:
        terrain_head.train()

    total_loss = 0.0
    total_contrastive = 0.0
    total_terrain = 0.0
    total_s2o_acc = 0.0
    total_o2s_acc = 0.0
    total_terrain_acc = 0.0
    num_batches = 0

    for step, (sar, optical, terrain_labels) in enumerate(train_loader):
        sar = normalize_sar(sar.to(device))
        optical = normalize_optical(optical.to(device))
        terrain_labels = terrain_labels.to(device)

        # Forward: get embeddings
        sar_emb, opt_emb = model(sar, optical)

        # Contrastive loss
        contrastive_loss, contrastive_metrics = loss_fn(sar_emb, opt_emb)
        loss = contrastive_loss

        # Terrain classification (multi-task)
        terrain_loss_val = 0.0
        terrain_acc_val = 0.0
        if terrain_head is not None:
            sar_feat, opt_feat = model.get_backbone_features(sar, optical)
            terrain_logits = terrain_head(sar_feat.detach(), opt_feat.detach())
            terrain_loss = nn.functional.cross_entropy(terrain_logits, terrain_labels)
            loss = loss + cfg.terrain_loss_weight * terrain_loss
            terrain_loss_val = terrain_loss.item()
            terrain_acc_val = (terrain_logits.argmax(1) == terrain_labels).float().mean().item()

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if terrain_head is not None:
            torch.nn.utils.clip_grad_norm_(terrain_head.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        # Accumulate metrics
        total_loss += loss.item()
        total_contrastive += contrastive_metrics["loss"]
        total_terrain += terrain_loss_val
        total_s2o_acc += contrastive_metrics["sar2opt_acc"]
        total_o2s_acc += contrastive_metrics["opt2sar_acc"]
        total_terrain_acc += terrain_acc_val
        num_batches += 1

        # Logging
        if (step + 1) % cfg.log_every == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  [{epoch}][{step+1}/{len(train_loader)}] "
                f"loss={loss.item():.4f}  "
                f"contra={contrastive_metrics['loss']:.4f}  "
                f"s2o_acc={contrastive_metrics['sar2opt_acc']:.1%}  "
                f"o2s_acc={contrastive_metrics['opt2sar_acc']:.1%}  "
                f"terrain_acc={terrain_acc_val:.1%}  "
                f"temp={contrastive_metrics['temperature']:.4f}  "
                f"lr={lr:.2e}"
            )

    return {
        "loss": total_loss / max(num_batches, 1),
        "contrastive_loss": total_contrastive / max(num_batches, 1),
        "terrain_loss": total_terrain / max(num_batches, 1),
        "sar2opt_acc": total_s2o_acc / max(num_batches, 1),
        "opt2sar_acc": total_o2s_acc / max(num_batches, 1),
        "terrain_acc": total_terrain_acc / max(num_batches, 1),
    }


@torch.no_grad()
def evaluate(
    model: DualEncoder,
    loss_fn: ContrastiveLoss,
    terrain_head: TerrainClassifier | None,
    val_loader: DataLoader,
    device: str,
) -> dict[str, float]:
    """Evaluate on the validation set."""
    model.eval()
    loss_fn.eval()
    if terrain_head is not None:
        terrain_head.eval()

    total_loss = 0.0
    total_s2o_acc = 0.0
    total_o2s_acc = 0.0
    total_terrain_acc = 0.0
    num_batches = 0

    for sar, optical, terrain_labels in val_loader:
        sar = normalize_sar(sar.to(device))
        optical = normalize_optical(optical.to(device))
        terrain_labels = terrain_labels.to(device)

        sar_emb, opt_emb = model(sar, optical)
        _, metrics = loss_fn(sar_emb, opt_emb)

        terrain_acc = 0.0
        if terrain_head is not None:
            sar_feat, opt_feat = model.get_backbone_features(sar, optical)
            terrain_logits = terrain_head(sar_feat, opt_feat)
            terrain_acc = (terrain_logits.argmax(1) == terrain_labels).float().mean().item()

        total_loss += metrics["loss"]
        total_s2o_acc += metrics["sar2opt_acc"]
        total_o2s_acc += metrics["opt2sar_acc"]
        total_terrain_acc += terrain_acc
        num_batches += 1

    return {
        "val_loss": total_loss / max(num_batches, 1),
        "val_sar2opt_acc": total_s2o_acc / max(num_batches, 1),
        "val_opt2sar_acc": total_o2s_acc / max(num_batches, 1),
        "val_terrain_acc": total_terrain_acc / max(num_batches, 1),
    }


# ── Checkpointing ───────────────────────────────────────────────────────

def save_checkpoint(
    path: Path,
    epoch: int,
    model: DualEncoder,
    loss_fn: ContrastiveLoss,
    terrain_head: TerrainClassifier | None,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    metrics: dict,
    model_cfg: ModelConfig,
) -> None:
    """Save a training checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "loss_fn": loss_fn.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "metrics": metrics,
        "model_config": {
            "backbone": model_cfg.backbone,
            "embed_dim": model_cfg.embed_dim,
            "projection_hidden": model_cfg.projection_hidden,
        },
    }
    if terrain_head is not None:
        state["terrain_head"] = terrain_head.state_dict()
    torch.save(state, path)
    print(f"  Checkpoint saved: {path}")


def load_checkpoint(
    path: str,
    model: DualEncoder,
    loss_fn: ContrastiveLoss,
    terrain_head: TerrainClassifier | None,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: str,
) -> int:
    """Load a checkpoint. Returns the epoch to resume from."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    loss_fn.load_state_dict(ckpt["loss_fn"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    if terrain_head is not None and "terrain_head" in ckpt:
        terrain_head.load_state_dict(ckpt["terrain_head"])
    print(f"  Resumed from checkpoint: {path} (epoch {ckpt['epoch']})")
    return ckpt["epoch"]


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Optical-SAR Dual Encoder")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--embed-dim", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers (default: 0 on Windows, 4 otherwise)")
    parser.add_argument("--no-terrain", action="store_true", help="Disable terrain classification head")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    # Build config with CLI overrides
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()

    if args.backbone:
        model_cfg.backbone = args.backbone
    if args.embed_dim:
        model_cfg.embed_dim = args.embed_dim
    if args.epochs:
        train_cfg.epochs = args.epochs
    if args.batch_size:
        train_cfg.batch_size = args.batch_size
    if args.lr:
        train_cfg.lr = args.lr
    if args.image_size:
        train_cfg.image_size = args.image_size
    if args.data_root:
        train_cfg.data_root = args.data_root
    if args.resume:
        train_cfg.resume_from = args.resume
    if args.num_workers is not None:
        train_cfg.num_workers = args.num_workers
    elif platform.system() == "Windows":
        train_cfg.num_workers = 0  # multiprocess DataLoader is slow/broken on Windows
    if args.no_terrain:
        train_cfg.use_terrain_head = False
    if args.device:
        train_cfg.device = args.device

    device = train_cfg.resolve_device()

    print("=" * 70)
    print("  Optical-SAR Dual Encoder Training")
    print("=" * 70)
    print(f"  Backbone:     {model_cfg.backbone}")
    print(f"  Embed dim:    {model_cfg.embed_dim}")
    print(f"  Batch size:   {train_cfg.batch_size}")
    print(f"  Epochs:       {train_cfg.epochs}")
    print(f"  LR:           {train_cfg.lr}")
    print(f"  Image size:   {train_cfg.image_size}")
    print(f"  Terrain head: {train_cfg.use_terrain_head}")
    print(f"  Device:       {device}")
    print(f"  Data root:    {train_cfg.data_root}")
    print("=" * 70)

    # ── Data ──
    print("\nLoading datasets...")
    pair_transform_train = PairedTransform(size=train_cfg.image_size, augment=True)
    pair_transform_val = PairedTransform(size=train_cfg.image_size, augment=False)

    train_ds = SEN12Dataset(
        root=train_cfg.data_root,
        terrains=train_cfg.terrains,
        split="train",
        pair_transform=pair_transform_train,
        return_metadata=True,
    )
    val_ds = SEN12Dataset(
        root=train_cfg.data_root,
        terrains=train_cfg.terrains,
        split="val",
        pair_transform=pair_transform_val,
        return_metadata=True,
    )

    print(f"  Train: {len(train_ds):,} pairs")
    print(f"  Val:   {len(val_ds):,} pairs")

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        pin_memory=train_cfg.pin_memory,
        collate_fn=collate_with_terrain,
        drop_last=True,  # Required for contrastive loss (need full batches)
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        pin_memory=train_cfg.pin_memory,
        collate_fn=collate_with_terrain,
    )

    # ── Model ──
    print("\nBuilding model...")
    model = DualEncoder(
        backbone=model_cfg.backbone,
        pretrained=model_cfg.pretrained,
        embed_dim=model_cfg.embed_dim,
        projection_hidden=model_cfg.projection_hidden,
    ).to(device)

    loss_fn = ContrastiveLoss(
        temperature=model_cfg.temperature,
        learn_temperature=model_cfg.learn_temperature,
    ).to(device)

    terrain_head = None
    if train_cfg.use_terrain_head:
        # Get backbone feature dim
        feat_dim = 512 if model_cfg.backbone in ("resnet18", "resnet34") else 2048
        terrain_head = TerrainClassifier(feature_dim=feat_dim, num_classes=4).to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model params: {total_params:.1f}M")

    # ── Optimizer ──
    param_groups = [
        {"params": model.parameters()},
        {"params": loss_fn.parameters()},
    ]
    if terrain_head is not None:
        param_groups.append({"params": terrain_head.parameters()})

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

    # ── Resume ──
    start_epoch = 1
    if train_cfg.resume_from:
        start_epoch = load_checkpoint(
            train_cfg.resume_from, model, loss_fn, terrain_head,
            optimizer, scheduler, device,
        ) + 1

    # ── Training loop ──
    print(f"\nStarting training from epoch {start_epoch}...\n")
    best_val_loss = float("inf")
    history: list[dict] = []

    for epoch in range(start_epoch, train_cfg.epochs + 1):
        t0 = time.time()

        # Train
        train_metrics = train_one_epoch(
            model, loss_fn, terrain_head, train_loader,
            optimizer, scheduler, device, epoch, train_cfg,
        )

        # Evaluate
        val_metrics = {}
        if epoch % train_cfg.eval_every == 0:
            val_metrics = evaluate(model, loss_fn, terrain_head, val_loader, device)

        elapsed = time.time() - t0
        all_metrics = {**train_metrics, **val_metrics, "epoch": epoch, "time": elapsed}
        history.append(all_metrics)

        # Epoch summary
        val_str = ""
        if val_metrics:
            val_str = (
                f"  val_loss={val_metrics['val_loss']:.4f}  "
                f"val_s2o={val_metrics['val_sar2opt_acc']:.1%}  "
                f"val_o2s={val_metrics['val_opt2sar_acc']:.1%}  "
                f"val_terrain={val_metrics['val_terrain_acc']:.1%}"
            )
        print(
            f"Epoch {epoch}/{train_cfg.epochs}  "
            f"loss={train_metrics['loss']:.4f}  "
            f"s2o={train_metrics['sar2opt_acc']:.1%}  "
            f"o2s={train_metrics['opt2sar_acc']:.1%}  "
            f"terrain={train_metrics['terrain_acc']:.1%}"
            f"{val_str}  "
            f"[{elapsed:.1f}s]"
        )

        # Save checkpoint
        if epoch % train_cfg.save_every == 0:
            save_checkpoint(
                train_cfg.checkpoint_path / f"epoch_{epoch}.pt",
                epoch, model, loss_fn, terrain_head,
                optimizer, scheduler, all_metrics, model_cfg,
            )

        # Save best model
        if val_metrics and val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            save_checkpoint(
                train_cfg.checkpoint_path / "best.pt",
                epoch, model, loss_fn, terrain_head,
                optimizer, scheduler, all_metrics, model_cfg,
            )
            print(f"  >> New best model! val_loss={best_val_loss:.4f}")

    # ── Save final model + training history ──
    save_checkpoint(
        train_cfg.checkpoint_path / "final.pt",
        train_cfg.epochs, model, loss_fn, terrain_head,
        optimizer, scheduler, history[-1] if history else {}, model_cfg,
    )

    history_path = train_cfg.checkpoint_path / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved: {history_path}")
    print("Training complete.")


if __name__ == "__main__":
    main()
