"""
Training script for the Physics-Guided Optical-SAR Dual Encoder.

Includes:
  - Hard-Negative Terrain Mining in NT-Xent ContrastiveLoss
  - 2D Fourier Phase-Consistency Loss (FFT edge alignment across SAR & Optical)
  - Cross-Gated Multi-Task Terrain Classification
  - Refined Lee SAR Speckle Filtering + Multiplicative Gamma Speckle Augmentation
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ML_ROOT = Path(__file__).resolve().parent.parent
for p in (str(PROJECT_ROOT), str(ML_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from ml.datasets.sen12 import SEN12Dataset
from ml.datasets.qxs import QXSDataset
from ml.fusion.config import ModelConfig, TrainConfig
from ml.fusion.model import (
    ContrastiveLoss,
    DualEncoder,
    FourierPhaseLoss,
    TerrainClassifier,
)
from ml.fusion.transforms import PairedTransform, normalize_optical, normalize_sar

TERRAIN_TO_IDX = {"agri": 0, "barrenland": 1, "grassland": 2, "urban": 3, "water": 4}


def collate_with_terrain(batch):
    """Custom collate that extracts terrain labels from metadata."""
    sars, opticals, metas = zip(*batch)
    sar_batch = torch.stack(sars)
    opt_batch = torch.stack(opticals)
    terrain_labels = torch.tensor(
        [TERRAIN_TO_IDX.get(m.get("terrain", ""), -1) for m in metas],
        dtype=torch.long,
    )
    return sar_batch, opt_batch, terrain_labels


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


def train_one_epoch(
    model: DualEncoder,
    loss_fn: ContrastiveLoss,
    phase_loss_fn: FourierPhaseLoss,
    terrain_head: TerrainClassifier | None,
    train_loaders: list[DataLoader],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler | None,
    use_amp: bool,
    device: str,
    epoch: int,
    cfg: TrainConfig,
    detach_terrain: bool = False,
) -> dict[str, float]:
    """Train for one epoch across all configured datasets."""
    model.train()
    loss_fn.train()
    if terrain_head is not None:
        terrain_head.train()

    total_loss = 0.0
    total_contrastive = 0.0
    total_phase = 0.0
    total_terrain = 0.0
    total_s2o_acc = 0.0
    total_o2s_acc = 0.0
    total_terrain_acc = 0.0
    num_batches = 0

    amp_device = "cuda" if "cuda" in device else "cpu"

    for step, batches in enumerate(zip(*train_loaders)):
        loss = torch.tensor(0.0, device=device)
        step_contrastive = 0.0
        step_phase = 0.0
        step_terrain = 0.0
        step_s2o = 0.0
        step_o2s = 0.0
        step_terrain_acc = 0.0
        last_temp = 0.0

        num_domains = len(batches)
        optimizer.zero_grad(set_to_none=True)

        for sar, optical, terrain_labels in batches:
            sar = normalize_sar(sar.to(device, non_blocking=True))
            optical = normalize_optical(optical.to(device, non_blocking=True))
            terrain_labels = terrain_labels.to(device, non_blocking=True)

            with torch.amp.autocast(amp_device, enabled=use_amp):
                sar_emb, opt_emb = model(sar, optical)

                # 1. Hard-Negative Terrain-Aware Contrastive Loss
                contrastive_loss, contrastive_metrics = loss_fn(
                    sar_emb, opt_emb, terrain_labels=terrain_labels
                )
                domain_loss = contrastive_loss

                # 2. 2D Fourier Phase-Consistency Loss on Spatial Feature Grids
                domain_phase_loss = 0.0
                if cfg.phase_loss_weight > 0.0:
                    sar_map, opt_map = model.get_spatial_feature_maps()
                    p_loss = phase_loss_fn(sar_map, opt_map)
                    domain_loss = domain_loss + cfg.phase_loss_weight * p_loss
                    domain_phase_loss = p_loss.item()

                # 3. Cross-Gated Terrain Classification Loss
                domain_terrain_loss = 0.0
                domain_terrain_acc = 0.0
                valid_mask = terrain_labels >= 0
                if terrain_head is not None and valid_mask.any():
                    sar_feat, opt_feat = model.get_backbone_features()
                    if detach_terrain:
                        sar_feat, opt_feat = sar_feat.detach(), opt_feat.detach()
                    terrain_logits = terrain_head(sar_feat[valid_mask], opt_feat[valid_mask])
                    valid_labels = terrain_labels[valid_mask]
                    terrain_loss = nn.functional.cross_entropy(terrain_logits, valid_labels)
                    domain_loss = domain_loss + cfg.terrain_loss_weight * terrain_loss
                    domain_terrain_loss = terrain_loss.item()
                    domain_terrain_acc = (terrain_logits.argmax(1) == valid_labels).float().mean().item()

                domain_loss = domain_loss / num_domains

            if scaler is not None:
                scaler.scale(domain_loss).backward()
            else:
                domain_loss.backward()

            loss = loss + domain_loss.detach()
            step_contrastive += contrastive_metrics["loss"]
            step_phase += domain_phase_loss
            step_terrain += domain_terrain_loss
            step_s2o += contrastive_metrics["sar2opt_acc"]
            step_o2s += contrastive_metrics["opt2sar_acc"]
            step_terrain_acc += domain_terrain_acc
            last_temp = contrastive_metrics["temperature"]

        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if terrain_head is not None:
                torch.nn.utils.clip_grad_norm_(terrain_head.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if terrain_head is not None:
                torch.nn.utils.clip_grad_norm_(terrain_head.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        total_loss += loss.item()
        total_contrastive += step_contrastive / num_domains
        total_phase += step_phase / num_domains
        total_terrain += step_terrain / num_domains
        total_s2o_acc += step_s2o / num_domains
        total_o2s_acc += step_o2s / num_domains
        total_terrain_acc += step_terrain_acc / num_domains
        num_batches += 1

        if (step + 1) % cfg.log_every == 0:
            lr = optimizer.param_groups[0]["lr"]
            num_steps = min(len(dl) for dl in train_loaders)
            print(
                f"  [{epoch}][{step+1}/{num_steps}] "
                f"loss={loss.item():.4f}  "
                f"contra={step_contrastive/num_domains:.4f}  "
                f"fft_phase={step_phase/num_domains:.4f}  "
                f"s2o={step_s2o/num_domains:.1%}  "
                f"o2s={step_o2s/num_domains:.1%}  "
                f"terrain={step_terrain_acc/num_domains:.1%}  "
                f"temp={last_temp:.4f}  "
                f"lr={lr:.2e}"
            )

    denom = max(num_batches, 1)
    return {
        "loss": total_loss / denom,
        "contrastive_loss": total_contrastive / denom,
        "phase_loss": total_phase / denom,
        "terrain_loss": total_terrain / denom,
        "sar2opt_acc": total_s2o_acc / denom,
        "opt2sar_acc": total_o2s_acc / denom,
        "terrain_acc": total_terrain_acc / denom,
    }


@torch.no_grad()
def evaluate(
    model: DualEncoder,
    loss_fn: ContrastiveLoss,
    terrain_head: TerrainClassifier | None,
    val_loaders: list[DataLoader],
    device: str,
    use_amp: bool = False,
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
    terrain_batches = 0

    amp_device = "cuda" if "cuda" in device else "cpu"

    for val_loader in val_loaders:
        for sar, optical, terrain_labels in val_loader:
            sar = normalize_sar(sar.to(device, non_blocking=True))
            optical = normalize_optical(optical.to(device, non_blocking=True))
            terrain_labels = terrain_labels.to(device, non_blocking=True)

            with torch.amp.autocast(amp_device, enabled=use_amp):
                sar_emb, opt_emb = model(sar, optical)
                _, metrics = loss_fn(sar_emb, opt_emb)

                valid_mask = terrain_labels >= 0
                if terrain_head is not None and valid_mask.any():
                    sar_feat, opt_feat = model.get_backbone_features()
                    terrain_logits = terrain_head(sar_feat[valid_mask], opt_feat[valid_mask])
                    terrain_acc = (
                        (terrain_logits.argmax(1) == terrain_labels[valid_mask]).float().mean().item()
                    )
                    total_terrain_acc += terrain_acc
                    terrain_batches += 1

            total_loss += metrics["loss"]
            total_s2o_acc += metrics["sar2opt_acc"]
            total_o2s_acc += metrics["opt2sar_acc"]
            num_batches += 1

    denom = max(num_batches, 1)
    return {
        "val_loss": total_loss / denom,
        "val_sar2opt_acc": total_s2o_acc / denom,
        "val_opt2sar_acc": total_o2s_acc / denom,
        "val_terrain_acc": total_terrain_acc / max(terrain_batches, 1),
    }


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
    scaler: torch.amp.GradScaler | None = None,
) -> None:
    """Save a training checkpoint with full architecture metadata for inference."""
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
            "use_attention": model_cfg.use_attention,
            "attn_dim": model_cfg.attn_dim,
            "attn_heads": model_cfg.attn_heads,
            "attn_layers": model_cfg.attn_layers,
            "sar_in_channels": model_cfg.sar_in_channels,
            "use_cross_gate": model_cfg.use_cross_gate,
            "hard_negative_weight": model_cfg.hard_negative_weight,
        },
    }
    if terrain_head is not None:
        state["terrain_head"] = terrain_head.state_dict()
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
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
    scaler: torch.amp.GradScaler | None = None,
) -> int:
    """Load a checkpoint. Returns the epoch to resume from."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    loss_fn.load_state_dict(ckpt["loss_fn"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    if terrain_head is not None and "terrain_head" in ckpt:
        terrain_head.load_state_dict(ckpt["terrain_head"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    print(f"  Resumed from checkpoint: {path} (epoch {ckpt['epoch']})")
    return ckpt["epoch"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Optical-SAR Dual Encoder")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--embed-dim", type=int, default=None)
    parser.add_argument("--sar-channels", type=int, default=None, choices=[1, 3])
    parser.add_argument("--use-attention", action="store_true", help="Enable Self-Attention block")
    parser.add_argument("--no-cross-gate", action="store_true", help="Disable Cross-Modal Gating in TerrainHead")
    parser.add_argument("--apply-lee", action="store_true", help="Apply Refined Lee SAR filter during loading")
    parser.add_argument("--phase-weight", type=float, default=None, help="Weight for 2D Fourier Phase Loss")
    parser.add_argument("--hard-neg-weight", type=float, default=None, help="Hard-negative terrain multiplier")
    parser.add_argument("--detach-terrain", action="store_true", help="Detach backbone features before terrain head")
    parser.add_argument("--auto-hardware", action="store_true", help="Auto-scale batch_size & num_workers from VRAM/CPU")
    parser.add_argument("--dataset", type=str, default=None, choices=["sen12", "qxs", "both"])
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--qxs-root", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--no-terrain", action="store_true", help="Disable terrain classification head")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    model_cfg = ModelConfig()
    train_cfg = TrainConfig()

    if args.auto_hardware:
        train_cfg.auto_scale_hardware()

    if args.backbone:
        model_cfg.backbone = args.backbone
    if args.embed_dim:
        model_cfg.embed_dim = args.embed_dim
    if args.sar_channels:
        model_cfg.sar_in_channels = args.sar_channels
    if args.use_attention:
        model_cfg.use_attention = True
    if args.no_cross_gate:
        model_cfg.use_cross_gate = False
    if args.hard_neg_weight is not None:
        model_cfg.hard_negative_weight = args.hard_neg_weight
    if args.apply_lee:
        train_cfg.apply_lee = True
    if args.phase_weight is not None:
        train_cfg.phase_loss_weight = args.phase_weight
    if args.epochs:
        train_cfg.epochs = args.epochs
    if args.dataset:
        train_cfg.dataset = args.dataset
    if args.batch_size:
        train_cfg.batch_size = args.batch_size
    if args.lr:
        train_cfg.lr = args.lr
    if args.image_size:
        train_cfg.image_size = args.image_size
    if args.data_root:
        train_cfg.data_root = args.data_root
    if args.qxs_root:
        train_cfg.qxs_root = args.qxs_root
    if args.checkpoint_dir:
        train_cfg.checkpoint_dir = args.checkpoint_dir
    if args.resume:
        train_cfg.resume_from = args.resume
    if args.num_workers is not None:
        train_cfg.num_workers = args.num_workers
    if args.save_every is not None:
        train_cfg.save_every = args.save_every
    if args.no_terrain:
        train_cfg.use_terrain_head = False
    if args.device:
        train_cfg.device = args.device

    device = train_cfg.resolve_device()

    print("=" * 70)
    print("  Optical-SAR Physics & Frequency-Guided Dual Encoder Training")
    print("=" * 70)
    print(f"  Backbone:       {model_cfg.backbone} (SAR channels={model_cfg.sar_in_channels})")
    print(f"  Embed dim:      {model_cfg.embed_dim} (Proj hidden={model_cfg.projection_hidden})")
    print(f"  Hard-Neg Mult:  {model_cfg.hard_negative_weight}x | FFT Phase Weight: {train_cfg.phase_loss_weight}")
    print(f"  Lee Speckle:    {train_cfg.apply_lee} (+ Gamma Multiplicative Speckle Aug)")
    print(f"  Terrain head:   {train_cfg.use_terrain_head} (Cross-Gate={model_cfg.use_cross_gate})")
    print(f"  Batch size:     {train_cfg.batch_size} | Workers: {train_cfg.num_workers}")
    print(f"  Epochs:         {train_cfg.epochs} | LR: {train_cfg.lr}")
    print(f"  Device:         {device} | Checkpoint dir: {train_cfg.checkpoint_dir}")
    print("=" * 70)

    print("\nLoading datasets...")
    pair_transform_train = PairedTransform(
        size=train_cfg.image_size, augment=True, apply_lee=train_cfg.apply_lee
    )
    pair_transform_val = PairedTransform(
        size=train_cfg.image_size, augment=False, apply_lee=train_cfg.apply_lee
    )

    train_datasets, val_datasets = [], []

    if train_cfg.dataset in ("sen12", "both"):
        train_datasets.append(SEN12Dataset(
            root=train_cfg.data_root,
            terrains=train_cfg.terrains,
            split="train",
            pair_transform=pair_transform_train,
            return_metadata=True,
        ))
        val_datasets.append(SEN12Dataset(
            root=train_cfg.data_root,
            terrains=train_cfg.terrains,
            split="val",
            pair_transform=pair_transform_val,
            return_metadata=True,
        ))

    if train_cfg.dataset in ("qxs", "both"):
        train_datasets.append(QXSDataset(
            root=train_cfg.qxs_root,
            split="train",
            pair_transform=pair_transform_train,
            return_metadata=True,
        ))
        val_datasets.append(QXSDataset(
            root=train_cfg.qxs_root,
            split="val",
            pair_transform=pair_transform_val,
            return_metadata=True,
        ))

    loader_kwargs = dict(
        batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers,
        pin_memory=train_cfg.pin_memory,
        collate_fn=collate_with_terrain,
        persistent_workers=train_cfg.num_workers > 0,
    )

    train_loaders = [DataLoader(ds, shuffle=True, drop_last=True, **loader_kwargs) for ds in train_datasets]
    val_loaders = [DataLoader(ds, shuffle=False, **loader_kwargs) for ds in val_datasets]

    print(f"  Train: {sum(len(ds) for ds in train_datasets):,} pairs across {len(train_datasets)} dataset(s)")
    print(f"  Val:   {sum(len(ds) for ds in val_datasets):,} pairs across {len(val_datasets)} dataset(s)")

    print("\nBuilding model...")
    model = DualEncoder(
        backbone=model_cfg.backbone,
        pretrained=model_cfg.pretrained,
        embed_dim=model_cfg.embed_dim,
        projection_hidden=model_cfg.projection_hidden,
        use_attention=model_cfg.use_attention,
        attn_dim=model_cfg.attn_dim,
        attn_heads=model_cfg.attn_heads,
        attn_layers=model_cfg.attn_layers,
        sar_in_channels=model_cfg.sar_in_channels,
    ).to(device)

    loss_fn = ContrastiveLoss(
        temperature=model_cfg.temperature,
        learn_temperature=model_cfg.learn_temperature,
        hard_negative_weight=model_cfg.hard_negative_weight,
    ).to(device)

    phase_loss_fn = FourierPhaseLoss().to(device)

    terrain_head = None
    if train_cfg.use_terrain_head:
        feat_dim = (
            768
            if model_cfg.backbone == "convnext_tiny"
            else (512 if model_cfg.backbone in ("resnet18", "resnet34") else 2048)
        )
        terrain_head = TerrainClassifier(
            feature_dim=feat_dim,
            num_classes=len(TERRAIN_TO_IDX),
            use_cross_gate=model_cfg.use_cross_gate,
        ).to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model params: {total_params:.1f}M")

    param_groups = [
        {"params": model.parameters()},
        {"params": loss_fn.parameters()},
    ]
    if terrain_head is not None:
        param_groups.append({"params": terrain_head.parameters()})

    optimizer = torch.optim.AdamW(param_groups, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    steps_per_epoch = min(len(dl) for dl in train_loaders)
    scheduler = get_cosine_schedule(
        optimizer, train_cfg.warmup_epochs, train_cfg.epochs, train_cfg.min_lr, steps_per_epoch
    )

    use_amp = train_cfg.use_amp and "cuda" in device
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    start_epoch = 1
    if train_cfg.resume_from:
        start_epoch = load_checkpoint(
            train_cfg.resume_from, model, loss_fn, terrain_head, optimizer, scheduler, device, scaler
        ) + 1

    print(f"\nStarting training from epoch {start_epoch}...\n")
    best_val_loss = float("inf")
    history: list[dict] = []

    for epoch in range(start_epoch, train_cfg.epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, loss_fn, phase_loss_fn, terrain_head, train_loaders,
            optimizer, scheduler, scaler, use_amp, device, epoch, train_cfg,
            detach_terrain=args.detach_terrain,
        )

        val_metrics = {}
        if epoch % train_cfg.eval_every == 0:
            val_metrics = evaluate(model, loss_fn, terrain_head, val_loaders, device, use_amp)

        elapsed = time.time() - t0
        all_metrics = {**train_metrics, **val_metrics, "epoch": epoch, "time": elapsed}
        history.append(all_metrics)

        val_str = (
            f"  val_loss={val_metrics['val_loss']:.4f}  "
            f"val_s2o={val_metrics['val_sar2opt_acc']:.1%}  "
            f"val_o2s={val_metrics['val_opt2sar_acc']:.1%}  "
            f"val_terrain={val_metrics['val_terrain_acc']:.1%}"
            if val_metrics else ""
        )
        print(
            f"Epoch {epoch}/{train_cfg.epochs}  "
            f"loss={train_metrics['loss']:.4f}  "
            f"phase={train_metrics['phase_loss']:.4f}  "
            f"s2o={train_metrics['sar2opt_acc']:.1%}  "
            f"o2s={train_metrics['opt2sar_acc']:.1%}  "
            f"terrain={train_metrics['terrain_acc']:.1%}"
            f"{val_str}  [{elapsed:.1f}s]"
        )

        save_checkpoint(
            train_cfg.checkpoint_path / "latest.pt",
            epoch, model, loss_fn, terrain_head, optimizer, scheduler, all_metrics, model_cfg, scaler,
        )

        if epoch % train_cfg.save_every == 0:
            save_checkpoint(
                train_cfg.checkpoint_path / f"epoch_{epoch}.pt",
                epoch, model, loss_fn, terrain_head, optimizer, scheduler, all_metrics, model_cfg, scaler,
            )

        if val_metrics and val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            save_checkpoint(
                train_cfg.checkpoint_path / "best.pt",
                epoch, model, loss_fn, terrain_head, optimizer, scheduler, all_metrics, model_cfg, scaler,
            )
            print(f"  >> New best model! val_loss={best_val_loss:.4f}")

    save_checkpoint(
        train_cfg.checkpoint_path / "final.pt",
        train_cfg.epochs, model, loss_fn, terrain_head,
        optimizer, scheduler, history[-1] if history else {}, model_cfg, scaler,
    )

    history_path = train_cfg.checkpoint_path / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved: {history_path}")
    print("Training complete.")


if __name__ == "__main__":
    main()