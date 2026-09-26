"""
Training script for the Siamese Change Detection / Change-VQA specialist model.

Trains the Siamese Vision Encoder + VLM Head jointly on bi-temporal image pairs,
optimizing both VQA question-answering accuracy and dense change mask segmentation (IoU)
via multi-task learning.

Usage:
    # From repository root:
    python -m ml.C_VQA.train

    # With custom flags:
    python -m ml.C_VQA.train --epochs 20 --batch-size 8 --backbone resnet18

    # Multi-source dataset built by prepare.py, starting from an earlier checkpoint:
    python -m ml.C_VQA.train --data-root data/change_vqa --init-from checkpoints/c_vqa_best.pt
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.C_VQA.config import ModelConfig, TrainConfig
from ml.C_VQA.dataset import CDVQADataset, cdvqa_collate_fn
from ml.C_VQA.model import ChangeVQALoss, SiameseChangeVQA, SimpleTokenizer
from ml.C_VQA.transforms import PairedBitemporalTransform


# ── Learning Rate Scheduler ─────────────────────────────────────────────

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    steps_per_epoch: int,
    min_lr_ratio: float = 1e-2,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create a learning rate scheduler with linear warmup and cosine decay.

    ``min_lr_ratio`` is a multiplier on each param group's base LR, since
    LambdaLR scales every group's own LR by the returned factor.
    """
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine_decay)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Trainer Class ────────────────────────────────────────────────────────

class ChangeVQATrainer:
    """End-to-end trainer for Siamese Change-VQA specialist model."""

    def __init__(
        self,
        model: SiameseChangeVQA,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        train_cfg: TrainConfig,
        model_cfg: ModelConfig,
        device: str,
        dataset_info: dict[str, Any] | None = None,
    ) -> None:
        self.model = model.to(device)
        self.dataset_info = dataset_info or {}
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = train_cfg
        self.model_cfg = model_cfg
        self.device = device

        # Multi-task criterion
        self.criterion = ChangeVQALoss(
            vqa_weight=self.cfg.vqa_loss_weight,
            mask_bce_weight=self.cfg.mask_bce_weight,
            mask_dice_weight=self.cfg.mask_dice_weight,
        )

        # Differential learning rates: lower for pretrained backbone, higher for heads
        backbone_params = list(self.model.backbone.parameters())
        head_params = [
            p for n, p in self.model.named_parameters()
            if not n.startswith("backbone.")
        ]

        self.optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": self.cfg.backbone_lr},
                {"params": head_params, "lr": self.cfg.lr},
            ],
            weight_decay=self.cfg.weight_decay,
        )

        steps_per_epoch = max(1, len(train_loader))
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            warmup_epochs=self.cfg.warmup_epochs,
            total_epochs=self.cfg.epochs,
            steps_per_epoch=steps_per_epoch,
            min_lr_ratio=self.cfg.min_lr / self.cfg.lr,
        )

        # Checkpointing state
        self.best_metric = 0.0
        self.start_epoch = 0
        self.history: list[dict[str, Any]] = []
        self.checkpoint_dir = Path(self.cfg.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Mixed-precision training (AMP): create the grad scaler once, before the
        # training loop, so its loss scale adapts across steps
        self.use_amp = self.cfg.use_amp and self.device == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        total_vqa_acc = 0.0
        total_mask_iou = 0.0
        num_batches = len(self.train_loader)
        start_time = time.time()

        for step, batch in enumerate(self.train_loader):
            t1 = batch["t1"].to(self.device)
            t2 = batch["t2"].to(self.device)
            q_ids = batch["question_ids"].to(self.device)
            ans_targets = batch["answer_targets"].to(self.device)
            mask_targets = batch["mask_targets"].to(self.device) if batch["mask_targets"] is not None else None

            self.optimizer.zero_grad(set_to_none=True)

            # Forward pass under autocast
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                outputs = self.model(t1=t1, t2=t2, question_ids=q_ids)
                loss, metrics = self.criterion(outputs, ans_targets, mask_targets)

            # Scaled backward pass; unscale before clipping so the norm is real
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item()
            total_vqa_acc += metrics.get("vqa_accuracy", 0.0)
            total_mask_iou += metrics.get("mask_iou", 0.0)

            if (step + 1) % self.cfg.log_every == 0 or (step + 1) == num_batches:
                current_lr = self.optimizer.param_groups[1]["lr"]
                print(
                    f"[Epoch {epoch:2d}/{self.cfg.epochs}] Step {step+1:3d}/{num_batches} | "
                    f"Loss: {loss.item():.4f} (VQA: {metrics['loss_vqa']:.4f}, Mask: {metrics.get('loss_mask_bce', 0.0):.4f}) | "
                    f"Acc: {metrics.get('vqa_accuracy', 0.0):.1%} | "
                    f"IoU: {metrics.get('mask_iou', 0.0):.3f} | "
                    f"LR: {current_lr:.2e}"
                )

        elapsed = time.time() - start_time
        return {
            "train_loss": total_loss / max(1, num_batches),
            "train_vqa_acc": total_vqa_acc / max(1, num_batches),
            "train_mask_iou": total_mask_iou / max(1, num_batches),
            "train_time_sec": elapsed,
        }

    @torch.no_grad()
    def evaluate(self) -> dict[str, Any]:
        """Evaluate on the validation split, overall and per source.

        Each source is scored separately so a small dataset (say, 10 m imagery)
        is not drowned out by a large one; model selection uses their mean.
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)
        correct: dict[str, int] = defaultdict(int)
        answered: dict[str, int] = defaultdict(int)
        by_type: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        overlap: dict[str, float] = defaultdict(float)
        union: dict[str, float] = defaultdict(float)

        for batch in self.val_loader:
            t1 = batch["t1"].to(self.device)
            t2 = batch["t2"].to(self.device)
            q_ids = batch["question_ids"].to(self.device)
            ans_targets = batch["answer_targets"].to(self.device)
            mask_targets = batch["mask_targets"].to(self.device) if batch["mask_targets"] is not None else None
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                outputs = self.model(t1=t1, t2=t2, question_ids=q_ids)
                loss, _ = self.criterion(outputs, ans_targets, mask_targets)
            total_loss += loss.item()

            hits = (outputs["answer_logits"].argmax(dim=-1) == ans_targets).tolist()
            valid = (ans_targets != -100).tolist()
            if mask_targets is not None:
                pred = (outputs["change_mask_prob"].float() > 0.5).float()
                target = mask_targets if mask_targets.dim() == 4 else mask_targets.unsqueeze(1)
                inter = (pred * target).sum(dim=(1, 2, 3)).tolist()
                either = ((pred + target) > 0).float().sum(dim=(1, 2, 3)).tolist()
            else:
                inter = either = [0.0] * len(hits)

            for i, source in enumerate(batch["sources"]):
                overlap[source] += inter[i]
                union[source] += either[i]
                if valid[i]:
                    answered[source] += 1
                    correct[source] += int(hits[i])
                    stats = by_type[batch["question_types"][i]]
                    stats[0] += int(hits[i])
                    stats[1] += 1

        per_source = {
            source: {
                "vqa_acc": correct[source] / max(1, answered[source]),
                # No change anywhere in a source's val tiles counts as a perfect mask.
                "mask_iou": overlap[source] / union[source] if union[source] else 1.0,
            }
            for source in sorted(set(answered) | set(union))
        }
        n = max(1, len(per_source))
        return {
            "val_loss": total_loss / max(1, num_batches),
            "val_vqa_acc": sum(correct.values()) / max(1, sum(answered.values())),
            "val_mask_iou": sum(overlap.values()) / max(1e-6, sum(union.values())),
            "val_score": sum(0.5 * m["vqa_acc"] + 0.5 * m["mask_iou"] for m in per_source.values()) / n,
            "val_per_source": per_source,
            "val_per_question_type": {k: v[0] / max(1, v[1]) for k, v in sorted(by_type.items())},
        }

    def _build_checkpoint(self, epoch: int) -> dict[str, Any]:
        """Build a checkpoint dict for the current training state."""
        return {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_metric": self.best_metric,
            "model_config": vars(self.model_cfg),
            "train_config": vars(self.cfg),
            "answers_vocab": self.model.answers_vocab,
            "question_vocab": self.model.tokenizer.vocab,
            "scaler": self.scaler.state_dict(),
            # Ground sample distances the training tiles covered; the backend
            # only routes imagery inside this range to the model.
            "gsd_range_m": self.dataset_info.get("gsd_range_m"),
            "sources": sorted(self.dataset_info.get("sources", {})),
        }

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save training checkpoint to disk.

        Always saves:
          - ``latest.pt``      — most recent epoch (for easy resume)
          - ``epoch_NNN.pt``   — individual epoch snapshot

        Conditionally saves:
          - ``best.pt``        — only when *is_best* is True
        """
        ckpt = self._build_checkpoint(epoch)

        # 1. Individual epoch file
        epoch_path = self.checkpoint_dir / f"epoch_{epoch:03d}.pt"
        torch.save(ckpt, epoch_path)

        # 2. latest.pt (always overwritten)
        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(ckpt, latest_path)

        print(f"[ChangeVQATrainer] Saved epoch_{epoch:03d}.pt + latest.pt")

        # 3. best.pt (only when improved)
        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(ckpt, best_path)
            print(f"[ChangeVQATrainer] New best checkpoint! (score={self.best_metric:.4f}) -> best.pt")

    def fit(self) -> list[dict[str, Any]]:
        """Run the full training loop across all epochs."""
        print(f"\n[ChangeVQATrainer] Starting training on {self.device} for {self.cfg.epochs} epochs...")
        print(f"[ChangeVQATrainer] Architecture: {self.model_cfg.backbone} Siamese backbone + VLM cross-attention head")

        for epoch in range(self.start_epoch + 1, self.cfg.epochs + 1):
            self.start_epoch = epoch
            epoch_log: dict[str, Any] = {"epoch": epoch}

            # Train
            train_metrics = self.train_epoch(epoch)
            epoch_log.update(train_metrics)

            # Evaluate
            if self.val_loader is not None and epoch % self.cfg.eval_every == 0:
                val_metrics = self.evaluate()
                epoch_log.update(val_metrics)
                print(
                    f"--> [Val Epoch {epoch:2d}] Loss: {val_metrics['val_loss']:.4f} | "
                    f"VQA Acc: {val_metrics['val_vqa_acc']:.1%} | "
                    f"Mask IoU: {val_metrics['val_mask_iou']:.3f} | "
                    f"Score (mean over sources): {val_metrics['val_score']:.3f}"
                )
                for source, m in val_metrics["val_per_source"].items():
                    print(f"      {source:18} VQA Acc {m['vqa_acc']:.1%} | Mask IoU {m['mask_iou']:.3f}")

                # Composite metric: mean over sources of 0.5 * VQA_Acc + 0.5 * Mask_IoU
                composite_score = val_metrics["val_score"]
                is_best = composite_score > self.best_metric
                if is_best:
                    self.best_metric = composite_score
            else:
                is_best = False

            self.history.append(epoch_log)

            # Checkpoint every epoch: epoch_NNN.pt + latest.pt (+ best.pt when improved)
            self.save_checkpoint(epoch, is_best=is_best)

            # Save metrics history
            with open(self.checkpoint_dir / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2)

        print(f"[ChangeVQATrainer] Training complete! Best score: {self.best_metric:.4f}")
        return self.history


# ── Entry Point ──────────────────────────────────────────────────────────

def train_main(args: argparse.Namespace) -> None:
    """Setup and run training."""
    model_cfg = ModelConfig(
        backbone=args.backbone,
        pretrained=not args.no_pretrained,
    )
    dataset_info_path = Path(args.data_root) / "dataset_info.json"
    dataset_info = json.loads(dataset_info_path.read_text()) if dataset_info_path.exists() else {}
    if args.init_from:
        # The architecture must match the checkpoint being fine-tuned.
        init_cfg = torch.load(args.init_from, map_location="cpu", weights_only=False).get("model_config", {})
        for key in ("backbone", "visual_feature_dim", "text_embed_dim", "mask_hidden_dim",
                    "cross_attention_layers", "num_cross_attention_heads", "feedforward_dim",
                    "spatial_token_resolution", "answer_hidden_dim"):
            if key in init_cfg:
                setattr(model_cfg, key, init_cfg[key])
    train_cfg = TrainConfig(
        data_root=args.data_root,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
    )

    device = train_cfg.resolve_device()

    # Datasets and Loaders (Strictly Real-Life Datasets)
    print(f"[ChangeVQATrainer] Loading real CDVQA dataset from '{train_cfg.data_root}'...")
    try:
        train_ds = CDVQADataset(
            root=train_cfg.data_root,
            split=train_cfg.split_train,
            max_question_length=model_cfg.max_question_length,
            auto_generate=False,
        )
        print(f"[ChangeVQATrainer] Loaded {len(train_ds)} real training samples for split '{train_cfg.split_train}'.")
    except FileNotFoundError as e:
        print(f"\n[ChangeVQATrainer Error] Failed to load real training data:\n{e}\n")
        print("Tip: Run 'python data/cdvqa/download_cdvqa.py' or provide the path to your real CDVQA dataset using --data-root.")
        sys.exit(1)

    try:
        # Val must index questions/answers exactly as the model was trained
        val_ds = CDVQADataset(
            root=train_cfg.data_root,
            split=train_cfg.split_val,
            max_question_length=model_cfg.max_question_length,
            auto_generate=False,
            question_vocab=train_ds.question_vocab,
            answer_vocab=train_ds.answer_vocab,
        )
        print(f"[ChangeVQATrainer] Loaded {len(val_ds)} real validation samples for split '{train_cfg.split_val}'.")
    except (FileNotFoundError, ValueError) as e:
        print(f"[ChangeVQATrainer] Validation split not available or empty ({e}); continuing with train split only.")
        val_ds = None

    # Workers decode PNG tiles in parallel; 0 stays the safe default on Windows.
    loader_args: dict[str, Any] = {
        "batch_size": train_cfg.batch_size,
        "collate_fn": cdvqa_collate_fn,
        "num_workers": args.num_workers,
        "pin_memory": device == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
    val_loader = (
        DataLoader(val_ds, shuffle=False, **loader_args)
        if val_ds is not None
        else None
    )

    # Size the embedding table and answer head to the data-derived vocabularies
    model_cfg.vocab_size = len(train_ds.question_vocab)
    model_cfg.num_classes = len(train_ds.answer_vocab)
    print(
        f"[ChangeVQATrainer] Question vocab: {model_cfg.vocab_size} tokens | "
        f"Answer classes: {model_cfg.num_classes}"
    )

    model = SiameseChangeVQA(config=model_cfg)
    model.tokenizer = SimpleTokenizer(train_ds.question_vocab)
    model.answers_vocab = list(train_ds.answer_vocab)
    if args.init_from:
        load_matching_weights(model, args.init_from)
    trainer = ChangeVQATrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_cfg=train_cfg,
        model_cfg=model_cfg,
        device=device,
        dataset_info=dataset_info,
    )

    trainer.fit()


def load_matching_weights(model: SiameseChangeVQA, checkpoint_path: str) -> None:
    """Initialise from a checkpoint, skipping tensors whose shape changed.

    The vision backbone, difference module and mask head carry over; the text
    embedding and answer head are resized to the new vocabularies and so
    start fresh.
    """
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["model"]
    own = model.state_dict()
    matching = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    model.load_state_dict(matching, strict=False)
    skipped = sorted(k for k in state if k not in matching)
    print(f"[ChangeVQATrainer] Initialised {len(matching)}/{len(own)} tensors from {checkpoint_path}; "
          f"{len(skipped)} re-initialised (e.g. {skipped[:3]})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Siamese Change-VQA specialist model")
    parser.add_argument("--data-root", default="data/cdvqa", help="Path to CDVQA dataset directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--backbone", default="resnet18", help="Vision backbone (resnet18, resnet34, resnet50)")
    parser.add_argument("--no-pretrained", action="store_true", help="Do not load ImageNet weights")
    parser.add_argument("--checkpoint-dir", default="checkpoints/change_detection", help="Directory to save checkpoints")
    parser.add_argument("--device", default="auto", help="Device (auto, cuda, cpu)")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (use 2-4 on Kaggle)")
    parser.add_argument("--init-from", default=None,
                        help="Checkpoint to initialise matching weights from (e.g. checkpoints/c_vqa_best.pt)")

    cli_args = parser.parse_args()
    train_main(cli_args)
