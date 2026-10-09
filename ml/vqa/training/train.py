"""
Train the single-image VQA specialist on VRSBench.

Prepare the data first:
    python -m ml.vqa.training.prepare_vrsbench

Then:
    python -m ml.vqa.training.train

The model answers by choosing from the answer vocabulary built from the
training data (data/vrsbench_vqa/answers.json). The checkpoint is picked on the
dev split (VQA pairs from held-out training images); the official VRSBench VQA
evaluation set is scored once at the end and plays no part in any choice.

Writes checkpoints/vqa/best.pt (best dev accuracy) and last.pt. This is the
single-image model; the two-image Change-VQA model lives in ml.C_VQA.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ml.vqa.config import ModelConfig, TrainConfig
from ml.vqa.model import VQALoss, VQAModel
from ml.vqa.training.dataset import UNKNOWN_ANSWER, VQADataset

logger = logging.getLogger(__name__)


def amp_dtype(device: str) -> torch.dtype | None:
    """bf16 where the GPU supports it natively, else fp16 (with loss scaling)."""
    if device != "cuda":
        return None
    if torch.cuda.is_bf16_supported(including_emulation=False):
        return torch.bfloat16
    return torch.float16


@torch.no_grad()
def evaluate(model: VQAModel, loader: DataLoader, device: str, dtype: torch.dtype | None) -> dict:
    """Exact-match accuracy, overall and per question type."""
    model.eval()
    dataset: VQADataset = loader.dataset
    hits: dict[str, list[int]] = defaultdict(list)
    for batch in loader:
        with torch.autocast(device_type=device, dtype=dtype, enabled=dtype is not None):
            out = model(image=batch["image"].to(device, non_blocking=True),
                        question_ids=batch["question_ids"].to(device, non_blocking=True))
        pred = out["answer_logits"].argmax(-1).cpu()
        correct = (pred == batch["answer_label"]) & (batch["answer_label"] != UNKNOWN_ANSWER)
        for ok, idx in zip(correct.tolist(), batch["index"].tolist()):
            hits["all"].append(ok)
            qtype = dataset.samples[idx].get("type")
            if qtype:
                hits[qtype].append(ok)
    return {name: {"count": len(v), "accuracy": sum(v) / len(v)} for name, v in hits.items()}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train the single-image VQA specialist")
    parser.add_argument("--data-root", default="data/vrsbench_vqa")
    parser.add_argument("--checkpoint-dir", default="checkpoints/vqa")
    parser.add_argument("--backbone", default=None,
                        help="convnext_tiny (default) | resnet18 | resnet34 | resnet50")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--max-samples", type=int, default=None, help="For quick tests")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    train_cfg = TrainConfig(data_root=args.data_root, epochs=args.epochs,
                            batch_size=args.batch_size, num_workers=args.num_workers,
                            checkpoint_dir=args.checkpoint_dir)
    if args.lr:
        train_cfg.lr = args.lr
    if args.image_size:
        train_cfg.image_size = args.image_size
    device = train_cfg.resolve_device()
    dtype = amp_dtype(device)

    root = Path(args.data_root)
    answers = json.loads((root / "answers.json").read_text(encoding="utf-8"))
    words = json.loads((root / "questions.json").read_text(encoding="utf-8"))
    model_cfg = ModelConfig(num_classes=len(answers), vocab_size=len(words) + 4)
    if args.backbone:
        model_cfg.backbone = args.backbone
    model = VQAModel(model_cfg, answers=answers, question_vocab=words).to(device)
    tok = model.tokenizer

    def loader(split: str, train: bool) -> DataLoader:
        ds = VQADataset(root, split, answers, tok, train_cfg, drop_unknown_answers=train,
                        max_question_length=model_cfg.max_question_length)
        if args.max_samples:
            ds.samples = ds.samples[:args.max_samples]
        workers = train_cfg.num_workers
        return DataLoader(ds, batch_size=train_cfg.batch_size, shuffle=train, drop_last=train,
                          num_workers=workers, pin_memory=device == "cuda",
                          persistent_workers=workers > 0)

    train_loader = loader("train", True)
    dev_loader = loader("dev", False)
    logger.info(f"Device {device} ({dtype}) | backbone {model_cfg.backbone} | image "
                f"{train_cfg.image_size}px | {len(answers)} answers | {len(tok.vocab)} words")
    logger.info(f"train {len(train_loader.dataset):,} (answer in vocabulary) | "
                f"dev {len(dev_loader.dataset):,}")

    backbone = [p for n, p in model.named_parameters() if n.startswith("backbone.")]
    heads = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    optimizer = torch.optim.AdamW(
        [{"params": backbone, "lr": train_cfg.backbone_lr}, {"params": heads, "lr": train_cfg.lr}],
        weight_decay=train_cfg.weight_decay,
    )
    total = train_cfg.epochs * len(train_loader)
    warmup = min(len(train_loader), total // 10)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: s / max(warmup, 1) if s < warmup
        else max(0.01, 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(total - warmup, 1)))),
    )
    scaler = torch.amp.GradScaler("cuda") if dtype == torch.float16 else None
    criterion = VQALoss()

    out_dir = Path(args.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history, best_acc = [], -1.0
    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        t0, running = time.time(), 0.0
        for step, batch in enumerate(train_loader, 1):
            with torch.autocast(device_type=device, dtype=dtype, enabled=dtype is not None):
                out = model(image=batch["image"].to(device, non_blocking=True),
                            question_ids=batch["question_ids"].to(device, non_blocking=True))
                loss = criterion(out, {"answer_label": batch["answer_label"].to(device)})["loss"]
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
            running += loss.item()
            if step % 200 == 0:
                logger.info(f"  [{epoch}][{step}/{len(train_loader)}] loss {running / step:.4f}")

        dev = evaluate(model, dev_loader, device, dtype)["all"]["accuracy"]
        history.append({"epoch": epoch, "loss": running / len(train_loader), "dev_acc": dev,
                        "time": time.time() - t0})
        extra = {"epoch": epoch, "dev_acc": dev, "history": history,
                 "train_config": dataclasses.asdict(train_cfg)}
        torch.save(model.checkpoint_dict(**extra), out_dir / "last.pt")
        marker = ""
        if dev > best_acc:
            best_acc = dev
            torch.save(model.checkpoint_dict(**extra), out_dir / "best.pt")
            marker = "  * best"
        logger.info(f"Epoch {epoch}/{train_cfg.epochs}  loss {running / len(train_loader):.4f}  "
                    f"dev acc {dev:.2%}  [{time.time() - t0:.0f}s]{marker}")

    # ── Final score of the best checkpoint on the official evaluation set ──
    best = VQAModel.from_checkpoint(str(out_dir / "best.pt"), device=device)
    test = evaluate(best, loader("test", False), device, dtype)
    logger.info(f"\nVRSBench VQA eval set — {out_dir / 'best.pt'}")
    logger.info(f"  {'question type':<20}{'count':>7}{'accuracy':>10}")
    for name, m in sorted(test.items(), key=lambda kv: (kv[0] != "all", -kv[1]["count"])):
        logger.info(f"  {name:<20}{m['count']:>7}{m['accuracy']:>10.1%}")
    ckpt = torch.load(out_dir / "best.pt", map_location="cpu", weights_only=False)
    ckpt["test_metrics"] = test
    torch.save(ckpt, out_dir / "best.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
