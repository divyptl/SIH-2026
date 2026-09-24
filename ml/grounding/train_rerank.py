"""
Train the candidate re-ranker on cached GroundingDINO outputs.

Build the caches first (see build_rerank_cache.py), then:

    python -m ml.grounding.train_rerank

The checkpoint is picked on a slice of the *train* images held out as a dev set,
so the VRSBench eval file, which is reported at the end, plays no part in model
selection.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.grounding.config import RerankConfig
from ml.grounding.rerank import CandidateReranker, candidate_ious, rerank_loss

FEATURE_KEYS = ("hidden", "roi", "boxes", "scores", "cand_mask", "tok_logits", "text", "text_mask")


class CachedCandidates:
    """A cached split held in memory, with the candidate list cut to K."""

    def __init__(self, cache_dir: Path, k: int) -> None:
        with open(cache_dir / "meta.json", encoding="utf-8") as f:
            self.meta = json.load(f)
        if k > self.meta["cache_k"]:
            raise ValueError(f"Asked for {k} candidates but the cache holds {self.meta['cache_k']}")

        self.data: dict[str, torch.Tensor] = {}
        for key in FEATURE_KEYS:
            array = np.load(cache_dir / f"{key}.npy")
            if key in ("hidden", "roi", "boxes", "scores", "cand_mask", "tok_logits"):
                array = array[:, :k]
            self.data[key] = torch.from_numpy(np.ascontiguousarray(array))
        self.gt = torch.from_numpy(np.load(cache_dir / "gt.npy"))
        self.samples = self.meta["samples"]

    def __len__(self) -> int:
        return len(self.gt)

    def batch(self, rows: torch.Tensor, device: str) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        inputs = {}
        for key, tensor in self.data.items():
            value = tensor[rows].to(device, non_blocking=True)
            inputs[key] = value.float() if value.is_floating_point() else value
        return inputs, self.gt[rows].to(device)


def dev_split(samples: list[dict], fraction: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Hold out whole images, chosen by a stable hash of the file name."""
    is_dev = torch.tensor([
        int(hashlib.md5(s["image"].encode()).hexdigest(), 16) % 10_000 < fraction * 10_000
        for s in samples
    ])
    rows = torch.arange(len(samples))
    return rows[~is_dev], rows[is_dev]


def box_iou_paired(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """IoU between matching rows of two (N, 4) cxcywh tensors."""
    return candidate_ious(a.unsqueeze(1), b, torch.ones(len(a), 1, dtype=torch.bool, device=a.device))[:, 0]


@torch.no_grad()
def evaluate(
    model: CandidateReranker | None,
    data: CachedCandidates,
    rows: torch.Tensor,
    device: str,
    batch_size: int = 1024,
) -> list[dict]:
    """Per-expression IoU of the chosen box. With no model, GroundingDINO's own top-1."""
    if model is not None:
        model.eval()
    records = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        inputs, gt = data.batch(chunk, device)
        cand_ious = candidate_ious(inputs["boxes"], gt, inputs["cand_mask"])
        if model is None:
            pick = torch.zeros(len(chunk), dtype=torch.long, device=device)
            iou_raw = iou_refined = cand_ious[:, 0]
        else:
            logits, refined = model(inputs)
            pick = logits.argmax(-1)
            ar = torch.arange(len(chunk), device=device)
            iou_raw = cand_ious[ar, pick]
            iou_refined = box_iou_paired(refined[ar, pick], gt)
        oracle = cand_ious.max(-1).values
        for j, row in enumerate(chunk.tolist()):
            s = data.samples[row]
            records.append({
                "row": row, "unique": s.get("unique"), "obj_cls": s.get("obj_cls"),
                "iou": iou_raw[j].item(), "iou_refined": iou_refined[j].item(),
                "oracle": oracle[j].item(), "pick": pick[j].item(),
            })
    return records


def summarize(records: list[dict], key: str) -> dict:
    def m(rs):
        n = max(len(rs), 1)
        return {
            "count": len(rs),
            "acc@0.5": sum(r[key] >= 0.5 for r in rs) / n,
            "acc@0.7": sum(r[key] >= 0.7 for r in rs) / n,
            "miou": sum(r[key] for r in rs) / n,
        }
    out = {"all": m(records)}
    if any(r["unique"] is not None for r in records):
        out["unique"] = m([r for r in records if r["unique"] is True])
        out["non_unique"] = m([r for r in records if r["unique"] is False])
    return out


def print_table(title: str, results: dict[str, dict]) -> None:
    print(f"\n  {title}")
    print(f"  {'':<26}{'subset':<12}{'count':>7}{'Acc@0.5':>10}{'Acc@0.7':>10}{'mIoU':>8}")
    for name, summary in results.items():
        for subset, m in summary.items():
            print(f"  {name:<26}{subset:<12}{m['count']:>7}{m['acc@0.5']:>10.1%}"
                  f"{m['acc@0.7']:>10.1%}{m['miou']:>8.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the GroundingDINO candidate re-ranker")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-candidates", type=int, default=None)
    parser.add_argument("--no-refine", action="store_true")
    parser.add_argument("--output", default=None, help="Checkpoint path")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = RerankConfig()
    if args.epochs:
        cfg.epochs = args.epochs
    if args.lr:
        cfg.lr = args.lr
    if args.num_candidates:
        cfg.num_candidates = args.num_candidates
    if args.no_refine:
        cfg.refine_boxes = False
    if args.output:
        cfg.checkpoint = args.output
    cache_dir = Path(args.cache_dir or cfg.cache_dir)

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading cached candidates...")
    train_data = CachedCandidates(cache_dir / "train", cfg.num_candidates)
    val_data = CachedCandidates(cache_dir / "validation", cfg.num_candidates)
    train_rows, dev_rows = dev_split(train_data.samples, cfg.dev_fraction)
    val_rows = torch.arange(len(val_data))
    print(f"  train {len(train_rows):,}  dev {len(dev_rows):,}  eval {len(val_rows):,}  "
          f"(K={cfg.num_candidates})")

    model = CandidateReranker(cfg).to(device)
    print(f"  Re-ranker params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = math.ceil(len(train_rows) / cfg.batch_size)
    total = cfg.epochs * steps_per_epoch
    warmup = cfg.warmup_epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: s / max(warmup, 1) if s < warmup
        else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(total - warmup, 1))),
    )

    iou_key = "iou_refined" if cfg.refine_boxes else "iou"
    best_acc, best_state, best_epoch = -1.0, None, 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()
        perm = train_rows[torch.randperm(len(train_rows))]
        running = {}
        for start in range(0, len(perm), cfg.batch_size):
            inputs, gt = train_data.batch(perm[start:start + cfg.batch_size], device)
            logits, refined = model(inputs)
            loss, stats = rerank_loss(logits, refined, inputs, gt, cfg.min_iou, cfg.refine_boxes)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            for k, v in stats.items():
                running[k] = running.get(k, 0.0) + v

        dev = summarize(evaluate(model, train_data, dev_rows, device), iou_key)["all"]
        loss_str = "  ".join(f"{k}={v / steps_per_epoch:.4f}" for k, v in running.items())
        marker = ""
        if dev["acc@0.5"] > best_acc:
            best_acc, best_epoch = dev["acc@0.5"], epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            marker = "  *"
        print(f"Epoch {epoch:>2}/{cfg.epochs}  {loss_str}  dev Acc@0.5={dev['acc@0.5']:.1%} "
              f"Acc@0.7={dev['acc@0.7']:.1%}  [{time.time() - t0:.1f}s]{marker}")

    model.load_state_dict(best_state)
    print(f"\nBest dev epoch: {best_epoch} (Acc@0.5 {best_acc:.1%})")

    baseline_dev = evaluate(None, train_data, dev_rows, device)
    model_dev = evaluate(model, train_data, dev_rows, device)
    baseline_val = evaluate(None, val_data, val_rows, device)
    model_val = evaluate(model, val_data, val_rows, device)

    oracle = lambda rs: [dict(r, iou=r["oracle"]) for r in rs]
    print_table("Dev (held-out train images)", {
        "GroundingDINO top-1": summarize(baseline_dev, "iou"),
        "re-ranked": summarize(model_dev, "iou"),
        "re-ranked + refined": summarize(model_dev, "iou_refined"),
    })
    val_summary = {
        "GroundingDINO top-1": summarize(baseline_val, "iou"),
        "re-ranked": summarize(model_val, "iou"),
        "re-ranked + refined": summarize(model_val, "iou_refined"),
        f"oracle (best of {cfg.num_candidates})": summarize(oracle(model_val), "iou"),
    }
    print_table(f"VRSBench eval ({len(val_rows):,} expressions)", val_summary)

    out = Path(cfg.checkpoint)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": best_state,
        "config": dataclasses.asdict(cfg),
        "detector_checkpoint": train_data.meta["checkpoint"],
        "epoch": best_epoch,
        "metrics": {"dev_acc@0.5": best_acc, "eval": val_summary},
    }, out)
    print(f"\nSaved re-ranker: {out}")


if __name__ == "__main__":
    main()
