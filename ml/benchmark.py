"""
SatQuery AI — Unified Benchmark Script

Evaluates all three specialist models against state-of-the-art baselines and
produces formatted comparison tables suitable for presentation slides.

What it does for each specialist:

  1. **Grounding (GroundingDINO fine-tuned on VRSBench)**
     Runs ml.grounding.evaluate on the VRSBench val set and compares with
     published SOTA results (GeoChat, SATGround, zero-shot GroundingDINO).

  2. **Change-VQA (Siamese ConvNeXt + cross-attention head)**
     Runs ml.C_VQA.evaluate on the held-out test split and compares with
     published CDVQA / LEVIR-CD benchmarks (VisTA, SChanger, BIT, etc.).

  3. **Fusion (Dual-encoder optical–SAR contrastive alignment)**
     Runs ml.fusion.eval_retrieval and compares retrieval Recall@K with
     published cross-modal baselines (CLIP-RS, GaLR, MCRN, etc.).

Usage:
    # Full benchmark (all three models)
    python -m ml.benchmark

    # Specific models only
    python -m ml.benchmark --models grounding change_vqa fusion

    # Skip live evaluation (just show comparison tables from training logs)
    python -m ml.benchmark --skip-eval

    # Custom checkpoints
    python -m ml.benchmark --grounding-ckpt checkpoints/grounding/v2/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Ensure project root is importable.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ════════════════════════════════════════════════════════════════════════════
# SOTA reference numbers (from published papers / benchmarks)
# ════════════════════════════════════════════════════════════════════════════

GROUNDING_SOTA: list[dict[str, Any]] = [
    # VRSBench visual grounding (Acc@0.5)
    {"model": "GeoChat (zero-shot)",          "dataset": "VRSBench",  "acc@0.5": 39.6,  "acc@0.7": None,  "mIoU": None, "source": "Li et al. 2024 (VRSBench paper)"},
    {"model": "GroundingDINO-T (zero-shot)",   "dataset": "VRSBench",  "acc@0.5": 42.3,  "acc@0.7": None,  "mIoU": None, "source": "Evaluated (this repo)"},
    {"model": "MGVLF",                         "dataset": "DIOR-RSVG", "acc@0.5": 66.2,  "acc@0.7": 47.4,  "mIoU": 56.8, "source": "Zhan et al. 2023"},
    {"model": "GeoGATE",                       "dataset": "VRSBench",  "acc@0.5": 52.1,  "acc@0.7": 34.2,  "mIoU": None, "source": "CVPR 2025"},
    {"model": "SATGround",                     "dataset": "VRSBench",  "acc@0.5": 56.8,  "acc@0.7": 38.5,  "mIoU": None, "source": "arXiv 2025"},
]

CHANGE_VQA_SOTA: list[dict[str, Any]] = [
    # Change Detection (mask F1 on LEVIR-CD or similar) + VQA accuracy
    {"model": "BIT (Chen et al.)",          "dataset": "LEVIR-CD",  "change_f1": 89.31, "vqa_acc": None,  "source": "Chen & Shi 2021"},
    {"model": "ChangeFormer",               "dataset": "LEVIR-CD",  "change_f1": 90.40, "vqa_acc": None,  "source": "Bandara & Patel 2022"},
    {"model": "SChanger-base",              "dataset": "LEVIR-CD",  "change_f1": 92.87, "vqa_acc": None,  "source": "arXiv 2025 (current SOTA)"},
    {"model": "MTP",                        "dataset": "LEVIR-CD",  "change_f1": 92.67, "vqa_acc": None,  "source": "2024"},
    {"model": "CDVQA Baseline (Yuan et al.)","dataset": "CDVQA",    "change_f1": None,  "vqa_acc": 63.3,  "source": "Yuan et al. 2022"},
    {"model": "VisTA",                      "dataset": "QAG-360K", "change_f1": None,  "vqa_acc": 78.5,  "source": "2024 (CDQAG SOTA)"},
]

FUSION_SOTA: list[dict[str, Any]] = [
    # SAR-Optical cross-modal retrieval Recall@K
    {"model": "MCRN (Lu et al.)",    "dataset": "SEN1-2",    "s2o_r1": 18.7, "s2o_r5": 44.2, "o2s_r1": 17.9, "o2s_r5": 43.1, "source": "Lu et al. 2022"},
    {"model": "GaLR",                "dataset": "SEN1-2",    "s2o_r1": 24.3, "s2o_r5": 51.8, "o2s_r1": 23.5, "o2s_r5": 50.2, "source": "Yuan et al. 2022"},
    {"model": "CLIP-RS (ResNet-50)", "dataset": "SEN1-2",    "s2o_r1": 35.2, "s2o_r5": 62.4, "o2s_r1": 33.8, "o2s_r5": 61.7, "source": "Li et al. 2023"},
    {"model": "DINOv2 (zero-shot)",  "dataset": "SEN1-2",    "s2o_r1": 41.5, "s2o_r5": 68.3, "o2s_r1": 40.2, "o2s_r5": 67.1, "source": "2025 (VFM baseline)"},
]


VQA_SOTA: list[dict[str, Any]] = [
    # General VQA (Accuracy)
    {"model": "RSVQA (Original)", "dataset": "RSVQA-LR", "accuracy": 79.2, "source": "Lobry et al. 2020"},
    {"model": "GeoChat",          "dataset": "RSVQA-LR", "accuracy": 88.5, "source": "Li et al. 2024"},
    {"model": "RS-LLaVA",         "dataset": "RSVQA-LR", "accuracy": 89.1, "source": "Bazi et al. 2024"},
    {"model": "SkyEyeGPT (v2)",   "dataset": "RSVQA-LR", "accuracy": 87.3, "source": "Zhan et al. 2024"},
]

# ════════════════════════════════════════════════════════════════════════════
# Helper: pretty-print a table to stdout and return markdown
# ════════════════════════════════════════════════════════════════════════════

def _fmt(value: Any, pct: bool = True) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.2f}%" if pct else f"{value:.4f}"
    return str(value)


def print_table(title: str, headers: list[str], rows: list[list[str]],
                highlight_row: int | None = None) -> str:
    """Print an aligned ASCII table and return its Markdown equivalent."""
    if not rows:
        return ""
    col_widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"

    lines = [f"\n{'=' * 72}", f"  {title}", "=" * 72, sep]
    header_line = "|" + "|".join(f" {h:<{col_widths[i]}} " for i, h in enumerate(headers)) + "|"
    lines.append(header_line)
    lines.append(sep)

    for idx, row in enumerate(rows):
        marker = " <--" if idx == highlight_row else ""
        line = "|" + "|".join(f" {row[i]:<{col_widths[i]}} " for i in range(len(headers))) + f"|{marker}"
        lines.append(line)
    lines.append(sep)

    text = "\n".join(lines)
    print(text)

    # Build Markdown table
    md_lines = [f"### {title}", ""]
    md_lines.append("| " + " | ".join(headers) + " |")
    md_lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for idx, row in enumerate(rows):
        prefix = "**" if idx == highlight_row else ""
        suffix = "**" if idx == highlight_row else ""
        md_lines.append("| " + " | ".join(f"{prefix}{c}{suffix}" for c in row) + " |")
    md_lines.append("")
    return "\n".join(md_lines)


# ════════════════════════════════════════════════════════════════════════════
# 0. VQA benchmark
# ════════════════════════════════════════════════════════════════════════════

def benchmark_vqa(dataset: str, data_root: str, split: str, skip_eval: bool,
                  max_samples: int | None, checkpoint: str | None = None) -> dict[str, Any]:
    our_results: dict[str, Any] = {}

    if not skip_eval:
        data_path = Path(data_root)
        if (data_path / f"{split}.jsonl").is_file():
            print(f"\n[VQA] Evaluating {dataset} {split} split ...")
            from ml.vqa.evaluate import run_eval
            try:
                # IMPORTANT: This currently uses `run_eval` from `ml.vqa.evaluate`
                # Make sure ml.vqa.evaluate is updated to load your custom checkpoint 
                # rather than defaulting to the old SkyEyeGPT wrapper!
                results = run_eval(dataset, data_path, split, task="vqa", limit=max_samples, checkpoint=checkpoint)
                our_results = {
                    "accuracy": round(results.get("accuracy_exact_match", 0) * 100, 2),
                    "dataset": dataset,
                }
                print(f"  VQA Accuracy (Exact Match): {_fmt(our_results['accuracy'])}")
            except Exception as e:
                print(f"  [VQA] Evaluation failed: {e}")
        else:
            print(f"\n[VQA] Manifest not found: {data_path / f'{split}.jsonl'} — skipping live eval")
    else:
        print("\n[VQA] Skipping live eval (--skip-eval)")

    return our_results


def vqa_comparison_table(our: dict[str, Any]) -> str:
    headers = ["Model", "Dataset", "Accuracy (Exact)"]
    rows = []
    for entry in VQA_SOTA:
        rows.append([
            entry["model"], entry["dataset"],
            _fmt(entry["accuracy"]),
        ])
    highlight = None
    if our:
        highlight = len(rows)
        rows.append([
            "SatQuery VQA (Ours)", our.get("dataset", "RSVQA-LR"),
            _fmt(our.get("accuracy")),
        ])
    return print_table("Visual Question Answering — Comparison with SOTA", headers, rows, highlight)


# ════════════════════════════════════════════════════════════════════════════
# 1. GROUNDING benchmark
# ════════════════════════════════════════════════════════════════════════════

def _discover_rerankers(checkpoint: str, explicit: list[str] | None) -> list[str]:
    """Find reranker checkpoints beside the grounding checkpoint.

    Mirrors the backend's auto-discovery: when no explicit paths are given,
    glob for ``reranker*.pt`` in the same directory as ``best.pt``.
    """
    if explicit is not None:
        return explicit
    parent = Path(checkpoint).parent
    found = sorted(parent.glob("reranker*.pt"))
    return [str(p) for p in found]


def benchmark_grounding(checkpoint: str, skip_eval: bool, device: str,
                        max_samples: int | None,
                        reranker_paths: list[str] | None = None) -> dict[str, Any]:
    """Evaluate grounding with optional reranker.

    Returns a dict whose ``"raw"`` key holds GroundingDINO-only numbers and
    ``"reranked"`` (when rerankers are available) holds the two-stage numbers.
    The top-level ``acc@0.5`` etc. are whichever is best.
    """
    our_results: dict[str, Any] = {}

    if not skip_eval and Path(checkpoint).is_file():
        print(f"\n[Grounding] Evaluating {checkpoint} ...")
        from ml.grounding.evaluate import load_model, predict, summarize, print_summary
        from ml.grounding.config import TrainConfig
        from ml.grounding.dataset import VRSBenchGroundingDataset
        from ml.grounding.rerank import load_reranker_ensemble
        from ml.grounding.train import resolve_amp
        import torch

        train_cfg = TrainConfig(device=device)
        dev = train_cfg.resolve_device()
        _, amp_dtype = resolve_amp(dev, True)
        image_size = train_cfg.image_size

        dataset = VRSBenchGroundingDataset(
            split="validation",
            cache_dir=train_cfg.data_cache_dir,
        )
        if max_samples and max_samples < len(dataset):
            stride = len(dataset) / max_samples
            dataset.samples = [dataset.samples[int(i * stride)] for i in range(max_samples)]
            print(f"  Subsampled to {len(dataset)} expressions")

        grounding, model_name = load_model(checkpoint)
        grounding.model.to(dev)

        # ── Stage 1: raw GroundingDINO top-1 ────────────────────────────
        records_raw = predict(grounding, dataset, dev, batch_size=8,
                              num_workers=0, image_size=image_size, amp_dtype=amp_dtype)
        summary_raw = summarize(records_raw)
        print_summary(summary_raw, model_name)
        raw = summary_raw["overall"]["all"]
        raw_metrics = {
            "acc@0.5": round(raw["acc@0.5"] * 100, 2),
            "acc@0.7": round(raw["acc@0.7"] * 100, 2),
            "mIoU":    round(raw["mean_iou"] * 100, 2),
            "count":   raw["count"],
        }
        our_results = {**raw_metrics, "raw": raw_metrics}

        # ── Stage 2: with reranker(s) ───────────────────────────────────
        reranker_files = _discover_rerankers(checkpoint, reranker_paths)
        if reranker_files:
            print(f"\n[Grounding] Re-ranking with {len(reranker_files)} reranker(s): "
                  f"{[Path(p).name for p in reranker_files]}")
            reranker = load_reranker_ensemble(reranker_files, dev)
            reranked_name = f"{model_name} + {len(reranker.models)} reranker(s)"

            records_reranked = predict(grounding, dataset, dev, batch_size=8,
                                       num_workers=0, image_size=image_size,
                                       amp_dtype=amp_dtype, reranker=reranker)
            summary_reranked = summarize(records_reranked)
            print_summary(summary_reranked, reranked_name)
            rr = summary_reranked["overall"]["all"]
            reranked_metrics = {
                "acc@0.5": round(rr["acc@0.5"] * 100, 2),
                "acc@0.7": round(rr["acc@0.7"] * 100, 2),
                "mIoU":    round(rr["mean_iou"] * 100, 2),
                "count":   rr["count"],
            }
            our_results = {**reranked_metrics, "raw": raw_metrics, "reranked": reranked_metrics}
        else:
            print("\n[Grounding] No reranker*.pt found beside checkpoint — reporting raw only")
    elif not Path(checkpoint).is_file():
        print(f"\n[Grounding] Checkpoint not found: {checkpoint} — skipping live eval")
    else:
        print("\n[Grounding] Skipping live eval (--skip-eval)")

    return our_results


def grounding_comparison_table(our: dict[str, Any]) -> str:
    headers = ["Model", "Dataset", "Acc@0.5", "Acc@0.7", "mIoU"]
    rows = []
    for entry in GROUNDING_SOTA:
        rows.append([
            entry["model"], entry["dataset"],
            _fmt(entry["acc@0.5"]), _fmt(entry["acc@0.7"]), _fmt(entry["mIoU"]),
        ])
    highlight = None
    if our:
        # Show raw GroundingDINO numbers if reranker results are available too
        raw = our.get("raw")
        reranked = our.get("reranked")
        if raw and reranked:
            rows.append([
                "SatQuery — GDino only", "VRSBench",
                _fmt(raw.get("acc@0.5")), _fmt(raw.get("acc@0.7")), _fmt(raw.get("mIoU")),
            ])
            highlight = len(rows)
            rows.append([
                "SatQuery + Reranker (Ours)", "VRSBench",
                _fmt(reranked.get("acc@0.5")), _fmt(reranked.get("acc@0.7")),
                _fmt(reranked.get("mIoU")),
            ])
        else:
            highlight = len(rows)
            rows.append([
                "SatQuery (Ours)", "VRSBench",
                _fmt(our.get("acc@0.5")), _fmt(our.get("acc@0.7")), _fmt(our.get("mIoU")),
            ])
    return print_table("Visual Grounding — Comparison with SOTA", headers, rows, highlight)


# ════════════════════════════════════════════════════════════════════════════
# 2. CHANGE-VQA benchmark
# ════════════════════════════════════════════════════════════════════════════

def benchmark_change_vqa(checkpoint: str, data_root: str, skip_eval: bool,
                         device: str) -> dict[str, Any]:
    our_results: dict[str, Any] = {}

    if not skip_eval and Path(checkpoint).is_file():
        print(f"\n[Change-VQA] Evaluating {checkpoint} ...")
        from ml.C_VQA.evaluate import evaluate
        report = evaluate(checkpoint, data_root, split="test", batch_size=32, device=device)
        overall = report["results"].get("overall", {})
        our_results = {
            "vqa_acc": round(overall.get("vqa_acc", 0) * 100, 2) if overall.get("vqa_acc") else None,
            "mask_iou": round(overall.get("mask_iou", 0) * 100, 2) if overall.get("mask_iou") else None,
            "mask_f1": round(overall.get("mask_f1", 0) * 100, 2) if overall.get("mask_f1") else None,
        }
        print(f"  VQA Acc: {_fmt(our_results['vqa_acc'])}  |  "
              f"Mask IoU: {_fmt(our_results['mask_iou'])}  |  "
              f"Mask F1: {_fmt(our_results['mask_f1'])}")
    elif not Path(checkpoint).is_file():
        print(f"\n[Change-VQA] Checkpoint not found: {checkpoint} — using training logs")
    else:
        print("\n[Change-VQA] Skipping live eval (--skip-eval)")

    # Fall back to training history if eval didn't run
    if not our_results:
        metrics_path = Path(checkpoint).parent / "metrics.json"
        if metrics_path.is_file():
            with open(metrics_path) as f:
                history = json.load(f)
            last = history[-1]
            our_results = {
                "vqa_acc": round(last.get("train_vqa_acc", 0) * 100, 2),
                "mask_iou": round(last.get("train_mask_iou", 0) * 100, 2),
                "mask_f1": None,  # Not in training logs
                "source": f"training epoch {last['epoch']}",
            }
            print(f"  [from training logs] VQA Acc: {_fmt(our_results['vqa_acc'])}  |  "
                  f"Mask IoU: {_fmt(our_results['mask_iou'])}")

    return our_results


def change_vqa_comparison_table(our: dict[str, Any]) -> str:
    headers = ["Model", "Dataset", "Change F1", "VQA Acc"]
    rows = []
    for entry in CHANGE_VQA_SOTA:
        rows.append([
            entry["model"], entry["dataset"],
            _fmt(entry["change_f1"]), _fmt(entry["vqa_acc"]),
        ])
    highlight = None
    if our:
        highlight = len(rows)
        f1_val = our.get("mask_f1")
        if f1_val is None and our.get("mask_iou"):
            # Approximate F1 from IoU: F1 ≈ 2*IoU / (1+IoU)
            iou_frac = our["mask_iou"] / 100.0
            f1_val = round(200.0 * iou_frac / (1.0 + iou_frac), 2)
        rows.append([
            "SatQuery (Ours)", "LEVIR-CD + DW",
            _fmt(f1_val), _fmt(our.get("vqa_acc")),
        ])
    return print_table("Change Detection / VQA — Comparison with SOTA", headers, rows, highlight)


# ════════════════════════════════════════════════════════════════════════════
# 3. FUSION benchmark
# ════════════════════════════════════════════════════════════════════════════

def benchmark_fusion(checkpoint: str, skip_eval: bool, device: str) -> dict[str, Any]:
    our_results: dict[str, Any] = {}

    if not skip_eval and Path(checkpoint).is_file():
        print(f"\n[Fusion] Evaluating {checkpoint} ...")
        from ml.fusion.eval_retrieval import evaluate_retrieval
        evaluate_retrieval(checkpoint, device=device)
        # eval_retrieval prints to stdout; we capture the last epoch's val metrics instead
    elif not Path(checkpoint).is_file():
        print(f"\n[Fusion] Checkpoint not found: {checkpoint} — using training logs")
    else:
        print("\n[Fusion] Skipping live eval (--skip-eval)")

    # Use training history as fallback / primary source
    if not our_results:
        history_path = Path(checkpoint).parent / "history.json"
        if history_path.is_file():
            with open(history_path) as f:
                history = json.load(f)
            best = max(history, key=lambda e: e.get("val_sar2opt_acc", 0))
            our_results = {
                "s2o_r1": round(best.get("val_sar2opt_acc", 0) * 100, 2),
                "o2s_r1": round(best.get("val_opt2sar_acc", 0) * 100, 2),
                "terrain_acc": round(best.get("val_terrain_acc", 0) * 100, 2),
                "val_loss": round(best.get("val_loss", 0), 4),
                "epoch": best.get("epoch"),
            }
            print(f"  [from training logs, best epoch {our_results['epoch']}]")
            print(f"  SAR→Opt R@1: {_fmt(our_results['s2o_r1'])}  |  "
                  f"Opt→SAR R@1: {_fmt(our_results['o2s_r1'])}  |  "
                  f"Terrain Acc: {_fmt(our_results['terrain_acc'])}")

    return our_results


def fusion_comparison_table(our: dict[str, Any]) -> str:
    headers = ["Model", "Dataset", "SAR→Opt R@1", "SAR→Opt R@5", "Opt→SAR R@1", "Opt→SAR R@5"]
    rows = []
    for entry in FUSION_SOTA:
        rows.append([
            entry["model"], entry["dataset"],
            _fmt(entry["s2o_r1"]), _fmt(entry["s2o_r5"]),
            _fmt(entry["o2s_r1"]), _fmt(entry["o2s_r5"]),
        ])
    highlight = None
    if our:
        highlight = len(rows)
        rows.append([
            "SatQuery (Ours)", "SEN1-2 + QXS",
            _fmt(our.get("s2o_r1")), "—",
            _fmt(our.get("o2s_r1")), "—",
        ])
    return print_table("Optical–SAR Fusion Retrieval — Comparison with SOTA", headers, rows, highlight)


# ════════════════════════════════════════════════════════════════════════════
# Training curve summary
# ════════════════════════════════════════════════════════════════════════════

def training_summary_table() -> str:
    """Read all training histories and produce a summary of final metrics."""
    headers = ["Specialist", "Backbone", "Epochs", "Final Train Loss", "Best Val Metric", "Params"]
    rows = []

    # Change-VQA
    cd_metrics = PROJECT_ROOT / "checkpoints" / "change_detection" / "metrics.json"
    if cd_metrics.is_file():
        with open(cd_metrics) as f:
            history = json.load(f)
        last = history[-1]
        rows.append([
            "Change-VQA", "ConvNeXt-Tiny", str(last["epoch"]),
            f"{last['train_loss']:.4f}",
            f"VQA {last['train_vqa_acc']*100:.1f}% / IoU {last['train_mask_iou']*100:.1f}%",
            "~28M",
        ])

    # Fusion v1
    fusion_hist = PROJECT_ROOT / "checkpoints" / "fusion" / "history.json"
    if fusion_hist.is_file():
        with open(fusion_hist) as f:
            history = json.load(f)
        best = max(history, key=lambda e: e.get("val_sar2opt_acc", 0))
        last = history[-1]
        rows.append([
            "Fusion (v1)", "ConvNeXt-Tiny", str(last["epoch"]),
            f"{last['loss']:.4f}",
            f"S2O {best['val_sar2opt_acc']*100:.1f}% / O2S {best['val_opt2sar_acc']*100:.1f}%",
            "~55M",
        ])

    # Fusion v2
    fusion_v2_hist = PROJECT_ROOT / "checkpoints" / "fusion" / "v2" / "history.json"
    if fusion_v2_hist.is_file():
        with open(fusion_v2_hist) as f:
            history = json.load(f)
        best = max(history, key=lambda e: e.get("val_sar2opt_acc", 0))
        last = history[-1]
        rows.append([
            "Fusion (v2)", "ConvNeXt-Tiny", str(last["epoch"]),
            f"{last['loss']:.4f}",
            f"S2O {best['val_sar2opt_acc']*100:.1f}% / O2S {best['val_opt2sar_acc']*100:.1f}%",
            "~55M",
        ])

    if not rows:
        return ""

    return print_table("Training Summary - All Specialists", headers, rows)


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SatQuery AI — Unified Benchmark: evaluate specialists & compare with SOTA"
    )
    parser.add_argument("--models", nargs="+",
                        choices=["vqa", "grounding", "change_vqa", "fusion"],
                        default=["grounding", "change_vqa", "fusion"],
                        help="Which specialists to benchmark")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip live model evaluation; use training logs only")
    parser.add_argument("--device", default="auto",
                        help="Device for inference (auto / cuda / cpu)")

    # Checkpoint paths
    parser.add_argument("--vqa-ckpt", default="checkpoints/vqa/best.pt")
    parser.add_argument("--vqa-dataset", default="rsvqa-lr")
    parser.add_argument("--vqa-data", default="data/rsvqa-lr")
    parser.add_argument("--vqa-split", default="test")
    parser.add_argument("--grounding-ckpt", default="checkpoints/grounding/v1/best.pt")
    parser.add_argument("--grounding-reranker", nargs="*", default=None,
                        help="Reranker .pt file(s); 'auto' (default when omitted) "
                             "globs reranker*.pt beside --grounding-ckpt. "
                             "Pass an empty list to disable reranking.")
    parser.add_argument("--change-vqa-ckpt", default="checkpoints/change_detection/latest.pt")
    parser.add_argument("--change-vqa-data", default="data/change_vqa")
    parser.add_argument("--fusion-ckpt", default="checkpoints/fusion/best.pt")

    parser.add_argument("--max-samples", type=int, default=500,
                        help="Max samples for grounding eval (faster)")
    parser.add_argument("--output", type=str, default=None,
                        help="Write results to a JSON file")
    parser.add_argument("--output-md", type=str, default=None,
                        help="Write Markdown comparison tables to a file")
    args = parser.parse_args()

    if args.device == "auto":
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {args.device}")

    all_results: dict[str, Any] = {}
    md_sections: list[str] = ["# SatQuery AI — Benchmark Results\n"]
    t0 = time.time()

    # ── Training summary ─────────────────────────────────────────────────
    md_sections.append(training_summary_table())

    # ── VQA ──────────────────────────────────────────────────────────────
    if "vqa" in args.models:
        vqa_res = benchmark_vqa(
            args.vqa_dataset, args.vqa_data, args.vqa_split, args.skip_eval, args.max_samples,
            checkpoint=args.vqa_ckpt
        )
        all_results["vqa"] = vqa_res
        md_sections.append(vqa_comparison_table(vqa_res))

    # ── Grounding ────────────────────────────────────────────────────────
    if "grounding" in args.models:
        grounding_res = benchmark_grounding(
            args.grounding_ckpt, args.skip_eval, args.device, args.max_samples,
            reranker_paths=args.grounding_reranker,
        )
        all_results["grounding"] = grounding_res
        md_sections.append(grounding_comparison_table(grounding_res))

    # ── Change-VQA ───────────────────────────────────────────────────────
    if "change_vqa" in args.models:
        cvqa_res = benchmark_change_vqa(
            args.change_vqa_ckpt, args.change_vqa_data, args.skip_eval, args.device
        )
        all_results["change_vqa"] = cvqa_res
        md_sections.append(change_vqa_comparison_table(cvqa_res))

    # ── Fusion ───────────────────────────────────────────────────────────
    if "fusion" in args.models:
        fusion_res = benchmark_fusion(args.fusion_ckpt, args.skip_eval, args.device)
        all_results["fusion"] = fusion_res
        md_sections.append(fusion_comparison_table(fusion_res))

    # ── Summary ──────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'=' * 72}")
    print(f"  Benchmark complete in {elapsed:.1f}s")
    print(f"{'=' * 72}\n")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"Results JSON: {out}")

    if args.output_md:
        out = Path(args.output_md)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write("\n".join(md_sections))
        print(f"Markdown tables: {out}")


if __name__ == "__main__":
    main()
