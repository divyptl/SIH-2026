"""
Evaluation harness for RSVQA and VRSBench.

This module does NOT fabricate benchmark numbers. It runs the actual
model over a (pre-flattened) dataset split and reports the metric
computed from real predictions.

Usage (after converting the raw dataset to a flat JSONL manifest — see
`iter_dataset` docstring):

    python -m ml.vqa.evaluate --dataset rsvqa-lr --split test \\
        --data-root /path/to/RSVQA-LR --out docs/vqa_results.md

    python -m ml.vqa.evaluate --dataset vrsbench --split test \\
        --data-root /path/to/VRSBench --task vqa --out docs/vqa_results.md

NOTE: This gives a quick exact-match sanity metric. For VRSBench,
prefer the *official* evaluation scripts (compute_metrics.py /
eval_vqa_gpt.ipynb / eval_caption_gpt.ipynb from the VRSBench repo) for
numbers you intend to report — treat this loop as a smoke test, not a
replacement for the official protocol (Section 32/33 of the project spec).
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from .inference import VQAModel, ModelRequest

logger = logging.getLogger(__name__)


@dataclass
class EvalExample:
    image_path: str
    question: str
    reference_answer: str
    modality: str = "optical"


def iter_dataset(dataset: str, data_root: Path, split: str) -> Iterator[EvalExample]:
    """
    Yields EvalExample records for a supported dataset.

    IMPORTANT: This expects a pre-flattened JSONL manifest at
    `{data_root}/{split}.jsonl`, one record per line:
        {"image": "<relative path>", "question": "...", "answer": "...", "modality": "optical"}

    RSVQA and VRSBench each ship their own raw formats; write a small
    one-off converter per their official documentation (RSVQA project
    page / VRSBench GitHub) rather than re-parsing raw formats here on
    every run. Do NOT mix train/test splits when building the manifest —
    keep the held-out test split untouched (Section 31).
    """
    manifest = data_root / f"{split}.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(
            f"Expected a flattened manifest at {manifest}. Convert the raw "
            f"{dataset} '{split}' split into JSONL lines of "
            '{"image": ..., "question": ..., "answer": ...} first.'
        )
    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            yield EvalExample(
                image_path=str(data_root / rec["image"]),
                question=rec["question"],
                reference_answer=rec["answer"],
                modality=rec.get("modality", "optical"),
            )


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def run_eval(
    dataset: str,
    data_root: Path,
    split: str,
    task: str = "vqa",
    limit: Optional[int] = None,
) -> dict:
    model = VQAModel()
    total = 0
    correct = 0
    records = []

    for i, ex in enumerate(iter_dataset(dataset, data_root, split)):
        if limit is not None and i >= limit:
            break
        request = ModelRequest(
            query=ex.question,
            images=[ex.image_path],
            modalities=[ex.modality],
            task_hint=task,
        )
        response = model.predict(request)
        is_correct = _normalize(response.answer) == _normalize(ex.reference_answer)
        total += 1
        correct += int(is_correct)
        records.append(
            {
                "image": ex.image_path,
                "question": ex.question,
                "reference": ex.reference_answer,
                "prediction": response.answer,
                "confidence": response.confidence,
                "correct": is_correct,
            }
        )

    accuracy = correct / total if total else 0.0
    return {
        "dataset": dataset,
        "split": split,
        "task": task,
        "n_examples": total,
        "accuracy_exact_match": accuracy,
        "records": records,
    }


def write_report(results: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(f"\n## {results['dataset']} ({results['split']}, task={results['task']})\n\n")
        f.write(f"- Examples evaluated: {results['n_examples']}\n")
        f.write(f"- Exact-match accuracy: {results['accuracy_exact_match']:.4f}\n")
        f.write(
            "- Metric note: exact-match string accuracy is a coarse proxy "
            "computed by this script. For VRSBench, prefer the official "
            "GPT-based VQA/caption evaluation notebooks for the number you "
            "actually report.\n"
        )


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run RSVQA/VRSBench evaluation")
    p.add_argument("--dataset", required=True, choices=["rsvqa-lr", "rsvqa-hr", "vrsbench"])
    p.add_argument("--split", default="test")
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--task", default="vqa", choices=["vqa", "captioning"])
    p.add_argument(
        "--limit", type=int, default=None,
        help="Evaluate only the first N examples (smoke test before a full run)",
    )
    p.add_argument("--out", type=Path, default=Path("docs/vqa_results.md"))
    return p


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO)
    args = _build_arg_parser().parse_args(argv)
    results = run_eval(args.dataset, args.data_root, args.split, task=args.task, limit=args.limit)
    write_report(results, args.out)
    logger.info("Wrote evaluation report to %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
