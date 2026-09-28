"""
Evaluate the single-image VQA specialist.

Scores a checkpoint on a prepared manifest (see ml.vqa.training.prepare_vrsbench)
with exact-match accuracy, overall and per question type. For VRSBench the
`test` split is the official VQA evaluation set (37,409 questions).

    python -m ml.vqa.evaluate --data-root data/vrsbench_vqa --split test

Exact match is the right metric for this model, which answers from a fixed
vocabulary; VRSBench's paper scores open-ended LLM answers with a GPT judge
instead, so its numbers are not directly comparable.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from ml.vqa.config import TrainConfig
from ml.vqa.inference import DEFAULT_CHECKPOINT
from ml.vqa.model import VQAModel as CoreVQAModel
from ml.vqa.training.dataset import VQADataset
from ml.vqa.training.train import amp_dtype, evaluate

logger = logging.getLogger(__name__)


def run_eval(
    dataset: str,
    data_root: Path,
    split: str,
    task: str = "vqa",
    limit: Optional[int] = None,
    checkpoint: Optional[str] = None,
    batch_size: int = 64,
    num_workers: int = 4,
) -> dict:
    """Accuracy of a checkpoint on `{data_root}/{split}.jsonl`.

    Returns {"dataset", "split", "task", "n_examples", "accuracy_exact_match",
    "per_type": {type: {"count", "accuracy"}}}.
    """
    if task != "vqa":
        raise ValueError("The single-image VQA model answers questions only (task='vqa').")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CoreVQAModel.from_checkpoint(checkpoint or DEFAULT_CHECKPOINT, device=device)
    cfg = TrainConfig(image_size=getattr(model, "image_size", 256), augment=False)
    ds = VQADataset(Path(data_root), split, model.answers_vocab, model.tokenizer, cfg,
                    max_question_length=model.config.max_question_length)
    if limit is not None:
        ds.samples = ds.samples[:limit]
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    metrics = evaluate(model, loader, device, amp_dtype(device))
    return {
        "dataset": dataset,
        "split": split,
        "task": task,
        "n_examples": metrics["all"]["count"],
        "accuracy_exact_match": metrics["all"]["accuracy"],
        "per_type": {k: v for k, v in metrics.items() if k != "all"},
    }


def write_report(results: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(f"\n## {results['dataset']} ({results['split']}, task={results['task']})\n\n")
        f.write(f"- Examples evaluated: {results['n_examples']}\n")
        f.write(f"- Exact-match accuracy: {results['accuracy_exact_match']:.4f}\n")
        if results.get("per_type"):
            f.write("\n| Question type | Count | Accuracy |\n|---|---|---|\n")
            for name, m in sorted(results["per_type"].items(), key=lambda kv: -kv[1]["count"]):
                f.write(f"| {name} | {m['count']} | {m['accuracy']:.1%} |\n")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Evaluate the single-image VQA model")
    p.add_argument("--dataset", default="vrsbench")
    p.add_argument("--data-root", type=Path, default=Path("data/vrsbench_vqa"))
    p.add_argument("--split", default="test")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--limit", type=int, default=None, help="Evaluate only the first N questions")
    p.add_argument("--out", type=Path, default=None, help="Append a Markdown report here")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    results = run_eval(args.dataset, args.data_root, args.split, limit=args.limit,
                       checkpoint=args.checkpoint)
    print(f"{args.dataset} {args.split}: {results['n_examples']} questions, "
          f"exact-match accuracy {results['accuracy_exact_match']:.1%}")
    for name, m in sorted(results["per_type"].items(), key=lambda kv: -kv[1]["count"]):
        print(f"  {name:<20}{m['count']:>7}{m['accuracy']:>10.1%}")
    if args.out:
        write_report(results, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
