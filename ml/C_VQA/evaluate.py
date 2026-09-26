"""
Evaluate a Change-VQA checkpoint on the held-out split of a prepared dataset.

Reports answer accuracy and change-mask IoU / F1 per source, per question type
and per resolution band, so it is clear where the model can be trusted rather
than one blended number. The mask is scored once per tile, not once per question.

Usage:
    python -m ml.C_VQA.evaluate --checkpoint checkpoints/change_detection/best.pt \\
        --data-root data/change_vqa --split test --out results/change_vqa_test.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ml.C_VQA.dataset import CDVQADataset, cdvqa_collate_fn
from ml.C_VQA.inference import ChangeVQAModel

# Resolution bands (m/pixel) the report groups by.
GSD_BANDS = [(0.0, 1.0, "<1 m"), (1.0, 3.0, "1-3 m"), (3.0, 8.0, "3-8 m"), (8.0, 1e9, ">=8 m")]


def _band(gsd: float | None) -> str:
    if gsd is None:
        return "unknown"
    return next(name for low, high, name in GSD_BANDS if low <= gsd < high)


class _Tally:
    def __init__(self) -> None:
        self.correct = self.answered = 0
        self.tp = self.fp = self.fn = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "questions": self.answered,
            "vqa_acc": round(self.correct / self.answered, 4) if self.answered else None,
            "mask_iou": round(self.tp / (self.tp + self.fp + self.fn), 4) if self.tp + self.fp + self.fn else None,
            "mask_f1": round(2 * self.tp / (2 * self.tp + self.fp + self.fn), 4) if self.tp + self.fp + self.fn else None,
        }


@torch.no_grad()
def evaluate(checkpoint: str, data_root: str, split: str, batch_size: int, device: str) -> dict[str, Any]:
    specialist = ChangeVQAModel.from_checkpoint(checkpoint, device=device)
    model = specialist.model
    dataset = CDVQADataset(
        root=data_root,
        split=split,
        question_vocab=model.tokenizer.vocab,
        answer_vocab=list(model.answers_vocab),
        max_question_length=specialist.config.max_question_length,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=cdvqa_collate_fn)

    tallies: dict[str, _Tally] = defaultdict(_Tally)
    scored_tiles: set[str] = set()
    for batch in loader:
        outputs = model(
            t1=batch["t1"].to(specialist.device),
            t2=batch["t2"].to(specialist.device),
            question_ids=batch["question_ids"].to(specialist.device),
        )
        predictions = outputs["answer_logits"].argmax(dim=-1).cpu()
        masks = (outputs["change_mask_prob"].float().cpu() > specialist.config.mask_threshold)
        targets = batch["mask_targets"]

        for i, tile in enumerate(batch["tile_ids"]):
            groups = [
                "overall",
                f"source:{batch['sources'][i]}",
                f"question:{batch['question_types'][i]}",
                f"gsd:{_band(batch['gsds'][i])}",
            ]
            target = int(batch["answer_targets"][i])
            for group in groups:
                if target != -100:
                    tallies[group].answered += 1
                    tallies[group].correct += int(predictions[i] == target)
            # Every question of a tile shares its mask; score it once.
            if targets is not None and tile not in scored_tiles:
                scored_tiles.add(tile)
                pred, truth = masks[i, 0], targets[i, 0] > 0.5
                tp, fp, fn = float((pred & truth).sum()), float((pred & ~truth).sum()), float((~pred & truth).sum())
                for group in groups:
                    if group.startswith("question:"):
                        continue
                    tallies[group].tp += tp
                    tallies[group].fp += fp
                    tallies[group].fn += fn

    return {
        "checkpoint": checkpoint,
        "split": split,
        "gsd_range_m": list(specialist.gsd_range_m),
        "results": {group: tally.as_dict() for group, tally in sorted(tallies.items())},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a Change-VQA checkpoint per source, question type and GSD")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="data/change_vqa")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default=None, help="Write the report as JSON here")
    args = parser.parse_args(argv)

    report = evaluate(args.checkpoint, args.data_root, args.split, args.batch_size, args.device)
    print(f"{'group':38} {'questions':>9} {'VQA acc':>8} {'mask IoU':>9} {'mask F1':>8}")
    for group, r in report["results"].items():
        fmt = lambda v: f"{v:.3f}" if v is not None else "-"  # noqa: E731
        print(f"{group:38} {r['questions']:>9} {fmt(r['vqa_acc']):>8} {fmt(r['mask_iou']):>9} {fmt(r['mask_f1']):>8}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
