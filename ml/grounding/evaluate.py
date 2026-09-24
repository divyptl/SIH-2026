"""
Accuracy evaluation for the grounding model on VRSBench.

Reports the metrics VRSBench uses for visual grounding: the share of referring
expressions whose top-scoring predicted box overlaps the ground-truth box at
IoU >= 0.5 and IoU >= 0.7, split by whether the referred object is the only
one of its class in the image ("unique") or not. Mean IoU is reported too.

Note that VRSBench ships a single evaluation file, which training also uses as
its validation set to pick `best.pt`, so these numbers are slightly optimistic.

Usage:
    # Fine-tuned checkpoint
    python -m ml.grounding.evaluate --checkpoint checkpoints/grounding/best.pt

    # Zero-shot baseline, for comparison
    python -m ml.grounding.evaluate --pretrained

    # Quick check on a subset, saving per-sample results
    python -m ml.grounding.evaluate --checkpoint checkpoints/grounding/best.pt \
        --max-samples 1000 --output results.json

    # Two-stage: GroundingDINO candidates chosen between by the re-ranker
    python -m ml.grounding.evaluate --checkpoint checkpoints/grounding/best.pt \
        --reranker checkpoints/grounding/reranker.pt
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import box_convert, box_iou

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.grounding.config import ModelConfig, TrainConfig
from ml.grounding.dataset import VRSBenchGroundingDataset, collate_fn
from ml.grounding.model import GroundingModel
from ml.grounding.rerank import CandidateReranker, load_reranker, rerank_outputs
from ml.grounding.train import resolve_amp
from ml.grounding.transforms import prepare_training_batch

IOU_THRESHOLDS = (0.5, 0.7)


class _Indexed(Dataset):
    """Yields (index, sample) so predictions can be matched to sample metadata."""

    def __init__(self, dataset: VRSBenchGroundingDataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        return idx, self.dataset[idx]


def _collate(batch):
    out = collate_fn([item for _, item in batch])
    out["indices"] = [idx for idx, _ in batch]
    return out


def load_model(checkpoint: str | None) -> tuple[GroundingModel, str]:
    """Load a fine-tuned checkpoint, or the pre-trained model when None."""
    if checkpoint is None:
        config = ModelConfig(freeze_backbone=False)
        return GroundingModel(config), f"{config.model_id} (zero-shot)"

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_id = ckpt.get("model_config", {}).get("model_id", ModelConfig.model_id)
    grounding = GroundingModel(ModelConfig(model_id=model_id, freeze_backbone=False))
    grounding.model.load_state_dict(ckpt["model"])
    return grounding, f"{checkpoint} (epoch {ckpt.get('epoch', '?')})"


@torch.no_grad()
def predict(
    grounding: GroundingModel,
    dataset: VRSBenchGroundingDataset,
    device: str,
    batch_size: int,
    num_workers: int,
    image_size: int,
    amp_dtype: torch.dtype,
    reranker: CandidateReranker | None = None,
) -> list[dict]:
    """Run the model over the dataset and score its top-1 box per expression.

    With a re-ranker, the top-1 box is the re-ranker's pick among
    GroundingDINO's candidates rather than GroundingDINO's most confident query.
    """
    grounding.model.eval()
    loader = DataLoader(
        _Indexed(dataset),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate,
    )

    records: list[dict] = []
    t0 = time.time()
    for step, batch in enumerate(loader):
        inputs, labels = prepare_training_batch(
            batch, grounding.processor, augmentation=None, device=device,
            image_size=image_size,
        )
        with torch.autocast(device_type=device, dtype=amp_dtype,
                            enabled=amp_dtype != torch.float32):
            outputs = grounding(
                pixel_values=inputs["pixel_values"],
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                token_type_ids=inputs.get("token_type_ids"),
            )

        # Each of the 900 queries scores against every text token; padded
        # tokens are -inf. A query's confidence is its best token, and the
        # prediction for the expression is the most confident query's box.
        if reranker is not None:
            picked = rerank_outputs(
                reranker, outputs, inputs["pixel_values"], inputs["input_ids"],
                inputs["attention_mask"],
            )
            best_score = picked["probs"][:, 0]
            pred_boxes = picked["boxes"][:, 0]
        else:
            scores = outputs.logits.float().sigmoid().max(dim=-1).values   # (B, Q)
            best_score, best_query = scores.max(dim=-1)                     # (B,)
            rows = torch.arange(len(best_query), device=best_query.device)
            pred_boxes = outputs.pred_boxes.float()[rows, best_query]       # (B, 4) cxcywh

        # IoU is unchanged by scaling each axis, so normalized coordinates
        # give the same value as pixel coordinates.
        pred_xyxy = box_convert(pred_boxes, "cxcywh", "xyxy")
        for j, idx in enumerate(batch["indices"]):
            gt_xyxy = box_convert(labels[j]["boxes"].float(), "cxcywh", "xyxy")
            iou = box_iou(pred_xyxy[j:j + 1], gt_xyxy).max().item()
            sample = dataset.samples[idx]
            records.append({
                "image": sample["image_name"],
                "text": sample["text"],
                "unique": sample.get("unique"),
                "obj_cls": sample.get("obj_cls"),
                "iou": round(iou, 4),
                "score": round(best_score[j].item(), 4),
                "pred_box_xyxy": [round(v, 4) for v in pred_xyxy[j].tolist()],
                "gt_box_xyxy": [round(v, 4) for v in gt_xyxy[0].tolist()],
            })

        if (step + 1) % 50 == 0:
            done = len(records)
            rate = done / (time.time() - t0)
            eta = (len(dataset) - done) / max(rate, 1e-9)
            print(f"  [{done}/{len(dataset)}] {rate:.1f} samples/s, ~{eta / 60:.0f} min left")

    return records


def summarize(records: list[dict]) -> dict:
    """Accuracy at each IoU threshold and mean IoU, overall and by subset."""

    def metrics(rs: list[dict]) -> dict:
        n = len(rs)
        out = {"count": n}
        for thr in IOU_THRESHOLDS:
            out[f"acc@{thr}"] = sum(r["iou"] >= thr for r in rs) / n if n else 0.0
        out["mean_iou"] = sum(r["iou"] for r in rs) / n if n else 0.0
        return out

    groups = {
        "all": records,
        "unique": [r for r in records if r["unique"] is True],
        "non_unique": [r for r in records if r["unique"] is False],
    }
    classes = sorted({r["obj_cls"] for r in records if r["obj_cls"]})
    return {
        "overall": {name: metrics(rs) for name, rs in groups.items() if rs},
        "per_class": {
            cls: metrics([r for r in records if r["obj_cls"] == cls]) for cls in classes
        },
    }


def print_summary(summary: dict, model_name: str) -> None:
    print("\n" + "=" * 70)
    print(f"  VRSBench grounding accuracy — {model_name}")
    print("=" * 70)
    header = f"  {'subset':<18}{'count':>8}{'Acc@0.5':>10}{'Acc@0.7':>10}{'mIoU':>8}"
    print(header)
    for name, m in summary["overall"].items():
        print(f"  {name:<18}{m['count']:>8}{m['acc@0.5']:>10.1%}"
              f"{m['acc@0.7']:>10.1%}{m['mean_iou']:>8.3f}")

    if summary["per_class"]:
        print("\n  per class")
        print(header)
        by_count = sorted(summary["per_class"].items(), key=lambda kv: -kv[1]["count"])
        for cls, m in by_count:
            print(f"  {cls:<18}{m['count']:>8}{m['acc@0.5']:>10.1%}"
                  f"{m['acc@0.7']:>10.1%}{m['mean_iou']:>8.3f}")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate grounding accuracy on VRSBench")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=str, help="Fine-tuned .pt checkpoint")
    source.add_argument("--pretrained", action="store_true",
                        help="Evaluate the zero-shot pre-trained model")
    parser.add_argument("--reranker", type=str, default=None,
                        help="Re-ranker .pt from train_rerank.py; picks among the "
                             "checkpoint's candidates")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None,
                        help="Must match training (default from TrainConfig)")
    parser.add_argument("--image-dir", type=str, default=None,
                        help="Directory of extracted VRSBench validation images")
    parser.add_argument("--image-zip", type=str, default=None,
                        help="Images_val.zip you downloaded yourself; read "
                             "directly, no unpacking needed")
    parser.add_argument("--annotations", type=str, default=None,
                        help="VRSBench_EVAL_referring.json you downloaded yourself")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Evaluate N expressions spread evenly across the file")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=str, default=None,
                        help="Write the summary and per-sample results to this JSON file")
    args = parser.parse_args()

    train_cfg = TrainConfig(device=args.device)
    device = train_cfg.resolve_device()
    _, amp_dtype = resolve_amp(device, not args.no_amp)
    image_size = args.image_size or train_cfg.image_size
    num_workers = args.num_workers
    if num_workers is None:
        num_workers = 0 if platform.system() == "Windows" else train_cfg.num_workers

    print(f"Device: {device}  |  precision: {amp_dtype}  |  image size: {image_size}")

    dataset = VRSBenchGroundingDataset(
        split="validation",
        cache_dir=train_cfg.data_cache_dir,
        image_dir=args.image_dir,
        annotations_file=args.annotations,
        image_zip=args.image_zip,
    )
    if args.max_samples and args.max_samples < len(dataset):
        # The file is ordered by image, so a leading slice covers only a few
        # scenes and classes. Take evenly spaced expressions instead.
        stride = len(dataset) / args.max_samples
        dataset.samples = [dataset.samples[int(i * stride)] for i in range(args.max_samples)]
        print(f"  Evaluating an evenly spaced subset of {len(dataset)} expressions")

    grounding, model_name = load_model(None if args.pretrained else args.checkpoint)
    grounding.model.to(device)
    reranker = None
    if args.reranker:
        if args.pretrained:
            parser.error("--reranker needs the --checkpoint it was trained on")
        reranker = load_reranker(args.reranker, device)
        trained_on = reranker.detector_checkpoint
        if trained_on and Path(trained_on).resolve() != Path(args.checkpoint).resolve():
            print(f"  Warning: re-ranker was trained on candidates from {trained_on}")
        model_name += f" + re-ranker {args.reranker}"
    print(f"Loaded {model_name}\n")

    records = predict(
        grounding, dataset, device, args.batch_size, num_workers, image_size, amp_dtype,
        reranker=reranker,
    )
    summary = summarize(records)
    print_summary(summary, model_name)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"model": model_name, "summary": summary, "samples": records}, f, indent=2)
        print(f"Results written to {out}")


if __name__ == "__main__":
    main()
