"""
Cache GroundingDINO candidates for training the re-ranker.

Runs the frozen, fine-tuned GroundingDINO once over a VRSBench split and stores
each expression's top-K candidates (see `rerank.extract_candidates`) as .npy
arrays, so the re-ranker trains on features instead of re-running the detector.

Usage:
    python -m ml.grounding.build_rerank_cache --checkpoint checkpoints/grounding/best.pt --split train
    python -m ml.grounding.build_rerank_cache --checkpoint checkpoints/grounding/best.pt --split validation

Roughly 20 samples/s on a laptop RTX 5070 Ti: about an hour for train
(~72K expressions) and 15 minutes for validation (~16K).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.grounding.config import RerankConfig, TrainConfig
from ml.grounding.dataset import VRSBenchGroundingDataset
from ml.grounding.evaluate import _collate, _Indexed, load_model
from ml.grounding.rerank import extract_candidates
from ml.grounding.train import resolve_amp
from ml.grounding.transforms import prepare_training_batch

# name → (dtype, per-sample shape given K, T, D)
FEATURES = {
    "hidden": (np.float16, lambda k, t, d: (k, d)),
    "roi": (np.float16, lambda k, t, d: (k, d)),
    "boxes": (np.float32, lambda k, t, d: (k, 4)),
    "scores": (np.float32, lambda k, t, d: (k,)),
    "cand_mask": (np.bool_, lambda k, t, d: (k,)),
    "tok_logits": (np.float16, lambda k, t, d: (k, t)),
    "text": (np.float16, lambda k, t, d: (t, d)),
    "text_mask": (np.bool_, lambda k, t, d: (t,)),
}


def load_split(split: str, cfg: TrainConfig) -> VRSBenchGroundingDataset:
    """Load a split the same way training and evaluation do."""
    return VRSBenchGroundingDataset(
        data_name=cfg.data_name,
        split=split,
        cache_dir=cfg.data_cache_dir,
        image_dir=cfg.image_dir,
        download_images=cfg.download_images,
        auto_extract_zip=cfg.auto_extract_zip,
        extracted_image_dir=cfg.extracted_image_dir if split == "train" else None,
        annotations_file=cfg.annotations_file if split == "train" else cfg.val_annotations_file,
        image_zip=cfg.image_zip if split == "train" else cfg.val_image_zip,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache GroundingDINO candidates for the re-ranker")
    parser.add_argument("--checkpoint", required=True, help="Fine-tuned GroundingDINO .pt")
    parser.add_argument("--split", choices=["train", "validation"], required=True)
    parser.add_argument("--out-dir", default=None, help="Default: RerankConfig.cache_dir")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None, help="For quick tests")
    args = parser.parse_args()

    rcfg = RerankConfig()
    tcfg = TrainConfig()
    device = tcfg.resolve_device()
    _, amp_dtype = resolve_amp(device, True)
    out_dir = Path(args.out_dir or rcfg.cache_dir) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_split(args.split, tcfg)
    if args.max_samples and args.max_samples < len(dataset):
        stride = len(dataset) / args.max_samples
        dataset.samples = [dataset.samples[int(i * stride)] for i in range(args.max_samples)]
    n = len(dataset)

    grounding, name = load_model(args.checkpoint)
    grounding.model.to(device).eval()
    print(f"Loaded {name}; caching {n} expressions to {out_dir}")

    k, t = rcfg.cache_k, rcfg.max_text_tokens
    d = grounding.model.config.d_model
    arrays = {
        key: np.lib.format.open_memmap(
            out_dir / f"{key}.npy", mode="w+", dtype=dtype, shape=(n, *shape(k, t, d)),
        )
        for key, (dtype, shape) in FEATURES.items()
    }
    gt = np.lib.format.open_memmap(out_dir / "gt.npy", mode="w+", dtype=np.float32, shape=(n, 4))

    workers = args.num_workers
    loader = DataLoader(
        _Indexed(dataset), batch_size=args.batch_size, shuffle=False,
        num_workers=workers, collate_fn=_collate,
        persistent_workers=workers > 0, prefetch_factor=4 if workers > 0 else None,
    )

    t0 = time.time()
    done = 0
    with torch.no_grad():
        for step, batch in enumerate(loader):
            inputs, labels = prepare_training_batch(
                batch, grounding.processor, augmentation=None, device=device,
                image_size=tcfg.image_size,
            )
            with torch.autocast(device_type=device, dtype=amp_dtype,
                                enabled=amp_dtype != torch.float32):
                outputs = grounding(
                    pixel_values=inputs["pixel_values"],
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    token_type_ids=inputs.get("token_type_ids"),
                )
            cands = extract_candidates(
                outputs, inputs["pixel_values"], inputs["attention_mask"],
                k=k, nms_iou=rcfg.nms_iou, max_text_tokens=t,
            )
            rows = np.asarray(batch["indices"])
            for key, array in arrays.items():
                array[rows] = cands[key].cpu().numpy().astype(array.dtype)
            # Every VRSBench expression refers to a single object.
            gt[rows] = torch.stack([lbl["boxes"][0] for lbl in labels]).cpu().numpy()

            done += len(rows)
            if (step + 1) % 100 == 0:
                rate = done / (time.time() - t0)
                print(f"  [{done}/{n}] {rate:.1f} samples/s, ~{(n - done) / rate / 60:.0f} min left")

    for array in (*arrays.values(), gt):
        array.flush()

    meta = [
        {
            "image": s["image_name"],
            "text": s["text"],
            "unique": s.get("unique"),
            "obj_cls": s.get("obj_cls"),
        }
        for s in dataset.samples
    ]
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "checkpoint": args.checkpoint,
            "split": args.split,
            "cache_k": k,
            "max_text_tokens": t,
            "nms_iou": rcfg.nms_iou,
            "samples": meta,
        }, f)
    print(f"Done: {n} expressions in {(time.time() - t0) / 60:.1f} min → {out_dir}")


if __name__ == "__main__":
    main()
