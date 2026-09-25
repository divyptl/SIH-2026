"""
DIOR-RSVG grounding dataset.

DIOR-RSVG (Zhan et al., TGRS 2023) pairs 800x800 DIOR optical images with short
referring expressions, one box each: ~27K train, ~3.8K val and 7.5K test
expressions over 20 object classes. It is the most widely reported remote-sensing
visual grounding benchmark, and adds training data beside VRSBench.

The Hugging Face copy (`danielz01/DIOR-RSVG`, gated: accept its terms and log in
with `hf auth login` first) stores one row per image and split, with the image
bytes and a list of (box, caption) pairs. The official split is over
expressions, not images, so one photo can appear in several splits; that is the
benchmark's protocol and is kept as-is.

`prepare()` downloads the parquet shards once, writes each photo to
`<root>/images/` and one annotation file per split to `<root>/annotations/`, then
deletes the shards. `DIORRSVGDataset` reads those files and yields samples in the
same format as `VRSBenchGroundingDataset`, so the two can be mixed freely.

    python -m ml.grounding.dior_rsvg            # prepare all splits (~8 GB download)
"""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

try:
    from PIL import Image
except ImportError:
    Image = None

HF_REPO_ID = "danielz01/DIOR-RSVG"
SPLITS = ("train", "val", "test")
DEFAULT_ROOT = "data/dior_rsvg"


def _split_files(split: str) -> list[str]:
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(HF_REPO_ID, repo_type="dataset")
    return sorted(f for f in files if f.startswith(f"data/{split}-") and f.endswith(".parquet"))


def prepare(root: str = DEFAULT_ROOT, splits: tuple[str, ...] = SPLITS) -> None:
    """Download DIOR-RSVG and unpack it into images + per-split annotation files.

    Idempotent: a split whose annotation file already exists is skipped, and an
    image already on disk is not rewritten.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    root_path = Path(root)
    image_dir = root_path / "images"
    ann_dir = root_path / "annotations"
    shard_dir = root_path / "parquet"
    image_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        ann_path = ann_dir / f"{split}.json"
        if ann_path.exists():
            continue
        files = _split_files(split)
        if not files:
            raise RuntimeError(f"No parquet files for split '{split}' in {HF_REPO_ID}")

        samples: list[dict] = []
        skipped = 0
        for n, remote in enumerate(files, 1):
            print(f"  DIOR-RSVG [{split}] shard {n}/{len(files)}: downloading...", flush=True)
            local = Path(hf_hub_download(
                HF_REPO_ID, remote, repo_type="dataset", local_dir=str(shard_dir),
            ))
            for batch in pq.ParquetFile(local).iter_batches(batch_size=64):
                for row in batch.to_pylist():
                    name = row["path"]
                    data = row["image"]["bytes"]
                    target = image_dir / name
                    if not target.exists():
                        tmp = target.with_suffix(target.suffix + ".tmp")
                        tmp.write_bytes(data)
                        os.replace(tmp, target)
                    width, height = Image.open(io.BytesIO(data)).size
                    objects = row["objects"]
                    for box, caption, category in zip(
                        objects["bbox"], objects["captions"], objects["categories"],
                    ):
                        x1, y1, x2, y2 = (int(v) for v in box)
                        text = (caption or "").strip()
                        if not text or not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                            skipped += 1
                            continue
                        samples.append({
                            "image": name, "text": text, "box_xyxy": [x1, y1, x2, y2],
                            "width": width, "height": height, "category": category,
                        })
            local.unlink()   # the shard is fully unpacked; free ~400 MB
        tmp = ann_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(samples), encoding="utf-8")
        os.replace(tmp, ann_path)
        print(f"  DIOR-RSVG [{split}]: {len(samples)} expressions"
              + (f", {skipped} skipped (empty text or box outside the image)" if skipped else ""))


class DIORRSVGDataset(Dataset):
    """DIOR-RSVG referring expressions, in VRSBenchGroundingDataset's format.

    Each sample returns:
        image: PIL.Image.Image (RGB)
        text:  str — the referring expression
        boxes: Tensor (1, 4) — (cx, cy, w, h) normalized 0–1
        class_labels: Tensor (1,) — zeros
        image_name: str

    Args:
        split: 'train', 'val' or 'test'.
        root: Directory holding (or to receive) the prepared data.
        download: Run `prepare()` for this split if it is missing.
        max_samples: Keep only the first N expressions (for debugging).
    """

    def __init__(
        self,
        split: str = "train",
        root: str = DEFAULT_ROOT,
        download: bool = True,
        max_samples: int | None = None,
    ) -> None:
        super().__init__()
        if split not in SPLITS:
            raise ValueError(f"Unknown DIOR-RSVG split '{split}'. Expected one of {SPLITS}.")
        self.split = split
        self.root = Path(root)
        ann_path = self.root / "annotations" / f"{split}.json"
        if not ann_path.exists():
            if not download:
                raise FileNotFoundError(
                    f"{ann_path} is missing. Run `python -m ml.grounding.dior_rsvg` first."
                )
            prepare(root, (split,))

        with open(ann_path, encoding="utf-8") as f:
            raw = json.load(f)

        self.samples: list[dict] = []
        for s in raw:
            x1, y1, x2, y2 = s["box_xyxy"]
            w, h = s["width"], s["height"]
            self.samples.append({
                "text": s["text"],
                "boxes": [[(x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h]],
                "image_name": s["image"],
                # DIOR-RSVG has no unique/non-unique split; the class name is
                # kept for the per-class breakdown.
                "unique": None,
                "obj_cls": s["category"],
                "source": "dior_rsvg",
            })
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        print(f"  DIOR-RSVG grounding [{split}]: {len(self.samples)} samples loaded")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if Image is None:
            raise ImportError("Pillow is required: pip install Pillow")
        sample = self.samples[idx]
        image = Image.open(self.root / "images" / sample["image_name"]).convert("RGB")
        boxes = torch.tensor(sample["boxes"], dtype=torch.float32)
        return {
            "image": image,
            "text": sample["text"],
            "boxes": boxes,
            "class_labels": torch.zeros(len(boxes), dtype=torch.long),
            "image_name": sample["image_name"],
        }


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import argparse

    parser = argparse.ArgumentParser(description="Download and unpack DIOR-RSVG")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    args = parser.parse_args()
    prepare(args.root, tuple(args.splits))
    for split in args.splits:
        DIORRSVGDataset(split, args.root, download=False)
