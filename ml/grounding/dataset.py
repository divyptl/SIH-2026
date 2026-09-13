"""
VRSBench grounding dataset loader.

The HuggingFace repo `xiang709/VRSBench` is a plain file repo — no loading
script, no named builder configs — so `load_dataset(..., name="VRSBench")`
(as the dataset card still suggests) fails with
`BuilderConfig 'VRSBench' not found. Available: ['default']`. The annotation
files and image archives are therefore pulled off the Hub directly:

    train       VRSBench_train.json            LLaVA-style conversations; the
                                               grounding samples are the turns
                                               tagged `[refer]`
    validation  VRSBench_EVAL_referring.json   one referring expression per entry
    images      Images_train.zip (8.4 GB) / Images_val.zip (4.0 GB)

VRSBench writes boxes as `{<x1><y1><x2><y2>}` normalized to 0–100; this loader
rescales them to 0–1 in (cx, cy, w, h) format as expected by GroundingDINO.
"""

from __future__ import annotations

import json
import os
import re
import sys
import zipfile
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None

try:
    from PIL import Image
except ImportError:
    Image = None


HF_REPO_ID = "xiang709/VRSBench"

# split → (annotation file, image archive) in the Hub repo
SPLIT_FILES: dict[str, tuple[str, str]] = {
    "train": ("VRSBench_train.json", "Images_train.zip"),
    "validation": ("VRSBench_EVAL_referring.json", "Images_val.zip"),
    "test": ("VRSBench_EVAL_referring.json", "Images_val.zip"),
}

# approximate archive sizes, only used to warn before a large download
ARCHIVE_SIZES_GB = {"Images_train.zip": 8.4, "Images_val.zip": 4.0}

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# "{<25><40><33><60>}" → four 0–100 coordinates
_BOX_RE = re.compile(
    r"\{\s*<\s*(-?[\d.]+)\s*>\s*<\s*(-?[\d.]+)\s*>"
    r"\s*<\s*(-?[\d.]+)\s*>\s*<\s*(-?[\d.]+)\s*>\s*\}"
)
# the referring expression is wrapped in <p>...</p> in the train conversations
_EXPR_RE = re.compile(r"<p>(.*?)</p>", re.DOTALL)


class _ImageSource:
    """Resolves an image file name to a PIL image.

    Images come either from a local directory (searched recursively) or from
    the VRSBench zip archive downloaded from the Hub — reading straight out of
    the zip avoids a second, unpacked copy of ~8 GB on disk. Zip handles are
    opened lazily and per-process so each DataLoader worker gets its own.
    """

    def __init__(self, image_dir: str | None = None, zip_path: str | None = None) -> None:
        if image_dir is None and zip_path is None:
            raise ValueError("_ImageSource needs either image_dir or zip_path")
        self.image_dir = Path(image_dir) if image_dir else None
        self.zip_path = Path(zip_path) if zip_path else None
        self._index: dict[str, str] | None = None
        self._zips: dict[int, zipfile.ZipFile] = {}

    def __getstate__(self) -> dict:
        # ZipFile handles don't survive pickling into DataLoader workers
        state = self.__dict__.copy()
        state["_zips"] = {}
        return state

    def _zip(self) -> zipfile.ZipFile:
        if self.zip_path is None:
            raise RuntimeError("_ImageSource has no zip archive")
        pid = os.getpid()
        zf = self._zips.get(pid)
        if zf is None:
            zf = zipfile.ZipFile(self.zip_path)
            self._zips[pid] = zf
        return zf

    def index(self) -> dict[str, str]:
        """Map bare file name → locator, built once per process.

        The archives nest images under a folder whose name has changed between
        VRSBench releases, so match on the file name rather than a fixed prefix.
        """
        if self._index is None:
            if self.image_dir is not None:
                self._index = {
                    p.name: str(p)
                    for p in self.image_dir.rglob("*")
                    if p.suffix.lower() in _IMAGE_SUFFIXES
                }
            else:
                self._index = {
                    name.rsplit("/", 1)[-1]: name
                    for name in self._zip().namelist()
                    if not name.endswith("/")
                    and Path(name).suffix.lower() in _IMAGE_SUFFIXES
                }
        return self._index

    def has(self, name: str) -> bool:
        return name in self.index()

    def open(self, name: str):
        if Image is None:
            raise ImportError("Pillow is required: pip install Pillow")
        locator = self.index().get(name)
        if locator is None:
            raise FileNotFoundError(f"Image '{name}' not found in {self.describe()}")
        if self.image_dir is not None:
            return Image.open(locator).convert("RGB")
        with self._zip().open(locator) as fh:
            return Image.open(fh).convert("RGB")

    def describe(self) -> str:
        return str(self.image_dir if self.image_dir is not None else self.zip_path)


class VRSBenchGroundingDataset(Dataset):
    """PyTorch dataset for VRSBench grounding (object referring) annotations.

    Each sample returns:
        image: PIL.Image.Image (RGB)
        text:  str — the grounding expression
        boxes: Tensor (N, 4) — boxes in (cx, cy, w, h) normalized 0–1
        class_labels: Tensor (N,) — all zeros (single-class grounding)

    Args:
        data_name: HuggingFace repo id holding the VRSBench files.
        split: 'train', 'validation', or 'test' ('test' reuses the val files).
        cache_dir: Local cache directory for HF downloads.
        image_dir: Directory of already-extracted VRSBench images. When given,
            the multi-GB image archive is not downloaded.
        download_images: Download the split's image archive from the Hub when
            no image_dir is given. Set False to fail fast instead.
        local_annotations: Path to a local JSON file (alternative to the Hub).
        local_image_dir: Image directory used with local_annotations.
        max_samples: Limit the number of samples (for debugging).
    """

    def __init__(
        self,
        data_name: str = HF_REPO_ID,
        split: str = "train",
        cache_dir: str | None = None,
        image_dir: str | None = None,
        download_images: bool = True,
        local_annotations: str | None = None,
        local_image_dir: str | None = None,
        max_samples: int | None = None,
    ) -> None:
        super().__init__()
        self.split = split
        self.image_source: _ImageSource | None = None

        if local_annotations is not None:
            # Load from local JSON + image directory
            self.samples = self._load_local(local_annotations, local_image_dir or ".")
        else:
            # Load from HuggingFace Hub
            self.samples = self._load_from_hub(
                data_name, split, cache_dir, image_dir, download_images,
            )

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        print(f"  VRSBench grounding [{split}]: {len(self.samples)} samples loaded")

    # ── HuggingFace Hub loading ─────────────────────────────────────────

    def _load_from_hub(
        self,
        data_name: str,
        split: str,
        cache_dir: str | None,
        image_dir: str | None,
        download_images: bool,
    ) -> list[dict]:
        """Load grounding samples from the VRSBench files on the Hub."""
        if hf_hub_download is None:
            raise ImportError(
                "The 'huggingface_hub' library is required to download VRSBench. "
                "Install it with: pip install huggingface_hub"
            )
        if split not in SPLIT_FILES:
            raise ValueError(
                f"Unknown split '{split}'. Expected one of {sorted(SPLIT_FILES)}."
            )

        ann_file, archive = SPLIT_FILES[split]

        print(f"  Loading VRSBench annotations from {data_name}/{ann_file}...")
        ann_path = hf_hub_download(
            repo_id=data_name,
            filename=ann_file,
            repo_type="dataset",
            cache_dir=cache_dir,
        )

        images = self._resolve_images(
            data_name, archive, cache_dir, image_dir, download_images,
        )
        self.image_source = images

        with open(ann_path, encoding="utf-8") as f:
            annotations = json.load(f)

        samples: list[dict] = []
        missing_images = 0
        for item in annotations:
            for sample in self._parse_hub_item(item):
                if not images.has(sample["image_name"]):
                    missing_images += 1
                    continue
                samples.append(sample)

        if missing_images:
            print(f"  Skipped {missing_images} references with no matching image file")
        if not samples:
            raise RuntimeError(
                f"No grounding samples parsed from {ann_file}. The annotation "
                f"format may have changed upstream."
            )
        return samples

    def _resolve_images(
        self,
        data_name: str,
        archive: str,
        cache_dir: str | None,
        image_dir: str | None,
        download_images: bool,
    ) -> _ImageSource:
        """Pick the image directory, or download the split's image archive."""
        if image_dir:
            path = Path(image_dir)
            if not path.is_dir():
                raise FileNotFoundError(f"image_dir does not exist: {path}")
            print(f"  Using local images: {path}")
            return _ImageSource(image_dir=str(path))

        if not download_images:
            raise RuntimeError(
                f"VRSBench images are needed but no image_dir was given and "
                f"downloading is disabled. Either pass --image-dir pointing at "
                f"extracted VRSBench images, or drop --no-download-images to "
                f"fetch {archive} (~{ARCHIVE_SIZES_GB.get(archive, 0):.1f} GB) "
                f"from {data_name}."
            )

        size = ARCHIVE_SIZES_GB.get(archive, 0)
        print(
            f"  Downloading {archive} (~{size:.1f} GB) from {data_name} — "
            f"cached after the first run, but that first run takes a while. "
            f"Pass --image-dir to use an existing copy instead."
        )
        zip_path = hf_hub_download(
            repo_id=data_name,
            filename=archive,
            repo_type="dataset",
            cache_dir=cache_dir,
        )
        return _ImageSource(zip_path=zip_path)

    def _parse_hub_item(self, item: dict) -> list[dict]:
        """Parse one annotation entry into zero or more grounding samples.

        Handles both VRSBench layouts: the conversation-style train file and
        the flat eval-referring file.
        """
        if "conversations" in item:
            return self._parse_conversation_item(item)
        return self._parse_referring_item(item)

    def _parse_conversation_item(self, item: dict) -> list[dict]:
        """Parse the `[refer]` turns out of a VRSBench_train.json conversation."""
        image_name = item.get("image") or item.get("img_name") or ""
        if not image_name:
            return []

        turns = item.get("conversations") or []
        samples = []
        for human, gpt in zip(turns[::2], turns[1::2]):
            prompt = str(human.get("value", ""))
            if "[refer]" not in prompt:
                continue

            text = self._parse_expression(prompt)
            boxes = self._parse_boxes(gpt.get("value"))
            if text and boxes:
                samples.append({
                    "text": text,
                    "boxes": boxes,
                    "image_name": image_name,
                })
        return samples

    def _parse_referring_item(self, item: dict) -> list[dict]:
        """Parse an entry of VRSBench_EVAL_referring.json."""
        image_name = (
            item.get("image_id") or item.get("image") or item.get("img_name") or ""
        )
        if not image_name:
            return []

        text = self._parse_expression(
            item.get("question") or item.get("expression") or item.get("caption") or ""
        )
        boxes = self._parse_boxes(item.get("ground_truth") or item.get("bbox"))
        if not text or not boxes:
            return []

        return [{
            "text": text,
            "boxes": boxes,
            "image_name": image_name,
            # Kept for evaluation: VRSBench reports accuracy separately for
            # objects that are / are not the only one of their class in the image
            "unique": item.get("unique"),
            "obj_cls": item.get("obj_cls"),
        }]

    @staticmethod
    def _parse_expression(prompt: str) -> str:
        """Strip VRSBench prompt scaffolding down to the referring expression."""
        if not prompt:
            return ""

        match = _EXPR_RE.search(prompt)
        if match:
            return match.group(1).strip()

        # No <p> tags: drop the <image> token and the "[refer]" instruction
        # prefix, keeping the expression itself.
        text = prompt.replace("<image>", "").strip()
        if "[refer]" in text:
            text = text.split("[refer]", 1)[1]
        return text.strip().strip("?").strip()

    # ── Local JSON loading ──────────────────────────────────────────────

    def _load_local(
        self,
        annotations_path: str,
        image_dir: str,
    ) -> list[dict]:
        """Load grounding samples from a local JSON annotations file.

        Accepts either VRSBench layout, or a flat list of:
        [
            {
                "img_name": "image_001.png",
                "expression": "the bridge over the river",
                "bbox": [x1, y1, x2, y2]   // normalized 0–100
            },
            ...
        ]
        """
        with open(annotations_path, encoding="utf-8") as f:
            annotations = json.load(f)

        images = _ImageSource(image_dir=image_dir)
        self.image_source = images

        samples = []
        for ann in annotations:
            parsed = self._parse_hub_item(ann)
            if not parsed:
                # flat layout with 'expression' / 'bbox'
                text = ann.get("expression", ann.get("caption", ""))
                boxes = self._parse_boxes(ann.get("bbox", ann.get("bboxes", [])))
                img_name = ann.get("img_name", ann.get("image_id", ""))
                if text and boxes and img_name:
                    parsed = [{
                        "text": text.strip(),
                        "boxes": boxes,
                        "image_name": img_name,
                    }]

            samples.extend(s for s in parsed if images.has(s["image_name"]))

        return samples

    # ── Box parsing ─────────────────────────────────────────────────────

    def _parse_boxes(self, raw_boxes) -> list[list[float]] | None:
        """Parse various box formats into a list of [cx, cy, w, h] in 0–1.

        VRSBench uses 0–100 normalization; this converts to 0–1. Input can be
        the `{<x1><y1><x2><y2>}` string VRSBench answers with, a JSON string,
        [x1, y1, x2, y2], or [[x1,y1,x2,y2], ...].
        """
        if raw_boxes is None:
            return None

        if isinstance(raw_boxes, str):
            # VRSBench answer format: one or more "{<x1><y1><x2><y2>}"
            tagged = _BOX_RE.findall(raw_boxes)
            if tagged:
                return self._convert_xyxy_boxes(
                    [[float(v) for v in box] for box in tagged]
                )
            else:
                try:
                    raw_boxes = json.loads(raw_boxes)
                except (json.JSONDecodeError, ValueError):
                    return None

        # Ensure it's a list
        if not isinstance(raw_boxes, (list, tuple)):
            return None

        # Single box [x1, y1, x2, y2] → wrap in list
        if len(raw_boxes) == 4 and all(isinstance(v, (int, float)) for v in raw_boxes):
            raw_boxes = [raw_boxes]

        return self._convert_xyxy_boxes(raw_boxes)

    @staticmethod
    def _convert_xyxy_boxes(raw_boxes) -> list[list[float]] | None:
        """Convert [x1, y1, x2, y2] boxes in 0–100 to [cx, cy, w, h] in 0–1."""
        parsed = []
        for box in raw_boxes:
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue

            x1, y1, x2, y2 = [float(v) for v in box]

            # Rescale from 0–100 to 0–1
            x1 /= 100.0
            y1 /= 100.0
            x2 /= 100.0
            y2 /= 100.0

            # Convert to center format (cx, cy, w, h)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            w = abs(x2 - x1)
            h = abs(y2 - y1)

            # Clamp to [0, 1]
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            w = max(0.0, min(1.0, w))
            h = max(0.0, min(1.0, h))

            if w > 0 and h > 0:
                parsed.append([cx, cy, w, h])

        return parsed if parsed else None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return a single grounding sample.

        Returns:
            dict with keys:
                'image': PIL.Image.Image (RGB)
                'text': str — grounding expression
                'boxes': Tensor (N, 4) — (cx, cy, w, h) normalized 0–1
                'class_labels': Tensor (N,) — all zeros (single-class grounding)
                'image_name': str
        """
        if self.image_source is None:
            raise RuntimeError("Dataset has no image source; construct it via __init__")

        sample = self.samples[idx]
        image = self.image_source.open(sample["image_name"])

        boxes = torch.tensor(sample["boxes"], dtype=torch.float32)  # (N, 4)

        # Class labels: for grounding, all boxes belong to the queried class (label=0)
        class_labels = torch.zeros(len(boxes), dtype=torch.long)

        return {
            "image": image,
            "text": sample["text"],
            "boxes": boxes,
            "class_labels": class_labels,
            "image_name": sample["image_name"],
        }


def collate_fn(batch: list[dict]) -> dict[str, Any]:
    """Custom collate for variable-length box annotations.

    GroundingDINO expects labels as a list of dicts (one per image),
    each with 'class_labels' and 'boxes' tensors.

    Returns:
        dict with:
            'images': list of PIL Images
            'texts': list of str
            'labels': list of dicts with 'class_labels' and 'boxes'
            'image_names': list of str
    """
    images = [item["image"] for item in batch]
    texts = [item["text"] for item in batch]
    labels = [
        {
            "class_labels": item["class_labels"],
            "boxes": item["boxes"],
        }
        for item in batch
    ]
    image_names = [item["image_name"] for item in batch]

    return {
        "images": images,
        "texts": texts,
        "labels": labels,
        "image_names": image_names,
    }


# ── Quick test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("Testing VRSBenchGroundingDataset...")

    ds = VRSBenchGroundingDataset.__new__(VRSBenchGroundingDataset)
    ds.split = "test"
    ds.samples = []

    # Single box [x1, y1, x2, y2] in 0–100
    result = ds._parse_boxes([10, 20, 50, 60])
    assert result is not None
    assert len(result) == 1
    cx, cy, w, h = result[0]
    assert abs(cx - 0.30) < 0.01  # (10+50)/2 / 100
    assert abs(cy - 0.40) < 0.01  # (20+60)/2 / 100
    assert abs(w - 0.40) < 0.01   # (50-10) / 100
    assert abs(h - 0.40) < 0.01   # (60-20) / 100
    print("  Box parsing: single box OK")

    # Multiple boxes
    result = ds._parse_boxes([[0, 0, 50, 50], [50, 50, 100, 100]])
    assert result is not None
    assert len(result) == 2
    print("  Box parsing: multiple boxes OK")

    # JSON string
    result = ds._parse_boxes('[10, 20, 50, 60]')
    assert result is not None
    print("  Box parsing: JSON string OK")

    # VRSBench tagged answer format
    result = ds._parse_boxes("{<25><40><33><60>}")
    assert result is not None and len(result) == 1
    cx, cy, w, h = result[0]
    assert abs(cx - 0.29) < 0.01 and abs(h - 0.20) < 0.01
    print("  Box parsing: VRSBench {<x1><y1><x2><y2>} OK")

    # Invalid
    result = ds._parse_boxes(None)
    assert result is None
    result = ds._parse_boxes([])
    assert result is None
    print("  Box parsing: invalid inputs OK")

    # Expression extraction
    assert ds._parse_expression(
        "<image>\n[refer] give me the location of <p>The toll station</p>"
    ) == "The toll station"
    assert ds._parse_expression(
        "The large yellow vehicle closest to the green area."
    ) == "The large yellow vehicle closest to the green area."
    print("  Expression parsing OK")

    print("\nAll dataset tests passed.")
