"""
VRSBench grounding dataset loader.

Loads the grounding subset of VRSBench from HuggingFace Hub, where each
sample contains a remote-sensing image, a natural-language expression, and
bounding box(es) identifying the referred object(s).

VRSBench box coordinates are normalized to 0–100; this loader rescales
them to 0–1 in (cx, cy, w, h) format as expected by GroundingDINO.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

try:
    from PIL import Image
except ImportError:
    Image = None


class VRSBenchGroundingDataset(Dataset):
    """PyTorch dataset for VRSBench grounding annotations.

    Loads data from HuggingFace Hub or from a local directory.

    Each sample returns:
        image: PIL.Image.Image (RGB)
        text:  str — the grounding expression
        boxes: Tensor (N, 4) — boxes in (cx, cy, w, h) normalized 0–1
        metadata: dict — original annotation info

    Args:
        data_name: HuggingFace dataset name.
        data_subset: HuggingFace dataset config/subset name.
        split: 'train', 'validation', or 'test'.
        cache_dir: Local cache directory for HF downloads.
        local_annotations: Path to local JSON file (alternative to HF).
        local_image_dir: Path to local image directory (used with local_annotations).
        max_samples: Limit the number of samples (for debugging).
    """

    def __init__(
        self,
        data_name: str = "xiang709/VRSBench",
        data_subset: str = "VRSBench",
        split: str = "train",
        cache_dir: str | None = None,
        local_annotations: str | None = None,
        local_image_dir: str | None = None,
        max_samples: int | None = None,
    ) -> None:
        super().__init__()
        self.split = split

        if local_annotations is not None:
            # Load from local JSON + image directory
            self.samples = self._load_local(local_annotations, local_image_dir or ".")
        else:
            # Load from HuggingFace Hub
            self.samples = self._load_from_hub(data_name, data_subset, split, cache_dir)

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        print(f"  VRSBench grounding [{split}]: {len(self.samples)} samples loaded")

    def _load_from_hub(
        self,
        data_name: str,
        data_subset: str,
        split: str,
        cache_dir: str | None,
    ) -> list[dict]:
        """Load grounding samples from HuggingFace Hub."""
        if load_dataset is None:
            raise ImportError(
                "The 'datasets' library is required to load from HuggingFace Hub. "
                "Install it with: pip install datasets"
            )

        print(f"  Loading VRSBench from HuggingFace Hub ({data_name})...")
        ds = load_dataset(data_name, name=data_subset, split=split, cache_dir=cache_dir)

        samples = []
        for item in ds:
            # VRSBench grounding entries have 'bbox' and 'expression' fields
            # Filter to only grounding samples (those with bbox annotations)
            if not self._has_grounding_annotation(item):
                continue

            sample = self._parse_hub_item(item)
            if sample is not None:
                samples.append(sample)

        return samples

    def _has_grounding_annotation(self, item: dict) -> bool:
        """Check if a VRSBench item has grounding annotations."""
        # VRSBench stores grounding data in various possible field names
        for key in ("bbox", "bboxes", "box", "boxes", "grounding_bbox"):
            if key in item and item[key] is not None:
                val = item[key]
                # Check it's not empty
                if isinstance(val, (list, tuple)) and len(val) > 0:
                    return True
                if isinstance(val, str) and val.strip():
                    return True
        return False

    def _parse_hub_item(self, item: dict) -> dict | None:
        """Parse a single HuggingFace dataset item into our format."""
        # Extract image
        image = item.get("image")
        if image is None:
            return None

        # Extract text expression
        text = (
            item.get("expression")
            or item.get("caption")
            or item.get("text")
            or item.get("query")
            or ""
        )
        if not text:
            return None

        # Extract bounding boxes (VRSBench uses 0–100 normalized coords)
        raw_boxes = (
            item.get("bbox")
            or item.get("bboxes")
            or item.get("box")
            or item.get("boxes")
            or item.get("grounding_bbox")
        )
        boxes = self._parse_boxes(raw_boxes)
        if boxes is None or len(boxes) == 0:
            return None

        return {
            "image": image,
            "text": text.strip(),
            "boxes": boxes,
            "image_name": item.get("img_name", item.get("image_id", "")),
        }

    def _load_local(
        self,
        annotations_path: str,
        image_dir: str,
    ) -> list[dict]:
        """Load grounding samples from a local JSON annotations file.

        Expected JSON format (list of dicts):
        [
            {
                "img_name": "image_001.png",
                "expression": "the bridge over the river",
                "bbox": [x1, y1, x2, y2]   // normalized 0–100
            },
            ...
        ]
        """
        if Image is None:
            raise ImportError("Pillow is required: pip install Pillow")

        ann_path = Path(annotations_path)
        img_dir = Path(image_dir)

        with open(ann_path) as f:
            annotations = json.load(f)

        samples = []
        for ann in annotations:
            img_name = ann.get("img_name", ann.get("image_id", ""))
            img_path = img_dir / img_name

            if not img_path.exists():
                continue

            text = ann.get("expression", ann.get("caption", ""))
            raw_boxes = ann.get("bbox", ann.get("bboxes", []))
            boxes = self._parse_boxes(raw_boxes)

            if text and boxes is not None and len(boxes) > 0:
                samples.append({
                    "image": str(img_path),
                    "text": text.strip(),
                    "boxes": boxes,
                    "image_name": img_name,
                })

        return samples

    def _parse_boxes(self, raw_boxes) -> list[list[float]] | None:
        """Parse various box formats into a list of [cx, cy, w, h] in 0–1.

        VRSBench uses 0–100 normalization; this converts to 0–1.
        Input can be [x1, y1, x2, y2] or [[x1,y1,x2,y2], ...] or a JSON string.
        """
        if raw_boxes is None:
            return None

        # Handle JSON string
        if isinstance(raw_boxes, str):
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

        # Convert each box from [x1, y1, x2, y2] (0–100) to [cx, cy, w, h] (0–1)
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
        sample = self.samples[idx]

        # Load / convert image
        image = sample["image"]
        if isinstance(image, str):
            if Image is None:
                raise ImportError("Pillow is required: pip install Pillow")
            image = Image.open(image).convert("RGB")
        elif hasattr(image, "convert"):
            image = image.convert("RGB")

        boxes = torch.tensor(sample["boxes"], dtype=torch.float32)  # (N, 4)

        # Class labels: for grounding, all boxes belong to the queried class (label=0)
        class_labels = torch.zeros(len(boxes), dtype=torch.long)

        return {
            "image": image,
            "text": sample["text"],
            "boxes": boxes,
            "class_labels": class_labels,
            "image_name": sample.get("image_name", ""),
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
    print("Testing VRSBenchGroundingDataset...")

    # Test box parsing
    ds = VRSBenchGroundingDataset.__new__(VRSBenchGroundingDataset)
    ds.split = "test"
    ds.samples = []

    # Test _parse_boxes with various formats
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

    # Invalid
    result = ds._parse_boxes(None)
    assert result is None
    result = ds._parse_boxes([])
    assert result is None
    print("  Box parsing: invalid inputs OK")

    print("\nAll dataset tests passed.")
