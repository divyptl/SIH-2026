"""
PyTorch Dataset and DataLoaders for the Real-Life CDVQA Benchmark.

Operates strictly on real-world remote sensing datasets:
    1. Standard CDVQA benchmark structure with real paired optical images (T1, T2),
       ground truth change masks, and question-answer annotations.
    2. Multi-temporal synchronized data augmentation and radiometric normalization.
    3. Pure real-data pipeline: does not synthesize fake sample data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from PIL import Image

import torch
from torch.utils.data import Dataset

from ml.C_VQA.model import CANONICAL_ANSWERS, SimpleTokenizer
from ml.C_VQA.transforms import PairedBitemporalTransform


def load_real_satellite_image(path: Path | str) -> Image.Image | np.ndarray:
    """Load a real satellite image supporting GeoTIFF, TIFF, PNG, and JPEG."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Real satellite image not found at: {path}")

    try:
        with Image.open(path) as img:
            if img.mode in ("I;16", "I", "F"):
                arr = np.array(img, dtype=np.float32)
                arr_min, arr_max = arr.min(), arr.max()
                if arr_max > arr_min:
                    norm = ((arr - arr_min) / (arr_max - arr_min) * 255.0).astype(np.uint8)
                else:
                    norm = np.zeros_like(arr, dtype=np.uint8)
                return Image.fromarray(norm).convert("RGB")
            elif img.mode != "RGB":
                return img.convert("RGB")
            else:
                return img.copy()
    except Exception:
        pass

    raise IOError(f"Unable to read satellite image at: {path}")


def load_real_satellite_mask(path: Path | str) -> Image.Image:
    """Load a real binary change mask."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Real change mask not found at: {path}")

    with Image.open(path) as m:
        mask = m.convert("L")
        arr = np.array(mask)
        if arr.max() == 1:
            arr = (arr * 255).astype(np.uint8)
            mask = Image.fromarray(arr)
        return mask


# ── Canonical Question Templates for CDVQA ───────────────────────────────

CDVQA_QUESTION_TEMPLATES = [
    ("Has the built-up area increased, decreased, or remained unchanged?", "urban"),
    ("Did any new structures appear between the two dates?", "urban"),
    ("What major environmental change occurred between T1 and T2?", "general"),
    ("Has the vegetation cover increased or decreased?", "vegetation"),
    ("Has the water body expanded, receded, or remained the same?", "water"),
    ("Where did the primary land-cover change take place?", "spatial"),
    ("Are there signs of recent construction activity?", "urban"),
    ("Did deforestation or vegetation loss happen in this region?", "vegetation"),
    ("Is there any flooding or water accumulation visible?", "water"),
]


class CDVQADataset(Dataset):
    """PyTorch Dataset for Real-Life Remote Sensing Change Detection VQA.

    Operates strictly on real satellite image pairs, masks, and question annotations.

    Args:
        root: Root directory of CDVQA dataset containing images_t1, images_t2, masks, and annotations.
        split: 'train', 'val', or 'test'.
        transform: PairedBitemporalTransform instance.
        max_question_length: Max token length for query questions.
    """

    def __init__(
        self,
        root: str | Path = "data/cdvqa",
        split: str = "train",
        transform: Callable | None = None,
        max_question_length: int = 32,
        auto_generate: bool = False,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.max_question_length = max_question_length
        self.tokenizer = SimpleTokenizer()

        if transform is None:
            self.transform = PairedBitemporalTransform(
                image_size=256,
                augment=(split == "train"),
            )
        else:
            self.transform = transform

        # Build vocabulary mapping for answers
        self.answer_to_idx = {ans.lower(): i for i, ans in enumerate(CANONICAL_ANSWERS)}
        self.idx_to_answer = {i: ans for i, ans in enumerate(CANONICAL_ANSWERS)}

        # Load samples strictly from real dataset
        self.samples: list[dict[str, Any]] = []
        self._load_dataset()

    def _load_dataset(self) -> None:
        """Load real annotations JSON or fail with descriptive error if missing."""
        annotation_file = self.root / f"{self.split}_annotations.json"
        sub_annotation_file = self.root / self.split / "annotations.json"
        general_annotation_file = self.root / "annotations.json"

        target_file = None
        for cand in (annotation_file, sub_annotation_file, general_annotation_file):
            if cand.exists():
                target_file = cand
                break

        if target_file is None:
            raise FileNotFoundError(
                f"[CDVQADataset] No real dataset annotations found for split '{self.split}' at root '{self.root}'.\n"
                f"Expected one of: {annotation_file} or {general_annotation_file}.\n"
                f"Please place your real CDVQA dataset in '{self.root}' or specify --data-root pointing to the real dataset."
            )

        with open(target_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and "samples" in data:
                raw_samples = data["samples"]
            elif isinstance(data, dict) and "annotations" in data:
                raw_samples = data["annotations"]
            elif isinstance(data, list):
                raw_samples = data
            else:
                raw_samples = []

            # Filter split if annotated and unpack multi-question items
            for s in raw_samples:
                if "split" in s and s["split"] != self.split:
                    continue

                # Support multi-question real benchmark format
                if "questions" in s and isinstance(s["questions"], list):
                    for q_info in s["questions"]:
                        self.samples.append({
                            "id": f"{s.get('id', 'pair')}_{q_info.get('id', len(self.samples))}",
                            "image_t1": s.get("image_t1") or s.get("img_t1") or s.get("t1"),
                            "image_t2": s.get("image_t2") or s.get("img_t2") or s.get("t2"),
                            "mask": s.get("mask") or s.get("change_mask"),
                            "question": q_info.get("question", "What changed between these two dates?"),
                            "answer": q_info.get("answer", "unchanged"),
                            "change_type": q_info.get("change_type", s.get("change_type", "general")),
                        })
                else:
                    self.samples.append(s)

        if len(self.samples) == 0:
            raise FileNotFoundError(
                f"[CDVQADataset] Found annotation file '{target_file}' but 0 samples for split '{self.split}'. "
                f"Please verify your real dataset annotations."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_path(self, rel_or_abs: str | Path) -> Path:
        """Resolve a file path relative to dataset root or split dir."""
        p = Path(rel_or_abs)
        if p.is_absolute() and p.exists():
            return p

        cand1 = self.root / p
        if cand1.exists():
            return cand1

        cand2 = self.root / self.split / p
        if cand2.exists():
            return cand2

        return cand1

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return a single real Change-VQA sample."""
        sample_info = self.samples[index]

        # 1. Load real satellite images T1 and T2
        path_t1 = self._resolve_path(sample_info["image_t1"])
        path_t2 = self._resolve_path(sample_info["image_t2"])

        img_t1 = load_real_satellite_image(path_t1)
        img_t2 = load_real_satellite_image(path_t2)

        # 2. Load real change mask if available
        mask = None
        if "mask" in sample_info and sample_info["mask"]:
            path_mask = self._resolve_path(sample_info["mask"])
            if path_mask.exists():
                mask = load_real_satellite_mask(path_mask)

        # 3. Apply paired transforms
        t1_tensor, t2_tensor, mask_tensor = self.transform(img_t1, img_t2, mask)

        # 4. Tokenize question
        question = sample_info.get("question", "What changed between these two dates?")
        q_ids, q_mask = self.tokenizer.encode(question, max_length=self.max_question_length)

        # 5. Encode answer target
        raw_answer = str(sample_info.get("answer", "unchanged")).lower().strip()
        ans_idx = self.answer_to_idx.get(raw_answer, 0)

        item: dict[str, Any] = {
            "t1": t1_tensor,
            "t2": t2_tensor,
            "question_ids": torch.tensor(q_ids, dtype=torch.long) if torch else q_ids,
            "question_mask": torch.tensor(q_mask, dtype=torch.bool) if torch else q_mask,
            "answer_idx": torch.tensor(ans_idx, dtype=torch.long) if torch else ans_idx,
            "raw_question": question,
            "raw_answer": raw_answer,
            "sample_id": sample_info.get("id", f"sample_{index:04d}"),
            "change_type": sample_info.get("change_type", "general"),
        }

        if mask_tensor is not None:
            item["mask"] = mask_tensor

        return item


def cdvqa_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Custom collate function for DataLoader batches."""
    t1_batch = torch.stack([item["t1"] for item in batch])
    t2_batch = torch.stack([item["t2"] for item in batch])
    q_ids_batch = torch.stack([item["question_ids"] for item in batch])
    q_mask_batch = torch.stack([item["question_mask"] for item in batch])
    ans_batch = torch.stack([item["answer_idx"] for item in batch])

    mask_batch = torch.stack([item["mask"] for item in batch]) if "mask" in batch[0] else None

    collated: dict[str, Any] = {
        "t1": t1_batch,
        "t2": t2_batch,
        "question_ids": q_ids_batch,
        "question_mask": q_mask_batch,
        "answer_targets": ans_batch,
        "raw_questions": [item["raw_question"] for item in batch],
        "raw_answers": [item["raw_answer"] for item in batch],
        "sample_ids": [item["sample_id"] for item in batch],
        "mask_targets": mask_batch,
    }

    return collated
