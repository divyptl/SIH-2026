"""
PyTorch Dataset for VQA.

Loads image-question-answer triples from a flattened JSONL manifest.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from ml.vqa.config import TrainConfig
from ml.vqa.model import SimpleTokenizer, CANONICAL_ANSWERS
from ml.vqa.preprocessing import load_image


class VQADataset(Dataset):
    """
    Dataset for single-image VQA.

    Expects a JSONL manifest at `{data_root}/{split}.jsonl` where each line is:
    {"image": "path/to/img.jpg", "question": "What is this?", "answer": "road", "modality": "optical"}
    """

    def __init__(self, data_root: str | Path, split: str, config: TrainConfig | None = None) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.split = split
        self.config = config or TrainConfig()

        manifest = self.data_root / f"{split}.jsonl"
        if not manifest.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest}")

        self.samples = []
        with manifest.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.samples.append(json.loads(line))

        self.tokenizer = SimpleTokenizer()

        # Build answer vocabulary mapping
        self.answer_to_idx = {ans: i for i, ans in enumerate(CANONICAL_ANSWERS)}

        # Image transforms
        if self.split == "train" and self.config.augment:
            self.transform = transforms.Compose([
                transforms.Resize((self.config.image_size, self.config.image_size)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((self.config.image_size, self.config.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self) -> int:
        return len(self.samples)

    def _get_answer_label(self, answer: str) -> int:
        """Map a string answer to a class index."""
        clean_ans = answer.strip().lower()
        if clean_ans in self.answer_to_idx:
            return self.answer_to_idx[clean_ans]
        return 0  # Default to first class if unknown/unmapped (could also map to a specific "unknown" class)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.samples[idx]
        image_path = self.data_root / record["image"]
        
        # Kaggle workaround: If Images.zip extracts into Images/Images/..., gracefully fall back
        if not image_path.exists():
            fallback_path = self.data_root / "Images" / record["image"]
            if fallback_path.exists():
                image_path = fallback_path
        
        # Load image (load_image returns PIL Image)
        img = load_image(image_path, modality=record.get("modality", "optical"))
        if not isinstance(img, Image.Image):
            # In case preprocessing returns a numpy array for SAR/multispectral
            if hasattr(img, "astype"):
                import numpy as np
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8)
                img = Image.fromarray(img)
            else:
                img = img.convert("RGB")

        # Apply transforms
        img_tensor = self.transform(img)

        # Tokenize question
        question = record["question"]
        q_ids, _ = self.tokenizer.encode(question, max_length=32)
        q_tensor = torch.tensor(q_ids, dtype=torch.long)

        # Answer label
        ans_label = self._get_answer_label(record["answer"])
        ans_tensor = torch.tensor(ans_label, dtype=torch.long)

        return {
            "image": img_tensor,
            "question_ids": q_tensor,
            "answer_label": ans_tensor,
            "question_text": question,
            "answer_text": record["answer"],
        }
