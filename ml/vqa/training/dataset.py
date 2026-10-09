"""
PyTorch Dataset for single-image VQA.

Reads the JSONL manifests written by ml.vqa.training.prepare_vrsbench.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from ml.vqa.config import TrainConfig
from ml.vqa.model import SimpleTokenizer
from ml.vqa.preprocessing import load_image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Answer label for a question whose answer is not in the vocabulary. Such
# questions are dropped from training and scored as wrong in evaluation.
UNKNOWN_ANSWER = -1


def eval_transform(image_size: int) -> transforms.Compose:
    """The preprocessing used at evaluation and inference time."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class VQADataset(Dataset):
    """
    Single-image VQA samples from `{data_root}/{split}.jsonl`, one per line:
        {"image": "<path relative to data_root>", "question": "...", "answer": "...",
         "type": "<question type or null>", "modality": "optical"}

    Args:
        data_root: Directory holding the manifests.
        split: Manifest name ('train', 'dev', 'test').
        answers: Answer vocabulary; index = class label.
        tokenizer: Question tokenizer (the model's).
        config: Image size, augmentation, question length.
        drop_unknown_answers: Skip questions whose answer is not in `answers`
            (for training; evaluation keeps them and counts them as wrong).
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        answers: list[str],
        tokenizer: SimpleTokenizer,
        config: TrainConfig | None = None,
        drop_unknown_answers: bool = False,
        max_question_length: int = 32,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.split = split
        self.config = config or TrainConfig()
        self.tokenizer = tokenizer
        self.max_question_length = max_question_length
        self.answer_to_idx = {a: i for i, a in enumerate(answers)}

        manifest = self.data_root / f"{split}.jsonl"
        if not manifest.exists():
            raise FileNotFoundError(
                f"Manifest not found: {manifest}. Run `python -m ml.vqa.training.prepare_vrsbench`."
            )
        with manifest.open("r", encoding="utf-8") as f:
            samples = [json.loads(line) for line in f if line.strip()]
        if drop_unknown_answers:
            samples = [s for s in samples if s["answer"] in self.answer_to_idx]
        self.samples = samples

        # No flips: they would turn "left" answers into wrong labels.
        if split == "train" and self.config.augment:
            self.transform = transforms.Compose([
                transforms.Resize((self.config.image_size, self.config.image_size)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        else:
            self.transform = eval_transform(self.config.image_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.samples[idx]
        image = load_image(self.data_root / record["image"], modality=record.get("modality", "optical"))
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")

        q_ids, _ = self.tokenizer.encode(record["question"], max_length=self.max_question_length)
        return {
            "image": self.transform(image),
            "question_ids": torch.tensor(q_ids, dtype=torch.long),
            "answer_label": torch.tensor(
                self.answer_to_idx.get(record["answer"], UNKNOWN_ANSWER), dtype=torch.long,
            ),
            "index": idx,
        }
