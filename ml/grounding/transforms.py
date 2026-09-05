"""
Data augmentation and preprocessing for grounding.

Provides training-time augmentations (random horizontal flip, color jitter,
random resize) that consistently update bounding box coordinates, plus
a wrapper around the HF GroundingDinoProcessor for tokenization and
image normalization.
"""

from __future__ import annotations

import random

import torch
import torchvision.transforms.functional as TF

try:
    from PIL import Image, ImageFilter
except ImportError:
    Image = None
    ImageFilter = None


class GroundingAugmentation:
    """Apply augmentations to a (image, boxes) pair for grounding training.

    Geometric transforms update box coordinates accordingly.
    Photometric transforms only modify pixel values.

    Args:
        augment: Whether to apply random augmentations.
        min_scale: Minimum random resize scale factor.
        max_scale: Maximum random resize scale factor.
    """

    def __init__(
        self,
        augment: bool = True,
        min_scale: float = 0.8,
        max_scale: float = 1.2,
    ) -> None:
        self.augment = augment
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(
        self,
        image: "Image.Image",
        boxes: torch.Tensor,
    ) -> tuple["Image.Image", torch.Tensor]:
        """Augment an image and its boxes.

        Args:
            image: PIL RGB image.
            boxes: (N, 4) tensor in (cx, cy, w, h) format, normalized 0–1.

        Returns:
            (augmented_image, augmented_boxes)
        """
        if not self.augment:
            return image, boxes

        # Random horizontal flip (50% chance)
        if random.random() < 0.5:
            image = TF.hflip(image)
            # Flip cx: new_cx = 1.0 - cx
            boxes = boxes.clone()
            boxes[:, 0] = 1.0 - boxes[:, 0]

        # Random color jitter (photometric — no box update needed)
        if random.random() < 0.5:
            image = TF.adjust_brightness(image, uniform(0.8, 1.2))
        if random.random() < 0.3:
            image = TF.adjust_contrast(image, uniform(0.8, 1.2))
        if random.random() < 0.3:
            image = TF.adjust_saturation(image, uniform(0.8, 1.2))

        # Random Gaussian blur (slight, for robustness)
        if random.random() < 0.1 and ImageFilter is not None:
            image = image.filter(ImageFilter.GaussianBlur(radius=1))

        return image, boxes


def uniform(low: float, high: float) -> float:
    """Sample from uniform distribution."""
    return random.uniform(low, high)


def prepare_training_batch(
    batch: dict,
    processor,
    augmentation: GroundingAugmentation | None = None,
    device: str = "cpu",
) -> tuple[dict, list[dict]]:
    """Prepare a collated batch for GroundingDINO training.

    Takes raw collated output from the dataset and runs it through:
    1. Optional augmentation (per image)
    2. HF processor for tokenization + pixel normalization

    Args:
        batch: Output from dataset.collate_fn with 'images', 'texts', 'labels'.
        processor: GroundingDinoProcessor from HF.
        augmentation: Optional augmentation to apply per sample.
        device: Target device for tensors.

    Returns:
        (inputs, labels) where:
            inputs: dict with 'pixel_values', 'input_ids', 'attention_mask', etc.
            labels: list of dicts with 'class_labels' and 'boxes' tensors.
    """
    images = batch["images"]
    texts = batch["texts"]
    labels = batch["labels"]

    # Apply augmentation per sample
    if augmentation is not None:
        aug_images = []
        aug_labels = []
        for img, lbl in zip(images, labels):
            aug_img, aug_boxes = augmentation(img, lbl["boxes"])
            aug_images.append(aug_img)
            aug_labels.append({
                "class_labels": lbl["class_labels"],
                "boxes": aug_boxes,
            })
        images = aug_images
        labels = aug_labels

    # Run through HF processor (handles image normalization + text tokenization)
    inputs = processor(
        images=images,
        text=texts,
        return_tensors="pt",
        padding=True,
    )

    # Move inputs to device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    # Move labels to device
    labels = [
        {
            "class_labels": lbl["class_labels"].to(device),
            "boxes": lbl["boxes"].to(device),
        }
        for lbl in labels
    ]

    return inputs, labels
