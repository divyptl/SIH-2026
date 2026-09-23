"""
Data augmentation and preprocessing for grounding.

Provides training-time augmentations (random horizontal flip, color jitter,
random resize) that consistently update bounding box coordinates, plus
a wrapper around the HF GroundingDinoProcessor for tokenization and
image normalization.
"""

from __future__ import annotations

import random
import re

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
        text: str = "",
    ) -> tuple["Image.Image", torch.Tensor, str]:
        """Augment an image, its boxes and the referring expression.

        Args:
            image: PIL RGB image.
            boxes: (N, 4) tensor in (cx, cy, w, h) format, normalized 0–1.
            text: Referring expression for the boxes.

        Returns:
            (augmented_image, augmented_boxes, augmented_text)
        """
        if not self.augment:
            return image, boxes, text

        # Random horizontal flip (50% chance)
        if random.random() < 0.5:
            image = TF.hflip(image)
            # Flip cx: new_cx = 1.0 - cx
            boxes = boxes.clone()
            boxes[:, 0] = 1.0 - boxes[:, 0]
            # ~63% of VRSBench train expressions say "left"/"right". Without
            # swapping them the flipped box contradicts the text, which stalls
            # box regression (train loss_bbox stayed flat at ~4x val).
            text = swap_left_right(text)

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

        return image, boxes, text


_LEFT_RIGHT_RE = re.compile(r"\b(left|right)", re.IGNORECASE)


def swap_left_right(text: str) -> str:
    """Swap left/right in a referring expression to match a horizontal flip.

    Matches word prefixes, so "leftmost", "right-most" and "top-left" all swap,
    and keeps the capitalisation of the first letter.
    """
    def swap(match: re.Match) -> str:
        word = match.group(1)
        new = "right" if word.lower() == "left" else "left"
        return new.capitalize() if word[0].isupper() else new

    return _LEFT_RIGHT_RE.sub(swap, text)


def uniform(low: float, high: float) -> float:
    """Sample from uniform distribution."""
    return random.uniform(low, high)


def prepare_training_batch(
    batch: dict,
    processor,
    augmentation: GroundingAugmentation | None = None,
    device: str = "cpu",
    image_size: int | None = None,
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
        image_size: Resize images to this square size. Defaults to the
            processor's own setting (shortest edge 800), which upscales
            VRSBench's native 512x512 tiles and inflates activation memory.

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
        aug_texts = []
        for img, lbl, text in zip(images, labels, texts):
            aug_img, aug_boxes, aug_text = augmentation(img, lbl["boxes"], text)
            aug_images.append(aug_img)
            aug_labels.append({
                "class_labels": lbl["class_labels"],
                "boxes": aug_boxes,
            })
            aug_texts.append(aug_text)
        images = aug_images
        labels = aug_labels
        texts = aug_texts

    # Build the GroundingDINO prompt here rather than letting the processor
    # infer it. The processor treats a list of period-free strings as candidate
    # labels for ONE image and merges them into a single prompt ("a cat. a
    # dog."), which collapses the text batch to 1 and blows up in the fusion
    # layer. Terminating every prompt with "." makes that heuristic a no-op, so
    # each image keeps its own text regardless of how the expression was
    # punctuated upstream.
    prompts = []
    for text in texts:
        prompt = text.strip().lower()
        if not prompt.endswith("."):
            prompt += "."
        prompts.append(prompt)

    # Run through HF processor (handles image normalization + text tokenization)
    processor_kwargs = {}
    if image_size is not None:
        processor_kwargs["size"] = {
            "shortest_edge": image_size,
            "longest_edge": image_size,
        }

    inputs = processor(
        images=images,
        text=prompts,
        return_tensors="pt",
        padding=True,
        **processor_kwargs,
    )

    # Move inputs to device (non_blocking enables overlapping transfer with compute if pinned)
    inputs = {
        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }

    # Move labels to device
    labels = [
        {
            "class_labels": lbl["class_labels"].to(device, non_blocking=True),
            "boxes": lbl["boxes"].to(device, non_blocking=True),
        }
        for lbl in labels
    ]

    return inputs, labels
