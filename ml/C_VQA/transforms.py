"""
Transforms and data augmentation for bi-temporal Change-VQA.

Applies synchronized spatial transformations (horizontal flip, vertical flip,
rotation) to pairs of images (T1 and T2) and their corresponding change mask,
ensuring pixel-level alignment across time steps.
"""

from __future__ import annotations

import random
from typing import Any

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import torch
    import torchvision.transforms.functional as TF
except ImportError:
    torch = None
    TF = None


# Standard ImageNet normalization statistics for optical satellite imagery
OPTICAL_MEAN = [0.485, 0.456, 0.406]
OPTICAL_STD = [0.229, 0.224, 0.225]


def normalize_image(tensor: Any) -> Any:
    """Normalize a (C, H, W) float tensor with ImageNet statistics."""
    if TF is not None and hasattr(tensor, "shape") and tensor.shape[0] == 3:
        return TF.normalize(tensor, mean=OPTICAL_MEAN, std=OPTICAL_STD)
    return tensor


def denormalize_image(tensor: Any) -> Any:
    """Reverse ImageNet normalization for visualization."""
    if torch is None or not hasattr(tensor, "clone"):
        return tensor
    inv_mean = [-m / s for m, s in zip(OPTICAL_MEAN, OPTICAL_STD)]
    inv_std = [1.0 / s for s in OPTICAL_STD]
    return TF.normalize(tensor.clone(), mean=inv_mean, std=inv_std)


class PairedBitemporalTransform:
    """Synchronized transforms for bi-temporal image pairs (T1, T2) and change mask.

    Spatial transformations (flips, rotations) are applied identically to T1, T2,
    and mask to preserve bi-temporal spatial correspondence. Photometric transforms
    (brightness, contrast) are applied independently or with small perturbations
    to simulate different acquisition conditions.
    """

    def __init__(
        self,
        image_size: int = 256,
        augment: bool = True,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
        rotation_prob: float = 0.5,
    ) -> None:
        self.image_size = image_size
        self.augment = augment
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.rotation_prob = rotation_prob

    def __call__(
        self,
        img_t1: Any,
        img_t2: Any,
        mask: Any | None = None,
    ) -> tuple[Any, Any, Any | None]:
        """Transform a (T1, T2, mask) triplet.

        Args:
            img_t1: PIL Image or Tensor for time 1.
            img_t2: PIL Image or Tensor for time 2.
            mask: Optional PIL Image or Tensor binary change mask (0 or 255/1).

        Returns:
            Tuple of (t1_tensor, t2_tensor, mask_tensor).
        """
        if TF is None or torch is None:
            return img_t1, img_t2, mask

        # Ensure PIL if not already tensor
        if not isinstance(img_t1, torch.Tensor) and Image is not None and not isinstance(img_t1, Image.Image):
            img_t1 = Image.fromarray(img_t1)
        if not isinstance(img_t2, torch.Tensor) and Image is not None and not isinstance(img_t2, Image.Image):
            img_t2 = Image.fromarray(img_t2)
        if mask is not None and not isinstance(mask, torch.Tensor) and Image is not None and not isinstance(mask, Image.Image):
            mask = Image.fromarray(mask)

        # Convert to RGB PIL if mode differs
        if Image is not None and isinstance(img_t1, Image.Image) and img_t1.mode != "RGB":
            img_t1 = img_t1.convert("RGB")
        if Image is not None and isinstance(img_t2, Image.Image) and img_t2.mode != "RGB":
            img_t2 = img_t2.convert("RGB")
        if Image is not None and isinstance(mask, Image.Image) and mask.mode != "L":
            mask = mask.convert("L")

        # Resize
        target_size = [self.image_size, self.image_size]
        img_t1 = TF.resize(img_t1, target_size)
        img_t2 = TF.resize(img_t2, target_size)
        if mask is not None:
            mask = TF.resize(mask, target_size, interpolation=TF.InterpolationMode.NEAREST)

        # Synchronized Data Augmentation (train mode)
        if self.augment:
            # Horizontal flip
            if random.random() < self.hflip_prob:
                img_t1 = TF.hflip(img_t1)
                img_t2 = TF.hflip(img_t2)
                if mask is not None:
                    mask = TF.hflip(mask)

            # Vertical flip
            if random.random() < self.vflip_prob:
                img_t1 = TF.vflip(img_t1)
                img_t2 = TF.vflip(img_t2)
                if mask is not None:
                    mask = TF.vflip(mask)

            # Random 90-degree rotations (preserves grid structure)
            if random.random() < self.rotation_prob:
                angle = random.choice([90, 180, 270])
                img_t1 = TF.rotate(img_t1, angle)
                img_t2 = TF.rotate(img_t2, angle)
                if mask is not None:
                    mask = TF.rotate(mask, angle)

        # Convert to tensor
        if not isinstance(img_t1, torch.Tensor):
            t1_tensor = TF.to_tensor(img_t1)
        else:
            t1_tensor = img_t1

        if not isinstance(img_t2, torch.Tensor):
            t2_tensor = TF.to_tensor(img_t2)
        else:
            t2_tensor = img_t2

        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                mask_tensor = TF.to_tensor(mask)
            else:
                mask_tensor = mask
            # Binarize mask to 0.0 or 1.0
            mask_tensor = (mask_tensor > 0.5).float()
            if mask_tensor.dim() == 2:
                mask_tensor = mask_tensor.unsqueeze(0)
        else:
            mask_tensor = None

        # Normalize optical imagery
        t1_tensor = normalize_image(t1_tensor)
        t2_tensor = normalize_image(t2_tensor)

        return t1_tensor, t2_tensor, mask_tensor
