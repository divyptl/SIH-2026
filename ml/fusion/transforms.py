"""
Data augmentation transforms for SEN1-2 SAR-optical pairs.

Augmentations are designed to be consistent across the SAR-optical pair
(same random crop, flip, rotation) while allowing modality-specific
transforms (e.g., color jitter only for optical).
"""

from __future__ import annotations

import random

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class PairedTransform:
    """Apply geometrically consistent transforms to SAR-optical pairs.

    Geometric transforms (crop, flip, rotate) use the same random params
    for both modalities. Photometric transforms (jitter, blur) are applied
    only to the optical image.

    Args:
        size: Target image size after resize/crop.
        augment: Whether to apply random augmentations (False = resize only).
    """

    def __init__(self, size: int = 224, augment: bool = True) -> None:
        self.size = size
        self.augment = augment

        # Optical-only photometric augmentation
        self.color_jitter = T.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.05,
        )

    def __call__(
        self,
        sar: torch.Tensor,
        optical: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform a SAR-optical pair.

        Args:
            sar: (1, H, W) SAR tensor.
            optical: (3, H, W) optical tensor.

        Returns:
            Transformed (sar, optical) tensors.
        """
        if not self.augment:
            # Validation/test: just resize
            sar = TF.resize(sar, [self.size, self.size], antialias=True)
            optical = TF.resize(optical, [self.size, self.size], antialias=True)
            return sar, optical

        # --- Geometric augmentations (same for both) ---

        # Random resize crop
        h, w = sar.shape[-2:]
        crop_size = min(h, w)
        i, j, ch, cw = T.RandomCrop.get_params(sar, (crop_size, crop_size))
        sar = TF.crop(sar, i, j, ch, cw)
        optical = TF.crop(optical, i, j, ch, cw)

        # Resize to target
        sar = TF.resize(sar, [self.size, self.size], antialias=True)
        optical = TF.resize(optical, [self.size, self.size], antialias=True)

        # Random horizontal flip
        if random.random() > 0.5:
            sar = TF.hflip(sar)
            optical = TF.hflip(optical)

        # Random vertical flip
        if random.random() > 0.5:
            sar = TF.vflip(sar)
            optical = TF.vflip(optical)

        # Random 90-degree rotation
        k = random.choice([0, 1, 2, 3])
        if k > 0:
            sar = torch.rot90(sar, k, dims=[-2, -1])
            optical = torch.rot90(optical, k, dims=[-2, -1])

        # --- Photometric augmentations (modality-specific) ---

        # Color jitter (optical only)
        if random.random() > 0.3:
            optical = self.color_jitter(optical)

        # Gaussian noise (SAR only, simulates speckle)
        if random.random() > 0.5:
            noise_std = random.uniform(0.01, 0.05)
            sar = sar + torch.randn_like(sar) * noise_std
            sar = sar.clamp(0.0, 1.0)

        return sar, optical


# Normalization constants (ImageNet for optical, empirical for SAR)
SAR_MEAN = [0.3]
SAR_STD = [0.2]
OPTICAL_MEAN = [0.485, 0.456, 0.406]
OPTICAL_STD = [0.229, 0.224, 0.225]


def normalize_sar(x: torch.Tensor) -> torch.Tensor:
    """Normalize SAR tensor with empirical mean/std."""
    mean = torch.tensor(SAR_MEAN, device=x.device).view(-1, 1, 1)
    std = torch.tensor(SAR_STD, device=x.device).view(-1, 1, 1)
    return (x - mean) / std


def normalize_optical(x: torch.Tensor) -> torch.Tensor:
    """Normalize optical tensor with ImageNet mean/std."""
    mean = torch.tensor(OPTICAL_MEAN, device=x.device).view(-1, 1, 1)
    std = torch.tensor(OPTICAL_STD, device=x.device).view(-1, 1, 1)
    return (x - mean) / std
