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

        # True Random Resized Crop (crucial for breaking overfitting)
        i, j, ch, cw = T.RandomResizedCrop.get_params(
            sar, scale=(0.5, 1.0), ratio=(0.75, 1.33)
        )
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


        # Random Erasing (Cutout) - apply to both independently to force cross-modality reliance
        if random.random() > 0.5:
            # Erase 5% to 20% of the image area (increased for stronger regularization)
            i, j, h, w, v = T.RandomErasing.get_params(sar, scale=(0.05, 0.2), ratio=(0.3, 3.3), value=[0.0])
            sar = TF.erase(sar, i, j, h, w, v)
        if random.random() > 0.5:
            i, j, h, w, v = T.RandomErasing.get_params(optical, scale=(0.05, 0.2), ratio=(0.3, 3.3), value=[0.0])
            optical = TF.erase(optical, i, j, h, w, v)

        # --- Photometric augmentations (modality-specific) ---

        # Color jitter (optical only)
        if random.random() > 0.1: # Increased probability from 0.7 to 0.9 (i.e. if rand > 0.1)
            optical = self.color_jitter(optical)

        # Gaussian noise (SAR only, simulates speckle)
        if random.random() > 0.2: # Increased probability from 0.5 to 0.8 (rand > 0.2)
            noise_std = random.uniform(0.02, 0.08) # Increased noise variance
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


# ── Spectral Indices ─────────────────────────────────────────────────────


def compute_ndwi(optical: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute Normalized Difference Water Index from an RGB optical image.

    True NDWI (McFeeters 1996) = (Green - NIR) / (Green + NIR), but
    Sentinel-2 true-colour composites contain only visible RGB bands.

    We approximate NIR as the inverse of visible brightness:
        pseudo_NIR = 1 - (R + G + B) / 3
    This is effective because water absorbs strongly in NIR and appears
    dark overall, while vegetation is bright in NIR and thus has a low
    pseudo_NIR value.

    Args:
        optical: (C, H, W) or (B, C, H, W) tensor with C >= 3, values in [0, 1].
        eps: Small constant to avoid division by zero.

    Returns:
        NDWI map of shape (1, H, W) or (B, 1, H, W), values in [-1, 1].
        High positive values indicate water.
    """
    squeeze = False
    if optical.dim() == 3:
        optical = optical.unsqueeze(0)
        squeeze = True

    red = optical[:, 0:1]
    green = optical[:, 1:2]
    blue = optical[:, 2:3]

    # Pseudo-NIR: high when scene is dark overall (water), low when bright (veg)
    pseudo_nir = 1.0 - (red + green + blue) / 3.0

    ndwi = (green - pseudo_nir) / (green + pseudo_nir + eps)

    if squeeze:
        ndwi = ndwi.squeeze(0)
    return ndwi


def compute_sar_water_mask(
    sar: torch.Tensor,
    threshold: float = 0.15,
) -> torch.Tensor:
    """Estimate a binary water mask from SAR backscatter.

    Smooth open water produces specular reflection away from the sensor,
    resulting in very low backscatter (dark pixels) in SAR imagery.

    Args:
        sar: (1, H, W) or (B, 1, H, W) SAR tensor, values in [0, 1].
        threshold: Backscatter values below this are classified as water.
                   The default (0.15) works well for normalised Sentinel-1.

    Returns:
        Binary mask of same shape: 1.0 = likely water, 0.0 = non-water.
    """
    return (sar < threshold).float()

