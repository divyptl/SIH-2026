"""
Physics-informed data augmentation and spectral/radar transforms for SAR-optical pairs.
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from fusion.sar_physics import calibrate_to_sigma0_db, refined_lee_filter

SAR_MEAN = [0.3]
SAR_STD = [0.2]
OPTICAL_MEAN = [0.485, 0.456, 0.406]
OPTICAL_STD = [0.229, 0.224, 0.225]


class PairedTransform:
    """Apply geometrically consistent transforms and physical SAR speckle to pairs."""

    def __init__(self, size: int = 224, augment: bool = True, apply_lee: bool = False) -> None:
        self.size = size
        self.augment = augment
        self.apply_lee = apply_lee
        self.color_jitter = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)

    def __call__(
        self,
        sar: torch.Tensor,
        optical: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.apply_lee:
            sar_np = sar.squeeze(0).cpu().numpy()
            sar_filt = refined_lee_filter(sar_np, window_size=5, enl=4.4)
            sar = torch.from_numpy(sar_filt).unsqueeze(0).to(sar.device)

        if not self.augment:
            sar = TF.resize(sar, [self.size, self.size], antialias=True)
            optical = TF.resize(optical, [self.size, self.size], antialias=True)
            return sar, optical

        if sar.shape[-2] >= self.size and sar.shape[-1] >= self.size:
            i, j, ch, cw = T.RandomCrop.get_params(sar, (self.size, self.size))
            sar = TF.crop(sar, i, j, ch, cw)
            optical = TF.crop(optical, i, j, ch, cw)
        else:
            sar = TF.resize(sar, [self.size, self.size], antialias=True)
            optical = TF.resize(optical, [self.size, self.size], antialias=True)

        if random.random() > 0.5:
            sar, optical = TF.hflip(sar), TF.hflip(optical)
        if random.random() > 0.5:
            sar, optical = TF.vflip(sar), TF.vflip(optical)

        if (k := random.choice([0, 1, 2, 3])) > 0:
            sar = torch.rot90(sar, k, dims=[-2, -1])
            optical = torch.rot90(optical, k, dims=[-2, -1])

        if random.random() > 0.5:
            i, j, h, w, v = T.RandomErasing.get_params(sar, scale=(0.05, 0.2), ratio=(0.3, 3.3), value=[0.0])
            sar = TF.erase(sar, i, j, h, w, v)
        if random.random() > 0.5:
            i, j, h, w, v = T.RandomErasing.get_params(optical, scale=(0.05, 0.2), ratio=(0.3, 3.3), value=[0.0])
            optical = TF.erase(optical, i, j, h, w, v)

        if random.random() > 0.1:
            optical = self.color_jitter(optical)

        # True Multiplicative Gamma Speckle Simulation (replaces non-physical additive Gaussian noise)
        if random.random() > 0.2:
            looks = random.uniform(3.0, 8.0)
            gamma_dist = torch.distributions.Gamma(
                concentration=torch.tensor(looks, device=sar.device),
                rate=torch.tensor(looks, device=sar.device),
            )
            speckle = gamma_dist.sample(sar.shape)
            sar = (sar * speckle).clamp(0.0, 1.0)

        return sar, optical


def normalize_sar(x: torch.Tensor) -> torch.Tensor:
    """Normalize 1-channel or 3-channel physics SAR tensor."""
    c = x.shape[-3] if x.dim() >= 3 else 1
    if c == 3:
        mean = torch.tensor([0.45, 0.35, 0.40], device=x.device).view(-1, 1, 1)
        std = torch.tensor([0.22, 0.22, 0.22], device=x.device).view(-1, 1, 1)
    else:
        mean = torch.tensor(SAR_MEAN, device=x.device).view(-1, 1, 1)
        std = torch.tensor(SAR_STD, device=x.device).view(-1, 1, 1)
    return (x - mean) / std


def normalize_optical(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(OPTICAL_MEAN, device=x.device).view(-1, 1, 1)
    std = torch.tensor(OPTICAL_STD, device=x.device).view(-1, 1, 1)
    return (x - mean) / std


def compute_ndwi(optical: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute NDWI from unnormalized [0, 1] optical tensor (supports 4-band NIR or 3-band RGB proxy)."""
    squeeze = optical.dim() == 3
    if squeeze:
        optical = optical.unsqueeze(0)

    green = optical[:, 1:2]
    if optical.shape[1] >= 4:
        nir = optical[:, 3:4]
    else:
        red, blue = optical[:, 0:1], optical[:, 2:3]
        nir = 1.0 - (red + green + blue) / 3.0

    ndwi = (green - nir) / (green + nir + eps)
    return ndwi.squeeze(0) if squeeze else ndwi


def compute_sar_water_mask(
    sar: torch.Tensor,
    threshold_db: float = -15.0,
) -> torch.Tensor:
    """
    Physics-calibrated SAR water mask using Refined Lee filtering and Sigma-0 dB thresholding
    (Sigma0 < -15 dB indicates specular reflection over calm water).
    """
    squeeze = sar.dim() == 3
    if squeeze:
        sar = sar.unsqueeze(0)

    masks = []
    for b in range(sar.shape[0]):
        band_np = (sar[b, 0].detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        _, sigma0_db = calibrate_to_sigma0_db(band_np)
        water_mask = (sigma0_db < threshold_db).astype(np.float32)
        masks.append(torch.from_numpy(water_mask).unsqueeze(0))

    out = torch.stack(masks, dim=0).to(sar.device)
    return out.squeeze(0) if squeeze else out