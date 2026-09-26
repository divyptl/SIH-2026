"""
Optical-SAR Fusion Module
=========================

CLIP-style dual-encoder for aligning Sentinel-1 (SAR) and Sentinel-2 (optical)
image embeddings via contrastive pretraining on the SEN1-2 dataset.

Components:
    - DualEncoder: SAR + optical encoders with shared embedding space
    - ContrastiveLoss: NT-Xent / InfoNCE loss for cross-modal alignment
    - TerrainClassifier: Optional downstream classifier (agri/barren/grass/urban/water)
    - Training and inference scripts
    - compute_ndwi / compute_sar_water_mask: Water detection spectral indices
"""

from ml.fusion.model import DualEncoder, ContrastiveLoss, TerrainClassifier
from ml.fusion.transforms import compute_ndwi, compute_sar_water_mask

__all__ = [
    "DualEncoder",
    "ContrastiveLoss",
    "TerrainClassifier",
    "compute_ndwi",
    "compute_sar_water_mask",
]
