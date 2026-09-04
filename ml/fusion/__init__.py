"""
Optical-SAR Fusion Module
=========================

CLIP-style dual-encoder for aligning Sentinel-1 (SAR) and Sentinel-2 (optical)
image embeddings via contrastive pretraining on the SEN1-2 dataset.

Components:
    - DualEncoder: SAR + optical encoders with shared embedding space
    - ContrastiveLoss: NT-Xent / InfoNCE loss for cross-modal alignment
    - TerrainClassifier: Optional downstream classifier (agri/barren/grass/urban)
    - Training and inference scripts
"""

from ml.fusion.model import DualEncoder, ContrastiveLoss, TerrainClassifier

__all__ = ["DualEncoder", "ContrastiveLoss", "TerrainClassifier"]
