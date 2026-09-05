"""
Text-Guided Region Grounding Module
====================================

GroundingDINO-based model for localizing objects in remote-sensing imagery
via natural-language text queries. Fine-tuned on the VRSBench grounding
subset (52K object references on 29K satellite images).

Components:
    - GroundingModel: Wrapper around HF GroundingDINO with selective freezing
    - GroundingInference: High-level inference API (load, ground, visualize)
    - VRSBenchGroundingDataset: PyTorch dataset for VRSBench grounding data
    - Training and inference scripts

Quick start:
    # Zero-shot inference (no fine-tuning needed)
    from ml.grounding.inference import GroundingInference
    model = GroundingInference.from_pretrained()
    results = model.ground("satellite.png", "water body near bridge")

    # Fine-tuned inference
    model = GroundingInference.from_checkpoint("checkpoints/grounding/best.pt")
    results = model.ground("satellite.png", "buildings")

    # Training
    python -m ml.grounding.train --epochs 20 --batch-size 4

Author: Jainee
"""

from ml.grounding.config import ModelConfig, TrainConfig
from ml.grounding.model import GroundingModel
from ml.grounding.inference import GroundingInference

__all__ = [
    "ModelConfig",
    "TrainConfig",
    "GroundingModel",
    "GroundingInference",
]
