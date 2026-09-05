"""
Siamese Change Detection / Change-VQA Specialist Module
======================================================

Specialist vision-language model for bi-temporal remote-sensing image analysis
and question answering, adapted for the CDVQA benchmark.

Components:
    - SiameseChangeVQA: End-to-end Siamese Vision Encoder + VLM Head
    - ChangeMaskHead: Dense change detection segmentation & grounding head
    - ChangeVQAModel: High-level inference engine compliant with SpecialistModel protocol
    - CDVQADataset: PyTorch Dataset for CDVQA benchmark and paired remote-sensing data
    - ModelConfig, TrainConfig: Architecture and training configuration dataclasses
"""

from ml.C_VQA.config import ModelConfig, TrainConfig
from ml.C_VQA.dataset import CDVQADataset
from ml.C_VQA.inference import ChangeVQAModel
from ml.C_VQA.model import (
    CANONICAL_ANSWERS,
    ChangeVQALoss,
    SiameseBackbone,
    SiameseChangeVQA,
)
from ml.C_VQA.transforms import PairedBitemporalTransform

__all__ = [
    "SiameseChangeVQA",
    "ChangeVQAModel",
    "ChangeVQALoss",
    "SiameseBackbone",
    "CDVQADataset",
    "PairedBitemporalTransform",
    "ModelConfig",
    "TrainConfig",
    "CANONICAL_ANSWERS",
]
