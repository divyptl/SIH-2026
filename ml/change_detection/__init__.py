"""
Change Detection / Change-VQA Package Alias
===========================================

Aliases ml.C_VQA for consistency with project documentation and controller schema.
"""

from ml.C_VQA import (
    CANONICAL_ANSWERS,
    CDVQADataset,
    ChangeVQALoss,
    ChangeVQAModel,
    ModelConfig,
    PairedBitemporalTransform,
    SiameseBackbone,
    SiameseChangeVQA,
    TrainConfig,
)

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
