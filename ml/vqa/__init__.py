"""
ml.vqa — Single-image Remote-Sensing VQA + Captioning specialist.

Owner: Yashvi (AI/ML — VQA / captioning)

Public API:
    VQAModel       — SatQuery controller-facing wrapper (predict()).
    CoreVQAModel   — backend PyTorch model.

Do NOT import ml.C_VQA from here — that module belongs to the
Change Detection / Change-VQA specialist (owner: Harivansh) and has a
different two-image (T1, T2) interface. This module is single-image only.
"""
from .inference import VQAModel
from .model import VQAModel as CoreVQAModel

__all__ = ["VQAModel", "CoreVQAModel"]
