"""
ml.vqa — Single-image Remote-Sensing VQA + Captioning specialist.

Owner: Yashvi (AI/ML — VQA / captioning)

Public API:
    VQAModel       — SatQuery controller-facing wrapper (predict()).
    BaseVQAModel   — backend interface (swap SkyEyeGPT for another RS-VLM
                     such as GeoChat / RS-LLaVA / BLIP-2 without touching
                     inference.py or the controller).
    SkyEyeGPTModel — MiniGPT-v2 runtime + SkyEyeGPT checkpoint backend.

Do NOT import ml.C_VQA from here — that module belongs to the
Change Detection / Change-VQA specialist (owner: Harivansh) and has a
different two-image (T1, T2) interface. This module is single-image only.
"""
from .inference import VQAModel
from .model import BaseVQAModel, SkyEyeGPTModel

__all__ = ["VQAModel", "BaseVQAModel", "SkyEyeGPTModel"]
