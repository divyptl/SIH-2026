"""
Configuration for the SatQuery VQA / Captioning module (ml/vqa).

Centralizes every path and hyperparameter needed to load MiniGPT-v2 +
the SkyEyeGPT checkpoint, and to run inference.

IMPORTANT — API STABILITY:
    The MiniGPT-v2 config keys referenced here mirror the structure of
    the official Vision-CAIR/MiniGPT-4 repo's `eval_configs/minigptv2_eval.yaml`
    at the time this was written. If the upstream repo has changed,
    re-check https://github.com/Vision-CAIR/MiniGPT-4 and update
    accordingly rather than assuming this is still correct.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Root of this vqa module (ml/vqa/)
MODULE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Checkpoints — NEVER commit these. See .gitignore note in README.
# ---------------------------------------------------------------------------
CHECKPOINTS_DIR = Path(
    os.environ.get("SATQUERY_CHECKPOINTS_DIR", MODULE_DIR.parent.parent / "checkpoints")
)
SKYEYEGPT_CHECKPOINT = CHECKPOINTS_DIR / "SkyEyeGPT.pth"

# MiniGPT-v2 is a separate cloned repo (not a pip package). It needs its own
# base LLM weights + an eval config YAML. These are environment-specific,
# so they're read from env vars with fallbacks a setup script should fill in.
MINIGPT_REPO_DIR = Path(
    os.environ.get("MINIGPT_REPO_DIR", MODULE_DIR.parent.parent / "third_party" / "MiniGPT-4")
)
MINIGPT_EVAL_CONFIG = Path(
    os.environ.get("MINIGPT_EVAL_CONFIG", MINIGPT_REPO_DIR / "eval_configs" / "minigptv2_eval.yaml")
)

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
DEVICE = os.environ.get("SATQUERY_DEVICE", "cuda:0")
DTYPE = os.environ.get("SATQUERY_DTYPE", "fp16")  # "fp16" | "bf16" | "fp32"


@dataclass
class GenerationConfig:
    max_new_tokens: int = 300
    num_beams: int = 1
    temperature: float = 1.0
    do_sample: bool = False
    repetition_penalty: float = 1.05


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
CAPTION_INSTRUCTION = (
    "Describe the remote sensing image in detail. Mention the major "
    "land-cover types, objects, structures, roads, water bodies and "
    "vegetation that are visually supported by the image. "
    "Do not invent information that is not visible."
)

# ---------------------------------------------------------------------------
# Supported inputs
# ---------------------------------------------------------------------------
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
SUPPORTED_MODALITIES = {"optical", "sar", "multispectral"}

# Documented band mapping used for multispectral -> RGB visualization.
# Sentinel-2-style band ordering assumed (Red=4, Green=3, Blue=2 in native
# Sentinel-2 numbering; here expressed as 1-indexed positions into whatever
# band stack is actually read). MUST be documented wherever it's used —
# see preprocessing.normalize_image().
DEFAULT_MULTISPECTRAL_RGB_BANDS = (3, 2, 1)  # (Red, Green, Blue) 1-indexed band positions


@dataclass
class VQAConfig:
    device: str = DEVICE
    dtype: str = DTYPE
    checkpoint_path: Path = SKYEYEGPT_CHECKPOINT
    minigpt_eval_config: Path = MINIGPT_EVAL_CONFIG
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    multispectral_rgb_bands: tuple = DEFAULT_MULTISPECTRAL_RGB_BANDS


DEFAULT_CONFIG = VQAConfig()
