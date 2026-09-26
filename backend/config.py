"""Runtime configuration for the SatQuery AI backend.

Values are read from the environment (``backend/.env`` is loaded automatically),
so nothing here needs to be edited to point at a different model or frontend.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Relative checkpoint paths resolve against the repository root, not the working
# directory, so the server finds them wherever it is started from.
REPO_ROOT = Path(__file__).resolve().parent.parent


def _csv(name: str, default: str) -> list[str]:
    return [part.strip() for part in os.getenv(name, default).split(",") if part.strip()]


def _flag(name: str, default: str) -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes"}


def _repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _checkpoint(name: str, default: str) -> Path | None:
    """A checkpoint path setting; set it to an empty value to disable that specialist."""
    value = os.getenv(name, default).strip()
    return _repo_path(value) if value else None


class Settings:
    """Process-wide settings, resolved once at import time."""

    def __init__(self) -> None:
        self.openrouter_api_key: str | None = os.getenv("OPENROUTER_API_KEY")

        # Vision-language model used for the actual image reasoning. Must be a
        # model with `image` in its OpenRouter input modalities.
        self.vision_model: str = os.getenv(
            "OPENROUTER_VISION_MODEL", "google/gemma-4-31b-it:free"
        )

        # Smaller/cheaper text-only call used by the controller to classify the
        # query into a task. Defaults to the same model to keep setup trivial.
        self.router_model: str = os.getenv("OPENROUTER_ROUTER_MODEL", self.vision_model)

        # Attribution headers OpenRouter shows on the account dashboard.
        self.app_title: str = os.getenv("OPENROUTER_APP_TITLE", "SatQuery AI")
        self.app_referer: str = os.getenv("OPENROUTER_APP_REFERER", "http://localhost:3000")

        self.request_timeout_s: float = float(os.getenv("OPENROUTER_TIMEOUT_S", "120"))
        self.max_image_bytes: int = int(os.getenv("MAX_IMAGE_BYTES", str(20 * 1024 * 1024)))

        # Longest edge the image is downscaled to before being sent upstream.
        # Keeps token cost bounded and stays well inside provider limits.
        self.max_image_edge_px: int = int(os.getenv("MAX_IMAGE_EDGE_PX", "1280"))

        # Indic <-> English translation layer (services/translation.py). Queries in
        # any supported Indian language are translated to English before routing,
        # and the answer is translated back. Both checkpoints are gated on Hugging
        # Face: accept their licence and put HF_TOKEN in this .env.
        self.translation_enabled: bool = os.getenv("TRANSLATION_ENABLED", "true").lower() in {
            "1",
            "true",
            "yes",
        }
        self.indic_en_model: str = os.getenv(
            "INDICTRANS_INDIC_EN_MODEL", "ai4bharat/indictrans2-indic-en-dist-200M"
        )
        self.en_indic_model: str = os.getenv(
            "INDICTRANS_EN_INDIC_MODEL", "ai4bharat/indictrans2-en-indic-dist-200M"
        )
        # "auto" picks CUDA when available, else CPU. Also accepts "cpu", "cuda:1", ...
        self.translation_device: str = os.getenv("TRANSLATION_DEVICE", "auto")
        self.translation_beams: int = int(os.getenv("TRANSLATION_BEAMS", "5"))
        self.translation_batch_size: int = int(os.getenv("TRANSLATION_BATCH_SIZE", "16"))
        # Load both checkpoints at startup instead of on the first non-English query.
        self.translation_preload: bool = os.getenv("TRANSLATION_PRELOAD", "false").lower() in {
            "1",
            "true",
            "yes",
        }

        # Fine-tuned specialists from ml/ (services/specialists.py). A task uses its
        # specialist when the checkpoint exists, and the OpenRouter baseline otherwise.
        # SPECIALISTS_ENABLED=false forces the baseline everywhere, for comparison.
        self.specialists_enabled: bool = _flag("SPECIALISTS_ENABLED", "true")
        self.grounding_checkpoint: Path | None = _checkpoint(
            "GROUNDING_CHECKPOINT", "checkpoints/grounding/v1/best.pt"
        )
        # "auto" uses every reranker*.pt beside the grounding checkpoint (they are
        # trained on that checkpoint's candidates); empty disables re-ranking.
        rerankers = os.getenv("GROUNDING_RERANKERS", "auto").strip()
        self.grounding_rerankers: list[Path] | None = (
            None if rerankers.lower() == "auto" else [_repo_path(p) for p in _csv("GROUNDING_RERANKERS", "")]
        )
        # Trained on LEVIR-CD (0.5 m aerial, building change): expect little signal
        # from coarse imagery such as 10 m Sentinel-2 or from non-building change.
        self.change_vqa_checkpoint: Path | None = _checkpoint(
            "CHANGE_VQA_CHECKPOINT", "checkpoints/c_vqa_best.pt"
        )
        self.fusion_checkpoint: Path | None = _checkpoint(
            "FUSION_CHECKPOINT", "checkpoints/fusion_best.pt"
        )
        # "auto" picks CUDA when available, else CPU. Also accepts "cpu", "cuda:1", ...
        self.specialist_device: str = os.getenv("SPECIALIST_DEVICE", "auto")
        # Load the specialists at startup instead of on their first request.
        self.specialist_preload: bool = _flag("SPECIALIST_PRELOAD", "false")

        # Plain-language narration of the Change-VQA result (agent/narration.py):
        # the VLM rewords the specialist's regions and measurements for non-expert
        # readers, and is rejected if it adds a region, number or place of its own.
        # Off by default: answers are the specialists' own wording.
        self.narration_enabled: bool = _flag("NARRATION_ENABLED", "false")
        self.narration_model: str = os.getenv("OPENROUTER_NARRATION_MODEL", self.vision_model)

        self.cors_origins: list[str] = _csv(
            "CORS_ORIGINS",
            "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173",
        )

    @property
    def is_configured(self) -> bool:
        return bool(self.openrouter_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
