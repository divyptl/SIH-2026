"""Runtime configuration for the SatQuery AI backend.

Values are read from the environment (``backend/.env`` is loaded automatically),
so nothing here needs to be edited to point at a different model or frontend.
"""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


def _csv(name: str, default: str) -> list[str]:
    return [part.strip() for part in os.getenv(name, default).split(",") if part.strip()]


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
