"""
Model abstraction for SatQuery VQA/Captioning.

`BaseVQAModel` defines the minimal interface every RS-VLM backend must
implement, so SkyEyeGPT can later be swapped for GeoChat / RS-LLaVA /
another remote-sensing VLM without touching inference.py or the
controller (see project README's note that BLIP-2/RS-LLaVA/GeoChat are
all possible VQA/captioning approaches).

`SkyEyeGPTModel` loads MiniGPT-v2's runtime with the SkyEyeGPT checkpoint.
This project treats SkyEyeGPT as "MiniGPT-v2 runtime + SkyEyeGPT
checkpoint", NOT a standalone package, because the official SkyEyeGPT
repo does not currently ship chatbot/inference code (marked "coming
soon" upstream).

NOTE ON API STABILITY (read before relying on this in production):
    The MiniGPT-v2 import paths and `Chat` class usage below reflect the
    upstream Vision-CAIR/MiniGPT-4 repository's demo/eval scripts as
    documented at the time this was written. Verify against the actual
    checked-out version of that repo — do not assume these signatures are
    unchanged, and adjust `_load()` / `generate()` if they differ.
"""
from __future__ import annotations

import abc
import logging
from pathlib import Path
from typing import Optional

from PIL import Image

from .config import VQAConfig, DEFAULT_CONFIG, CAPTION_INSTRUCTION

logger = logging.getLogger(__name__)


class BaseVQAModel(abc.ABC):
    """Common interface for any single-image RS-VLM backend."""

    @abc.abstractmethod
    def answer(self, image: Image.Image, question: str) -> str:
        """Answer a natural-language question about a single RS image."""
        raise NotImplementedError

    @abc.abstractmethod
    def caption(self, image: Image.Image) -> str:
        """Generate a detailed caption for a single RS image."""
        raise NotImplementedError

    @abc.abstractmethod
    def generate(self, image: Image.Image, prompt: str) -> str:
        """Free-form generation given an arbitrary instruction/prompt."""
        raise NotImplementedError

    def generation_score(self) -> Optional[float]:
        """
        Optional: a confidence-relevant signal (e.g. average token
        log-probability) for the *last* call to answer/caption/generate.
        Return None if the backend does not expose this — callers must
        fall back to a heuristic (see confidence.py).
        """
        return None


class SkyEyeGPTModel(BaseVQAModel):
    """
    RS-VLM backend = MiniGPT-v2 runtime + SkyEyeGPT checkpoint.

    Loading happens once in __init__ and is reused across calls.
    Do NOT re-instantiate this class per-request — it's expensive.
    """

    def __init__(self, config: VQAConfig = DEFAULT_CONFIG):
        self.config = config
        self._last_score: Optional[float] = None
        self._chat = None
        self._model = None
        self._vis_processor = None
        self._conv_template = None
        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        if not Path(self.config.checkpoint_path).exists():
            raise FileNotFoundError(
                f"SkyEyeGPT checkpoint not found at {self.config.checkpoint_path}. "
                "Download it from the SkyEyeGPT Hugging Face repo and place it "
                "there, or set SATQUERY_CHECKPOINTS_DIR. Do NOT commit this "
                "file to git (see .gitignore)."
            )
        if not Path(self.config.minigpt_eval_config).exists():
            raise FileNotFoundError(
                f"MiniGPT-v2 eval config not found at {self.config.minigpt_eval_config}. "
                "Clone https://github.com/Vision-CAIR/MiniGPT-4 and set "
                "MINIGPT_REPO_DIR, or edit config.py directly. That YAML's "
                "`model.ckpt` field must point at the SkyEyeGPT checkpoint "
                "above — SkyEyeGPT is distributed as a MiniGPT-v2-compatible "
                "checkpoint, not a separate codebase."
            )

        try:
            # Lazy import: MiniGPT-4 is a separate cloned repo, not a pip
            # package, so it must already be on PYTHONPATH (see README).
            from minigpt4.common.config import Config as MiniGPTConfig
            from minigpt4.common.registry import registry
            from minigpt4.conversation.conversation import Chat, CONV_VISION_minigptv2
        except ImportError as exc:
            raise ImportError(
                "Could not import minigpt4. Clone "
                "https://github.com/Vision-CAIR/MiniGPT-4 and add it to "
                "PYTHONPATH (or `pip install -e .` inside that repo) before "
                "using SkyEyeGPTModel. See ml/vqa/README.md, 'Setup'."
            ) from exc

        import argparse

        args = argparse.Namespace(
            cfg_path=str(self.config.minigpt_eval_config),
            options=[f"model.ckpt={self.config.checkpoint_path}"],
        )
        mgpt_cfg = MiniGPTConfig(args)

        model_cls = registry.get_model_class(mgpt_cfg.model_cfg.arch)
        model = model_cls.from_config(mgpt_cfg.model_cfg).to(self.config.device)
        model.eval()

        first_dataset_key = next(iter(mgpt_cfg.datasets_cfg))
        vis_processor_cfg = mgpt_cfg.datasets_cfg.get(first_dataset_key).vis_processor.train
        vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(
            vis_processor_cfg
        )

        self._model = model
        self._vis_processor = vis_processor
        self._chat = Chat(model, vis_processor, device=self.config.device)
        self._conv_template = CONV_VISION_minigptv2
        logger.info("SkyEyeGPT (via MiniGPT-v2) loaded on %s", self.config.device)

    # ------------------------------------------------------------------
    def generate(self, image: Image.Image, prompt: str) -> str:
        conv = self._conv_template.copy()
        img_list: list = []
        self._chat.upload_img(image, conv, img_list)
        self._chat.encode_img(img_list)
        self._chat.ask(prompt, conv)

        gen_cfg = self.config.generation
        raw = self._chat.answer(
            conv=conv,
            img_list=img_list,
            num_beams=gen_cfg.num_beams,
            temperature=gen_cfg.temperature,
            max_new_tokens=gen_cfg.max_new_tokens,
            max_length=2000,
        )

        # Some MiniGPT-v2 versions return (answer, score); others return
        # just the answer string. Handle both without assuming either.
        if isinstance(raw, tuple):
            answer, score = raw[0], (raw[1] if len(raw) > 1 else None)
        else:
            answer, score = raw, None

        self._last_score = score
        return str(answer).strip()

    def answer(self, image: Image.Image, question: str) -> str:
        return self.generate(image, question)

    def caption(self, image: Image.Image) -> str:
        return self.generate(image, CAPTION_INSTRUCTION)

    def generation_score(self) -> Optional[float]:
        return self._last_score
