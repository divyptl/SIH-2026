"""
Inference for the single-image VQA specialist.

Two entry points:
    VQAModel.answer(image, question)  -> {"answer", "confidence", "top"}
        used by the backend (backend/services/specialists.py)
    VQAModel.predict(ModelRequest)    -> ModelResponse
        the ml/controller/schema.py SpecialistModel protocol

The model is single-image only: it answers by choosing from the answer
vocabulary it was trained with. Two-image (change) questions belong to
ml.C_VQA, and captions are not something a classifier can write.

    python -m ml.vqa.inference --image tile.png --query "How many ships are there?"
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

from .model import VQAModel as CoreVQAModel
from .preprocessing import (
    ImageReadError,
    UnsupportedBandCountError,
    UnsupportedFormatError,
    load_image,
)
from .training.dataset import eval_transform

logger = logging.getLogger(__name__)

try:
    from ..controller.schema import ModelRequest, ModelResponse  # type: ignore
except Exception:  # pragma: no cover
    @dataclass
    class ModelRequest:  # type: ignore
        query: str
        images: List[str]
        modalities: List[str]
        task_hint: Optional[str] = None

    @dataclass
    class ModelResponse:  # type: ignore
        answer: str
        confidence: float
        evidence: list
        model_name: str
        execution_time_ms: float

DEFAULT_CHECKPOINT = "checkpoints/vqa/best.pt"
MODEL_NAME = "SatQuery-VQA"
# preprocessing.load_image renders each of these to RGB; the model was trained
# on optical RGB tiles, so other modalities are answered less reliably.
SUPPORTED_MODALITIES = {"optical", "multispectral", "sar"}


class VQARequestError(ValueError):
    """Malformed or unsupported request."""


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _to_pil(image, modality: str = "optical") -> Image.Image:
    if isinstance(image, (str, Path)):
        image = load_image(image, modality=modality)
    elif isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        image = Image.fromarray(image)
    if not isinstance(image, Image.Image):
        raise VQARequestError(f"Expected a path, PIL image or array, got {type(image).__name__}")
    return image.convert("RGB")


class VQAModel:
    """Single-image VQA specialist.

    Args:
        backend: A loaded ml.vqa.model.VQAModel. When omitted, it is loaded
            from `checkpoint` on first use.
        checkpoint: Checkpoint written by ml.vqa.training.train.
        device: 'auto', 'cpu', 'cuda', ...
    """

    def __init__(
        self,
        backend: Optional[CoreVQAModel] = None,
        checkpoint: str | Path = DEFAULT_CHECKPOINT,
        device: str = "auto",
    ) -> None:
        self.device = _resolve_device(device)
        self.checkpoint = Path(checkpoint)
        self._backend = backend
        if backend is not None:
            backend.to(self.device).eval()

    @classmethod
    def from_checkpoint(cls, checkpoint: str | Path, device: str = "auto") -> "VQAModel":
        wrapper = cls(checkpoint=checkpoint, device=device)
        _ = wrapper.backend      # load now, so a bad checkpoint fails here
        return wrapper

    @property
    def backend(self) -> CoreVQAModel:
        if self._backend is None:
            if not self.checkpoint.is_file():
                raise FileNotFoundError(
                    f"No VQA checkpoint at {self.checkpoint}. Train one with "
                    "`python -m ml.vqa.training.train`."
                )
            self._backend = CoreVQAModel.from_checkpoint(str(self.checkpoint), device=self.device)
        return self._backend

    @torch.no_grad()
    def answer(self, image, question: str, modality: str = "optical", top_k: int = 5) -> dict:
        """Answer one question about one image.

        Returns {"answer": str, "confidence": float, "top": [(answer, prob), ...]}
        with `top` sorted by probability, best first.
        """
        if not question or not question.strip():
            raise VQARequestError("Empty question.")
        model = self.backend
        size = getattr(model, "image_size", 256)
        pixels = eval_transform(size)(_to_pil(image, modality)).unsqueeze(0).to(self.device)
        out = model(image=pixels, question_text=question)
        probs = out["answer_logits"].float().softmax(-1)[0]
        values, indices = probs.topk(min(top_k, probs.numel()))
        top = [(model.answers_vocab[int(i)], float(v)) for v, i in zip(values, indices)]
        return {"answer": top[0][0], "confidence": top[0][1], "top": top}

    def predict(self, request: ModelRequest) -> ModelResponse:
        """SpecialistModel protocol: errors come back as an "ERROR: ..." answer."""
        start = time.perf_counter()
        try:
            if not request.images:
                raise VQARequestError("No image supplied. Exactly one image is required.")
            if len(request.images) > 1:
                raise VQARequestError(
                    "Multiple images supplied. Single-image VQA takes exactly one; questions "
                    "about change between two images go to Change-VQA (ml.C_VQA)."
                )
            modality = request.modalities[0] if request.modalities else "optical"
            if modality not in SUPPORTED_MODALITIES:
                raise VQARequestError(f"Unsupported modality: {modality}")
            if request.task_hint == "captioning":
                raise VQARequestError(
                    "Captioning is not supported: this model answers questions by choosing "
                    "from a fixed answer list."
                )
            result = self.answer(request.images[0], request.query, modality)
        except (VQARequestError, UnsupportedFormatError, ImageReadError,
                UnsupportedBandCountError) as exc:
            logger.warning("VQA request rejected: %s", exc)
            return ModelResponse(
                answer=f"ERROR: {exc}", confidence=0.0, evidence=[], model_name=MODEL_NAME,
                execution_time_ms=(time.perf_counter() - start) * 1000,
            )
        return ModelResponse(
            answer=result["answer"], confidence=result["confidence"], evidence=[],
            model_name=MODEL_NAME, execution_time_ms=(time.perf_counter() - start) * 1000,
        )


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Ask the single-image VQA model a question")
    p.add_argument("--image", required=True)
    p.add_argument("--query", required=True)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--modality", choices=sorted(SUPPORTED_MODALITIES), default="optical")
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    model = VQAModel.from_checkpoint(args.checkpoint, device=args.device)
    result = model.answer(args.image, args.query, args.modality)
    print(f"answer     : {result['answer']}")
    print(f"confidence : {result['confidence']:.3f}")
    print("top answers: " + ", ".join(f"{a} ({p:.2f})" for a, p in result["top"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
