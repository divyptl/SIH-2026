"""
SatQuery-facing wrapper for the VQA / Captioning specialist.

Implements:
    predict(request: ModelRequest) -> ModelResponse

per ml/controller/schema.py's SpecialistModel protocol.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)

try:
    from ..controller.schema import ModelRequest, ModelResponse  # type: ignore
except Exception:  # pragma: no cover
    logger.warning("Could not import schema; using local fallback.")

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

from .config import VQAConfig, DEFAULT_CONFIG, SUPPORTED_MODALITIES
from .model import VQAModel as CoreVQAModel
from .preprocessing import (
    load_image,
    UnsupportedFormatError,
    ImageReadError,
    UnsupportedBandCountError,
)
from torchvision import transforms


class VQARequestError(ValueError):
    """Malformed/unsupported ModelRequest content."""


class VQAModel:
    """
    SatQuery specialist model for single-image VQA + captioning.

    Wraps the PyTorch VQAModel backend behind the ModelRequest -> ModelResponse
    interface expected by the controller.
    """

    def __init__(self, backend: Optional[CoreVQAModel] = None, config: VQAConfig = DEFAULT_CONFIG):
        self.config = config
        self._backend = backend
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @property
    def backend(self) -> CoreVQAModel:
        if self._backend is None:
            if self.config.checkpoint_path and self.config.checkpoint_path.exists():
                self._backend = CoreVQAModel.from_checkpoint(str(self.config.checkpoint_path), device=self.config.device)
            else:
                logger.warning(f"No checkpoint found at {self.config.checkpoint_path}. Using uninitialized weights.")
                self._backend = CoreVQAModel()
                self._backend.to(self.config.device)
                self._backend.eval()
        return self._backend

    def _validate(self, request: ModelRequest) -> None:
        if not request.images:
            raise VQARequestError("No image supplied. Exactly one image required.")
        if len(request.images) > 1:
            raise VQARequestError("Multiple images supplied. Single-image VQA accepts exactly one.")
        if request.modalities:
            unknown = set(request.modalities) - SUPPORTED_MODALITIES
            if unknown:
                raise VQARequestError(f"Unsupported modalities: {sorted(unknown)}")

    def _resolve_task(self, request: ModelRequest) -> str:
        if request.task_hint in ("vqa", "captioning"):
            return request.task_hint
        return "vqa" if request.query and request.query.strip() else "captioning"

    def predict(self, request: ModelRequest) -> ModelResponse:
        start = time.perf_counter()
        try:
            self._validate(request)
            modality = request.modalities[0] if request.modalities else "optical"
            image_path = request.images[0]
            
            # Load and transform image
            img = load_image(image_path, modality=modality)
            if hasattr(img, "astype"):
                import numpy as np
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8)
                from PIL import Image
                img = Image.fromarray(img)
            else:
                img = img.convert("RGB")
            
            img_tensor = self.transform(img).unsqueeze(0).to(self.config.device)

            task = self._resolve_task(request)
            query_str = request.query if task == "vqa" else "Describe this image."

            with torch.no_grad():
                outputs = self.backend(
                    image=img_tensor,
                    question_text=query_str
                )
            
            answer = outputs["predicted_answer_text"][0]
            confidence = outputs["answer_confidence"][0].item()
            model_name = "SatQuery-VQA-Custom"

        except (
            VQARequestError,
            UnsupportedFormatError,
            ImageReadError,
            UnsupportedBandCountError,
        ) as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.warning("VQAModel.predict rejected request: %s", exc)
            return ModelResponse(
                answer=f"ERROR: {exc}",
                confidence=0.0,
                evidence=[],
                model_name="SatQuery-VQA-Custom",
                execution_time_ms=elapsed_ms,
            )

        elapsed_ms = (time.perf_counter() - start) * 1000
        return ModelResponse(
            answer=answer,
            confidence=confidence,
            evidence=[],
            model_name=model_name,
            execution_time_ms=elapsed_ms,
        )


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SatQuery VQA CLI (standalone test harness)")
    p.add_argument("--image", required=True, help="Path to the RS image (jpg/png/tif/tiff)")
    p.add_argument("--query", default="", help="Natural-language question")
    p.add_argument("--task", choices=["vqa", "captioning"], default=None)
    p.add_argument("--modality", choices=sorted(SUPPORTED_MODALITIES), default="optical")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO)
    args = _build_arg_parser().parse_args(argv)

    task_hint = args.task or ("vqa" if args.query.strip() else "captioning")
    request = ModelRequest(
        query=args.query,
        images=[args.image],
        modalities=[args.modality],
        task_hint=task_hint,
    )

    model = VQAModel()
    response = model.predict(request)

    print(f"model_name       : {response.model_name}")
    print(f"answer           : {response.answer}")
    print(f"confidence       : {response.confidence:.3f}")
    print(f"execution_time_ms: {response.execution_time_ms:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
