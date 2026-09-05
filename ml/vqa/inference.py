"""
SatQuery-facing wrapper for the VQA / Captioning specialist.

Implements:
    predict(request: ModelRequest) -> ModelResponse

per ml/controller/schema.py's SpecialistModel protocol (Section 5 of the
project spec). This interface must not be changed.

Also provides a CLI so the module can be exercised without the
frontend/backend/controller existing yet:

    python -m ml.vqa.inference --image sample.jpg --query "What type of land cover is visible?"
    python -m ml.vqa.inference --image sample.jpg --task captioning
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema: prefer the REAL project schema at ml/controller/schema.py. Fall
# back to a local copy only so this module is importable/testable
# standalone. If both definitions exist in the repo, ml/controller/schema.py
# is the source of truth — do not let this fallback drift from it.
# ---------------------------------------------------------------------------
try:
    from ..controller.schema import ModelRequest, ModelResponse  # type: ignore
except Exception:  # pragma: no cover - fallback only, exercised in isolated testing
    logger.warning(
        "Could not import ml.controller.schema; using a local fallback "
        "definition for ModelRequest/ModelResponse. Replace with the real "
        "import once ml/controller/schema.py is available on the path."
    )

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
from .model import BaseVQAModel, SkyEyeGPTModel
from .preprocessing import (
    load_image,
    UnsupportedFormatError,
    ImageReadError,
    UnsupportedBandCountError,
)
from .confidence import compute_confidence


class VQARequestError(ValueError):
    """Malformed/unsupported ModelRequest content, caught inside predict()."""


class VQAModel:
    """
    SatQuery specialist model for single-image VQA + captioning.

    Wraps a BaseVQAModel backend (SkyEyeGPT by default) behind the
    ModelRequest -> ModelResponse interface expected by the controller.
    The backend is constructed lazily so importing this module (e.g. for
    unit tests with a fake backend) never requires a GPU or checkpoint.
    """

    def __init__(self, backend: Optional[BaseVQAModel] = None, config: VQAConfig = DEFAULT_CONFIG):
        self.config = config
        self._backend = backend

    @property
    def backend(self) -> BaseVQAModel:
        if self._backend is None:
            self._backend = SkyEyeGPTModel(self.config)
        return self._backend

    # ------------------------------------------------------------------
    def _validate(self, request: ModelRequest) -> None:
        if not request.images:
            raise VQARequestError(
                "No image supplied. This module requires exactly one image."
            )
        if len(request.images) > 1:
            raise VQARequestError(
                "Multiple images supplied. Single-image VQA/captioning accepts "
                "exactly one image; multi-temporal questions belong to the "
                "Change-VQA specialist (ml/C_VQA), not this module."
            )
        if request.modalities:
            unknown = set(request.modalities) - SUPPORTED_MODALITIES
            if unknown:
                raise VQARequestError(f"Unsupported modality/modalities: {sorted(unknown)}")
        task = self._resolve_task(request)
        if task == "vqa" and not (request.query and request.query.strip()):
            raise VQARequestError("Empty query for a VQA request.")

    def _resolve_task(self, request: ModelRequest) -> str:
        if request.task_hint in ("vqa", "captioning"):
            return request.task_hint
        # No hint from the controller: infer a sensible default.
        return "vqa" if request.query and request.query.strip() else "captioning"

    # ------------------------------------------------------------------
    def predict(self, request: ModelRequest) -> ModelResponse:
        start = time.perf_counter()
        try:
            self._validate(request)

            modality = request.modalities[0] if request.modalities else "optical"
            image_path = request.images[0]
            image = load_image(image_path, modality=modality)

            task = self._resolve_task(request)
            if task == "captioning":
                answer = self.backend.caption(image)
                model_name = "SkyEyeGPT-Captioning"
            else:
                answer = self.backend.answer(image, request.query)
                model_name = "SkyEyeGPT-VQA"

            score = self.backend.generation_score()
            confidence = compute_confidence(answer, score)

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
                model_name="SkyEyeGPT-VQA",
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


# ---------------------------------------------------------------------------
# CLI: python -m ml.vqa.inference --image sample.jpg --query "..."
# ---------------------------------------------------------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SatQuery VQA/Captioning CLI (standalone test harness)")
    p.add_argument("--image", required=True, help="Path to the RS image (jpg/png/tif/tiff)")
    p.add_argument("--query", default="", help="Natural-language question (VQA mode)")
    p.add_argument(
        "--task", choices=["vqa", "captioning"], default=None,
        help="Force a task; default infers from --query",
    )
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
