"""Fine-tuned remote-sensing specialists from ``ml/``, served in-process.

Each runner is blocking (the controller calls it through ``asyncio.to_thread``)
and returns the same payload the OpenRouter baseline produces -- ``answer``,
``confidence`` and ``evidence`` with normalised ``box`` dicts -- so both paths
are aggregated, translated and exported identically.

Models load on first use, or at startup with SPECIALIST_PRELOAD, and stay
cached for the life of the process. ``ml`` itself is imported lazily so the API
still starts on a machine without the checkpoints or a GPU.
"""

from __future__ import annotations

import base64
import io
import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, Generic, TypeVar

import numpy as np
from PIL import Image

from config import REPO_ROOT, get_settings
from services.images import PreparedImage

logger = logging.getLogger("satquery.specialists")

T = TypeVar("T")

# Grounding returns the re-ranker's pick first, then other candidates above threshold.
MAX_GROUNDING_BOXES = 5
# Longest edge of the change-mask overlay; matches the preview it is drawn on.
MASK_OVERLAY_EDGE_PX = 768

TERRAIN_NAMES = {
    "agri": "agricultural land",
    "barrenland": "barren land",
    "grassland": "grassland",
    "urban": "urban area",
    "water": "water body",
}


class _Lazy(Generic[T]):
    """Load a model once, on first use, even when requests arrive concurrently."""

    def __init__(self, load: Callable[[], T]) -> None:
        self._load = load
        self._lock = threading.Lock()
        self._value: T | None = None

    def get(self) -> T:
        with self._lock:
            if self._value is None:
                self._value = self._load()
            return self._value


def checkpoint_label(path: Path) -> str:
    """How a checkpoint is named in the trace: repo-relative when possible."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.name


def _pil(image: PreparedImage) -> Image.Image:
    """The exact raster the baseline VLM would have been shown."""
    raw = base64.b64decode(image.data_uri.split(",", 1)[1])
    return Image.open(io.BytesIO(raw))


def _rerankers(checkpoint: Path) -> list[str]:
    configured = get_settings().grounding_rerankers
    paths = sorted(checkpoint.parent.glob("reranker*.pt")) if configured is None else configured
    return [str(path) for path in paths]


# --- Loaders ---------------------------------------------------------------------


def _load_grounding():
    from ml.grounding.inference import GroundingInference

    settings = get_settings()
    checkpoint = settings.grounding_checkpoint
    assert checkpoint is not None
    return GroundingInference.from_checkpoint(
        str(checkpoint),
        device=settings.specialist_device,
        reranker_path=_rerankers(checkpoint) or None,
    )


def _load_change_vqa():
    from ml.C_VQA.inference import ChangeVQAModel

    settings = get_settings()
    return ChangeVQAModel.from_checkpoint(
        settings.change_vqa_checkpoint, device=settings.specialist_device
    )


def _load_fusion():
    from ml.fusion.inference import FusionModel

    settings = get_settings()
    return FusionModel.from_checkpoint(
        str(settings.fusion_checkpoint), device=settings.specialist_device
    )


_grounding = _Lazy(_load_grounding)
_change_vqa = _Lazy(_load_change_vqa)
_fusion = _Lazy(_load_fusion)


def describe_grounding(checkpoint: Path) -> str:
    rerankers = _rerankers(checkpoint)
    suffix = f" + {len(rerankers)} re-ranker(s)" if rerankers else ""
    return f"{checkpoint_label(checkpoint)}{suffix}"


def preload(tasks: set[str]) -> None:
    """Load the specialists for ``tasks`` now, so no request waits on a model load."""
    loaders = {
        "grounding": _grounding,
        "change_vqa": _change_vqa,
        "change_description": _change_vqa,
        "fusion": _fusion,
    }
    for lazy in {loaders[task] for task in tasks if task in loaders}:
        try:
            lazy.get()
        except Exception:
            logger.exception("Specialist preload failed; it will be retried on first use")


# --- Runners ---------------------------------------------------------------------


def run_grounding(query: str, images: list[PreparedImage]) -> dict[str, Any]:
    """Localise the region the query refers to with the fine-tuned GroundingDINO."""
    image = _pil(images[0]).convert("RGB")
    width, height = image.size
    detections = _grounding.get().ground(image, query)[:MAX_GROUNDING_BOXES]

    evidence = []
    for rank, detection in enumerate(detections):
        x_min, y_min, x_max, y_max = detection["box"]
        label = detection["label"]
        evidence.append(
            {
                "description": (
                    "Best match for the query." if rank == 0 else "Other candidate region."
                ),
                # With a re-ranker the label is the whole query; only keep short phrases.
                "label": label if label and label.lower() != query.lower().strip() else None,
                "confidence": detection["score"],
                "image_index": 0,
                "box": {
                    "x_min": max(0.0, x_min / width),
                    "y_min": max(0.0, y_min / height),
                    "x_max": min(1.0, x_max / width),
                    "y_max": min(1.0, y_max / height),
                },
            }
        )

    if not detections:
        return {
            "answer": "The grounding model found no region matching the query.",
            "confidence": 0.0,
            "evidence": [],
        }
    others = len(detections) - 1
    answer = "The region the query refers to is marked as box 1."
    if others:
        answer += f" {others} other candidate region(s) scored above the detection threshold."
    return {"answer": answer, "confidence": detections[0]["score"], "evidence": evidence}


def change_vqa_gsd_range() -> tuple[float, float]:
    """Metres per pixel the Change-VQA checkpoint's training tiles covered."""
    return _change_vqa.get().gsd_range_m


def _mask_overlay(mask_prob: np.ndarray, threshold: float) -> str:
    """The change mask as a PNG data URI: opaque where changed, transparent elsewhere.

    The client uses it as a CSS mask over a coloured layer, so the colour stays
    a UI decision. It is stretched over the image, so only the aspect matters.
    """
    changed = Image.fromarray(((mask_prob >= threshold) * 255).astype(np.uint8), "L")
    height, width = mask_prob.shape
    scale = min(1.0, MASK_OVERLAY_EDGE_PX / max(width, height))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    overlay = Image.new("RGBA", size, (255, 255, 255, 0))
    overlay.putalpha(changed.resize(size, Image.NEAREST))
    buffer = io.BytesIO()
    overlay.save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def format_km2(area: float) -> str:
    """An area in km², rounded to the precision a reader can use."""
    if area < 1:
        return f"{area:.2f} km²"
    if area < 10:
        return f"{area:.1f} km²"
    return f"{area:.0f} km²"


def run_change_vqa(query: str, images: list[PreparedImage]) -> dict[str, Any]:
    """Answer a change question over a bi-temporal pair with the Siamese Change-VQA model.

    Besides the answer and evidence, the payload carries ``narration_facts``:
    everything the model measured, which the controller hands to the narrator
    (agent/narration.py) as the only facts it may put into words.
    """
    specialist = _change_vqa.get()
    after = _pil(images[1])
    result = specialist.analyze_pair(_pil(images[0]), after, query)

    # Ground area of the analysed raster, when the upload is georeferenced.
    gsd = images[1].analysis_gsd_m
    scene_km2 = after.width * after.height * gsd**2 / 1e6 if gsd else None

    def measure(share: float, where: str | None = None) -> str:
        area = f" (about {format_km2(share * scene_km2)})" if scene_km2 else ""
        text = f"{share:.1%} of the scene{area}"
        return f"{text}, in {where}" if where else text

    changed_share = result["change_percentage"] / 100
    changed = measure(changed_share)
    total = result["total_regions"]
    evidence: list[dict[str, Any]] = [
        {
            "label": "change mask",
            "description": (
                f"{changed[:1].upper()}{changed[1:]} is predicted as changed, "
                f"in {total} {'region' if total == 1 else 'regions'}."
            ),
            # Shown on the later acquisition, where the change is visible.
            "image_index": 1,
            "mask": _mask_overlay(result["mask_prob"], specialist.config.mask_threshold),
        }
    ]
    regions: list[dict[str, Any]] = []
    for region in result["bounding_boxes"]:
        x_min, y_min, x_max, y_max = region["normalized_bbox"]
        label = region["label"]
        where = measure(region["share"], region["sector"])
        regions.append(
            {
                # The number the region carries on screen: its 1-based evidence position.
                "number": len(evidence) + 1,
                "evidence_index": len(evidence),
                "label": label,
                "label_confidence": region["label_confidence"],
                "share": region["share"],
                "area_km2": region["share"] * scene_km2 if scene_km2 else None,
                "sector": region["sector"],
                "box": [x_min, y_min, x_max, y_max],
                "measure": f"{where[:1].upper()}{where[1:]}.",
            }
        )
        evidence.append(
            {
                "label": label,
                "description": f"{label[:1].upper()}{label[1:]}: {where}.",
                # How sure the model is of the label, the figure shown next to it.
                "confidence": region["label_confidence"],
                "image_index": 1,
                "box": {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max},
            }
        )

    return {
        "answer": result["answer"],
        "confidence": result["confidence"],
        "evidence": evidence,
        "narration_facts": {
            "model_answer": result["answer"],
            "model_confidence": result["confidence"],
            "changed_share": changed_share,
            "changed_km2": changed_share * scene_km2 if scene_km2 else None,
            "gsd_m": gsd,
            "total_regions": total,
            "regions": regions,
        },
    }


def run_fusion(query: str, images: list[PreparedImage]) -> dict[str, Any]:
    """Run the dual encoder on a co-registered optical + SAR pair.

    Returns specialist evidence (terrain classification, cross-modal similarity,
    NDWI water index, SAR water mask) **without** a canned answer.  The
    controller detects the missing answer and calls the VLM baseline with the
    specialist context injected so the language model can answer the user's
    actual question informed by domain-adapted data.
    """
    by_modality = {image.info.modality: image for image in images}
    sar = _pil(by_modality["sar"])
    optical = _pil(by_modality["optical"])
    sar_index = by_modality["sar"].info.index

    specialist = _fusion.get()
    similarity = specialist.compare(sar, optical)
    terrain, confidence, probabilities = specialist.classify_terrain(sar, optical)

    # Full analysis includes NDWI and SAR water mask
    analysis = specialist.get_analysis(sar, optical)
    water_info = analysis.get("water", {})
    ndwi_water_pct = water_info.get("ndwi_water_pixel_ratio", 0.0)
    sar_water_pct = water_info.get("sar_water_pixel_ratio", 0.0)
    ndwi_mean = water_info.get("ndwi_mean", 0.0)

    ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    terrain_summary = ", ".join(
        f"{TERRAIN_NAMES.get(name, name)} {p:.0%}" for name, p in ranked
    )

    evidence = [
        {
            "description": (
                f"Terrain probabilities from the fused optical and SAR features: "
                f"{terrain_summary}."
            ),
            "image_index": sar_index,
        },
        {
            "description": (
                f"Optical\u2013SAR embedding similarity is {similarity:.2f} "
                "(1 means the two views agree strongly; low values suggest a "
                "misaligned or mismatched pair)."
            ),
            "image_index": sar_index,
        },
        {
            "description": (
                f"NDWI (Normalized Difference Water Index) analysis: "
                f"mean NDWI = {ndwi_mean:.3f}, "
                f"{ndwi_water_pct:.0%} of optical pixels indicate water "
                f"(NDWI > 0). SAR low-backscatter water mask: "
                f"{sar_water_pct:.0%} of SAR pixels indicate smooth water."
            ),
            "image_index": sar_index,
        },
    ]

    # Structured context the VLM will receive in its prompt.
    specialist_context = (
        f"A fine-tuned optical-SAR dual encoder (terrain classifier) reports: "
        f"{terrain_summary}. "
        f"Top prediction: {TERRAIN_NAMES.get(terrain, terrain)} ({confidence:.0%}). "
        f"Optical\u2013SAR embedding cosine similarity: {similarity:.2f}. "
        f"NDWI water analysis: mean NDWI = {ndwi_mean:.3f}, "
        f"{ndwi_water_pct:.0%} of optical pixels indicate water (NDWI > 0). "
        f"SAR low-backscatter water mask: {sar_water_pct:.0%} of pixels are smooth water."
    )

    return {
        "answer": None,  # signals the controller to augment with VLM
        "confidence": confidence,
        "evidence": evidence,
        "specialist_context": specialist_context,
    }
