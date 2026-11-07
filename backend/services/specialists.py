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
# Single-image VQA shows this many runner-up answers as evidence.
VQA_ALTERNATIVES = 3
# Below this confidence the VQA answer is flagged as uncertain.
VQA_LOW_CONFIDENCE = 0.4
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


def _load_vqa():
    from ml.vqa.inference import VQAModel

    settings = get_settings()
    return VQAModel.from_checkpoint(settings.vqa_checkpoint, device=settings.specialist_device)


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
_vqa = _Lazy(_load_vqa)
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
        "vqa": _vqa,
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


def box_position(x_min: float, y_min: float, x_max: float, y_max: float) -> str:
    """Where a normalised box sits in the image, as 'top-left', 'right', 'centre', ...

    Image-relative on purpose: an upload's orientation is unknown, so compass
    directions would be guesses.
    """
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    vertical = "top" if cy < 1 / 3 else "bottom" if cy > 2 / 3 else ""
    horizontal = "left" if cx < 1 / 3 else "right" if cx > 2 / 3 else ""
    if vertical and horizontal:
        return f"{vertical}-{horizontal}"
    return vertical or horizontal or "centre"


def position_phrase(position: str) -> str:
    """'top-left' -> 'in the top-left corner'; 'left' -> 'on the left side'."""
    if "-" in position:
        return f"in the {position} corner"
    return {
        "top": "near the top",
        "bottom": "near the bottom",
        "left": "on the left side",
        "right": "on the right side",
    }.get(position, "in the centre")


def size_phrase(share: float) -> str:
    """How much of the image a box covers, in words."""
    if share < 0.01:
        return "a very small area"
    if share < 0.05:
        return "a small area"
    if share < 0.25:
        return "a medium-sized area"
    return "a large part of the image"


def certainty_phrase(score: float) -> str:
    """The grounding score in words a non-expert can use."""
    if score >= 0.7:
        return "confident"
    if score >= 0.4:
        return "fairly confident"
    return "not very confident"


def run_grounding(query: str, images: list[PreparedImage]) -> dict[str, Any]:
    """Localise the region the query refers to with the fine-tuned GroundingDINO.

    Besides the answer and evidence, the payload carries ``narration_facts``:
    the boxes with their position, size and score, which the controller hands
    to the narrator (agent/narration.py) as the only facts it may put into words.
    """
    image = _pil(images[0]).convert("RGB")
    width, height = image.size
    detections = _grounding.get().ground(image, query)[:MAX_GROUNDING_BOXES]
    if not detections:
        return {
            "answer": f'No area in the image clearly matches "{query.strip()}".',
            "confidence": 0.0,
            "evidence": [],
        }

    evidence: list[dict[str, Any]] = []
    regions: list[dict[str, Any]] = []
    for rank, detection in enumerate(detections):
        x_min, y_min, x_max, y_max = detection["box"]
        box = (
            max(0.0, x_min / width), max(0.0, y_min / height),
            min(1.0, x_max / width), min(1.0, y_max / height),
        )
        position = box_position(*box)
        share = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
        score = detection["score"]
        place = position_phrase(position)
        number = rank + 1
        role = "Best match" if rank == 0 else "Other possible match"
        label = detection["label"]
        evidence.append(
            {
                "description": (
                    f"{role}, {place}, covering {size_phrase(share)}."
                ),
                # With a re-ranker the label is the whole query; only keep short phrases.
                "label": label if label and label.lower() != query.lower().strip() else None,
                "confidence": score,
                "image_index": 0,
                "box": {"x_min": box[0], "y_min": box[1], "x_max": box[2], "y_max": box[3]},
            }
        )
        regions.append(
            {
                "number": number,
                "evidence_index": len(evidence) - 1,
                "box": list(box),
                "position": position,
                "place": place,
                "share": share,
                "size": size_phrase(share),
                "confidence": score,
                "certainty": certainty_phrase(score),
                # Appended by the controller after the narrator's sentence for the box.
                "measure": f"{role}, {place}, covering {size_phrase(share)}.",
            }
        )

    best = regions[0]
    answer = (
        f'Box 1 marks the area that best matches "{query.strip()}", {best["place"]} '
        f'of the image. The model is {best["certainty"]} about this match.'
    )
    others = len(regions) - 1
    if others == 1:
        answer += " One other possible match is marked as box 2."
    elif others > 1:
        answer += f" {others} other possible matches are marked as boxes 2 to {others + 1}."
    return {
        "answer": answer,
        "confidence": best["confidence"],
        "evidence": evidence,
        "narration_facts": {"kind": "grounding", "query": query.strip(), "regions": regions},
    }


def run_vqa(query: str, images: list[PreparedImage]) -> dict[str, Any]:
    """Answer a factual question about one image with the single-image VQA model.

    The model picks its answer from a fixed vocabulary; the runner-up answers
    are returned as evidence so an uncertain answer is visible as such.
    Two-image questions never reach this runner: the registry only routes
    single images to VQA, and image pairs go to Change-VQA or fusion.
    """
    if len(images) != 1:
        raise ValueError(f"Single-image VQA takes exactly one image, got {len(images)}")
    result = _vqa.get().answer(_pil(images[0]), query, top_k=VQA_ALTERNATIVES + 1)
    answer = result["answer"]
    confidence = result["confidence"]

    alternatives = result["top"][1:]
    evidence: list[dict[str, Any]] = []
    if alternatives:
        evidence.append({
            "description": "Other likely answers: "
            + ", ".join(f"{a} ({p:.0%})" for a, p in alternatives) + ".",
            "image_index": 0,
        })
    if confidence < VQA_LOW_CONFIDENCE:
        evidence.append({
            "description": (
                f"The model is unsure ({confidence:.0%} confidence); treat this answer "
                "with caution and check it against the image."
            ),
            "image_index": 0,
        })
    return {
        "answer": f"{answer[:1].upper()}{answer[1:]}.",
        "confidence": confidence,
        "evidence": evidence,
    }


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
    """Run the dual encoder + deterministic SAR/Optical physics pipeline on a pair."""
    import sys
    ml_dir = str(REPO_ROOT / "ml")
    if ml_dir not in sys.path:
        sys.path.insert(0, ml_dir)

    from fusion.sar_physics import coregister_optical_sar, process_sar_physics
    from verification.physics_verifier import verify_landcover_claim, verify_water_claim

    by_modality = {image.info.modality: image for image in images}
    sar_img = _pil(by_modality["sar"])
    opt_img = _pil(by_modality["optical"])
    sar_index = by_modality["sar"].info.index

    # 1. Run trained Dual-Encoder Specialist
    specialist = _fusion.get()
    similarity = specialist.compare(sar_img, opt_img)
    terrain, confidence, probabilities = specialist.classify_terrain(sar_img, opt_img)

    # 2. Run True SAR Physics Calibration (Refined Lee 7x7 + Sigma0 dB + GLCM Homogeneity)
    sar_np = np.array(sar_img)
    opt_np = np.array(opt_img.convert("RGB"))
    vv_band = sar_np[:, :, 0] if sar_np.ndim == 3 else sar_np
    vh_band = sar_np[:, :, 1] if (sar_np.ndim == 3 and sar_np.shape[2] >= 2) else None

    sar_phys = process_sar_physics(vv_band, vh_band)

    # 3. Sub-pixel Optical-SAR Co-registration check
    reg_note = ""
    if opt_np.shape[:2] == sar_phys.sigma0_vv_db.shape:
        aligned_vv, reg_meta = coregister_optical_sar(
            opt_np[:, :, 1].astype(np.float32), sar_phys.sigma0_vv_db
        )
        sar_phys.sigma0_vv_db = aligned_vv
        reg_note = (
            f"Sub-pixel co-registration shift: dx={reg_meta['shift_x_px']}px, "
            f"dy={reg_meta['shift_y_px']}px (error={reg_meta['registration_error']})."
        )

    # 4. Cross-Modal Physics Verification (Water, Built-up, Vegetation)
    red = opt_np[:, :, 0].astype(np.float32)
    green = opt_np[:, :, 1].astype(np.float32)
    # Use specialist get_analysis() for NDWI if available, plus our physics verifier
    analysis = specialist.get_analysis(sar_img, opt_img)
    water_info = analysis.get("water", {})
    ndwi_mean = water_info.get("ndwi_mean", 0.0)
    ndwi_water_pct = water_info.get("ndwi_water_pixel_ratio", 0.0)
    sar_water_pct = sar_phys.metadata["water_pixel_fraction_est"]
    vv_db_mean = sar_phys.metadata["vv_db_mean"]

    water_ver = verify_water_claim(
        model_confidence=confidence,
        green=green,
        nir=red,
        sar_vv_db=sar_phys.sigma0_vv_db,
        sar_homogeneity=sar_phys.glcm_homogeneity,
        is_specialist=True,
    )

    ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    terrain_summary = ", ".join(
        f"{TERRAIN_NAMES.get(name, name)} {p:.0%}" for name, p in ranked
    )

    evidence = [
        {
            "description": (
                f"Terrain probabilities from fused optical and SAR features: {terrain_summary}."
            ),
            "image_index": sar_index,
        },
        {
            "description": (
                f"Optical–SAR embedding cosine similarity is {similarity:.2f}. {reg_note}".strip()
            ),
            "image_index": sar_index,
        },
        {
            "description": (
                f"SAR Physics (Refined Lee 7x7): mean Sigma0 VV = {vv_db_mean:.2f} dB, "
                f"specular water fraction (< -15 dB & high GLCM homogeneity) = {sar_water_pct:.1%}. "
                f"Optical NDWI mean = {ndwi_mean:.3f} ({ndwi_water_pct:.1%} water pixels)."
            ),
            "image_index": sar_index,
        },
        {
            "description": f"Cross-Modal Physics Verifier [{water_ver.status}]: {water_ver.audit_explanation}",
            "image_index": sar_index,
        },
    ]

    best_terrain_name = TERRAIN_NAMES.get(terrain, terrain)
    answer = (
        f"Based on the fused optical and SAR analysis, this region is classified as {best_terrain_name} "
        f"with {confidence:.0%} confidence.\n\n"
        f"Supporting physics & neural analysis:\n"
        f"- Optical and SAR embeddings have a cosine similarity of {similarity:.2f}. {reg_note}\n"
        f"- Calibrated SAR Backscatter (Refined Lee): mean VV = {vv_db_mean:.2f} dB; "
        f"{sar_water_pct:.1%} of the scene exhibits smooth specular water reflection (< -15 dB).\n"
        f"- Optical NDWI analysis shows {ndwi_water_pct:.1%} water coverage (mean NDWI: {ndwi_mean:.3f}).\n"
        f"- Physics Verifier Status: {water_ver.status} ({water_ver.audit_explanation})\n"
        f"Top predicted terrains: {terrain_summary}."
    )

    return {
        "answer": answer,
        "confidence": water_ver.verified_confidence if "water" in query.lower() else confidence,
        "evidence": evidence,
        "specialist_context": answer,
    }