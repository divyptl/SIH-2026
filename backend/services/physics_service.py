from __future__ import annotations

import base64
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

import cv2
import numpy as np

# Ensure the sibling 'ml' directory is importable when running from 'backend'
_ML_DIR = str(Path(__file__).resolve().parents[2] / "ml")
if _ML_DIR not in sys.path:
    sys.path.insert(0, _ML_DIR)

from fusion.sar_physics import (
    SARPhysicsOutput,
    classify_modality,
    coregister_optical_sar,
    process_sar_physics,
)
from verification.physics_verifier import (
    VerificationResult,
    verify_landcover_claim,
    verify_water_claim,
)


def _decode_data_uri(data_uri: str) -> np.ndarray:
    raw_bytes = base64.b64decode(data_uri.split(",", 1)[1])
    np_arr = np.frombuffer(raw_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Failed to decode image bytes.")
    # Convert BGR/BGRA from OpenCV to RGB/RGBA order
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    return img


def process_images_physics(images: list[Any], query: str) -> dict[str, Any]:
    """
    Processes 1 or 2 PreparedImage objects, separating Optical and SAR streams,
    running Refined Lee + Sigma0 dB calibration on SAR, and co-registering pairs.
    """
    sar_data: Optional[SARPhysicsOutput] = None
    optical_data: Optional[np.ndarray] = None
    registration_info: Optional[dict[str, float]] = None
    primary_shape: Tuple[int, int] = (1024, 1024)

    for idx, prep_img in enumerate(images):
        arr = _decode_data_uri(prep_img.data_uri)
        if idx == 0:
            primary_shape = arr.shape[:2]

        declared_mod = getattr(prep_img.info, "modality", "").lower()
        if declared_mod == "sar":
            mod = "SAR"
        elif declared_mod == "optical":
            mod = "OPTICAL"
        else:
            mod = classify_modality(arr)["modality"]

        if mod == "SAR" and sar_data is None:
            vv = arr[:, :, 0] if arr.ndim == 3 else arr
            vh = arr[:, :, 1] if (arr.ndim == 3 and arr.shape[2] >= 2) else None
            sar_data = process_sar_physics(vv, vh)
        elif mod == "OPTICAL" and optical_data is None:
            optical_data = arr

    # If both Optical and SAR are present, run sub-pixel co-registration
    if optical_data is not None and sar_data is not None:
        opt_gray = (
            optical_data[:, :, 1].astype(np.float32)
            if optical_data.ndim == 3
            else optical_data.astype(np.float32)
        )
        if opt_gray.shape == sar_data.sigma0_vv_db.shape:
            aligned_vv, registration_info = coregister_optical_sar(
                opt_gray, sar_data.sigma0_vv_db
            )
            sar_data.sigma0_vv_db = aligned_vv

    return {
        "sar_data": sar_data,
        "optical_data": optical_data,
        "registration_info": registration_info,
        "primary_shape": primary_shape,
    }


def run_verification(
    query: str,
    vlm_confidence: float,
    vlm_bbox: Optional[Tuple[int, int, int, int]],
    physics_context: dict[str, Any],
    is_specialist: bool = False,
) -> Optional[VerificationResult]:
    """
    Routes water/flood, vegetation, and built-up queries to the deterministic
    spectral + SAR backscatter verifier.
    """
    q = query.lower()
    opt = physics_context.get("optical_data")
    sar: Optional[SARPhysicsOutput] = physics_context.get("sar_data")

    if opt is None and sar is None:
        return None

    # Extract standard bands if Optical image is present (RGB or Multispectral)
    red = opt[:, :, 0].astype(np.float32) if (opt is not None and opt.ndim == 3) else None
    green = opt[:, :, 1].astype(np.float32) if (opt is not None and opt.ndim == 3) else None
    # If 4-band multispectral is provided, band 3 is NIR; for 3-band RGB demos, approximate NIR proxy using Red/Blue inversion only when true NIR is absent
    nir = (
        opt[:, :, 3].astype(np.float32)
        if (opt is not None and opt.ndim == 3 and opt.shape[2] >= 4)
        else (opt[:, :, 0].astype(np.float32) if opt is not None and opt.ndim == 3 else None)
    )

    if any(k in q for k in ("water", "flood", "river", "lake", "inundat")):
        return verify_water_claim(
            model_confidence=vlm_confidence,
            bbox=vlm_bbox,
            green=green,
            nir=nir,
            sar_vv_db=sar.sigma0_vv_db if sar else None,
            sar_homogeneity=sar.glcm_homogeneity if sar else None,
            is_specialist=is_specialist,
        )

    if any(k in q for k in ("vegetat", "forest", "crop", "green", "tree")):
        return verify_landcover_claim(
            claim_type="vegetation",
            model_confidence=vlm_confidence,
            bbox=vlm_bbox,
            red=red,
            nir=nir,
            sar_rvi=sar.rvi if sar else None,
            is_specialist=is_specialist,
        )

    if any(k in q for k in ("built", "urban", "building", "settlement", "house")):
        return verify_landcover_claim(
            claim_type="built_up",
            model_confidence=vlm_confidence,
            bbox=vlm_bbox,
            nir=nir,
            swir=red,  # fallback proxy when SWIR band is unavailable in RGB/4-band
            sar_vv_db=sar.sigma0_vv_db if sar else None,
            is_specialist=is_specialist,
        )

    return None