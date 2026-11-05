from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, Any
import numpy as np


@dataclass
class VerificationResult:
    claim_type: str                 # "water", "vegetation", "built_up", "flood_change"
    modality_used: str              # "OPTICAL", "SAR", "OPTICAL_SAR_FUSION"
    raw_model_confidence: float
    verified_confidence: float
    status: str                     # "CONFIRMED", "UNCERTAIN", "CONTRADICTED", "SHADOW_REJECTED"
    supporting_pixel_fraction: float
    physical_metrics: Dict[str, float]
    audit_explanation: str


def _safe_normalized_difference(band_a: np.ndarray, band_b: np.ndarray) -> np.ndarray:
    a = band_a.astype(np.float32)
    b = band_b.astype(np.float32)
    return (a - b) / (a + b + 1e-8)


def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """McFeeters NDWI = (Green - NIR) / (Green + NIR). Water > 0.2."""
    return _safe_normalized_difference(green, nir)


def compute_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """NDVI = (NIR - Red) / (NIR + Red). Healthy vegetation > 0.3."""
    return _safe_normalized_difference(nir, red)


def compute_ndbi(swir: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """NDBI = (SWIR - NIR) / (SWIR + NIR). Built-up > 0.1."""
    return _safe_normalized_difference(swir, nir)


def compute_nbr(nir: np.ndarray, swir: np.ndarray) -> np.ndarray:
    """Normalized Burn Ratio = (NIR - SWIR) / (NIR + SWIR)."""
    return _safe_normalized_difference(nir, swir)


def _extract_roi(
    arr: np.ndarray,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    polygon_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """Extracts pixels inside a SAM-2 polygon mask or a [x1, y1, x2, y2] bounding box."""
    if polygon_mask is not None and polygon_mask.shape == arr.shape:
        roi = arr[polygon_mask > 0]
        if roi.size > 0:
            return roi
    if bbox is not None:
        x1, y1, x2, y2 = [max(0, int(v)) for v in bbox]
        h, w = arr.shape[:2]
        x2, y2 = min(w, max(x1 + 1, x2)), min(h, max(y1 + 1, y2))
        return arr[y1:y2, x1:x2].ravel()
    return arr.ravel()


def calibrate_confidence(
    model_conf: float,
    support_fraction: float,
    is_specialist: bool = False
) -> Tuple[float, str]:
    """
    Applies the SatQuery AI verification rules with the 0.75 hard cap for non-specialist models.
    >= 0.50 support -> CONFIRMED (x1.1, max 0.99)
    0.20 - 0.50     -> UNCERTAIN (x0.7)
    < 0.20          -> CONTRADICTED (x0.3)
    """
    if support_fraction >= 0.50:
        adjusted = min(model_conf * 1.1, 0.99)
        status = "CONFIRMED"
    elif support_fraction >= 0.20:
        adjusted = model_conf * 0.7
        status = "UNCERTAIN"
    else:
        adjusted = model_conf * 0.3
        status = "CONTRADICTED"

    ceiling = 0.99 if is_specialist else 0.75
    return round(min(adjusted, ceiling), 4), status


def verify_water_claim(
    model_confidence: float,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    polygon_mask: Optional[np.ndarray] = None,
    green: Optional[np.ndarray] = None,
    nir: Optional[np.ndarray] = None,
    sar_vv_db: Optional[np.ndarray] = None,
    sar_homogeneity: Optional[np.ndarray] = None,
    ndwi_threshold: float = 0.20,
    sar_water_db_threshold: float = -15.0,
    is_specialist: bool = False
) -> VerificationResult:
    """
    Unified Water Verifier supporting Optical-only, SAR-only, and Optical+SAR Fusion.
    Explicitly catches dark cloud/terrain shadows misclassified as water by optical VLMs.
    """
    has_opt = green is not None and nir is not None
    has_sar = sar_vv_db is not None

    metrics: Dict[str, float] = {}

    if has_opt:
        ndwi_map = compute_ndwi(green, nir)
        ndwi_roi = _extract_roi(ndwi_map, bbox, polygon_mask)
        opt_water_frac = float(np.mean(ndwi_roi > ndwi_threshold))
        metrics["mean_ndwi"] = round(float(np.mean(ndwi_roi)), 3)
        metrics["opt_water_fraction"] = round(opt_water_frac, 3)

    if has_sar:
        vv_roi = _extract_roi(sar_vv_db, bbox, polygon_mask)
        if sar_homogeneity is not None:
            hom_roi = _extract_roi(sar_homogeneity, bbox, polygon_mask)
            sar_water_mask = (vv_roi < sar_water_db_threshold) & (hom_roi > 0.55)
            metrics["mean_glcm_homogeneity"] = round(float(np.mean(hom_roi)), 3)
        else:
            sar_water_mask = vv_roi < sar_water_db_threshold
        sar_water_frac = float(np.mean(sar_water_mask))
        metrics["mean_sar_vv_db"] = round(float(np.mean(vv_roi)), 2)
        metrics["sar_water_fraction"] = round(sar_water_frac, 3)

    # Case 1: Optical + SAR Fusion (Cross-Modal Disambiguation)
    if has_opt and has_sar:
        # Shadow trap: Optical looks somewhat dark/ambiguous, but SAR has high backscatter (> -10 dB)
        if opt_water_frac >= 0.25 and sar_water_frac < 0.15 and metrics["mean_sar_vv_db"] > -11.0:
            final_conf = round(min(model_confidence * 0.3, 0.75), 4)
            return VerificationResult(
                claim_type="water",
                modality_used="OPTICAL_SAR_FUSION",
                raw_model_confidence=model_confidence,
                verified_confidence=final_conf,
                status="SHADOW_REJECTED",
                supporting_pixel_fraction=round(sar_water_frac, 3),
                physical_metrics=metrics,
                audit_explanation=(
                    f"Optical NDWI suggested water ({opt_water_frac:.1%}), but SAR backscatter is high "
                    f"({metrics['mean_sar_vv_db']} dB > {sar_water_db_threshold} dB), indicating terrain/cloud shadow rather than specular water."
                ),
            )
        # Joint fusion support (weighted 50% optical, 50% SAR, or SAR override under thick cloud)
        fused_frac = max(sar_water_frac, 0.5 * opt_water_frac + 0.5 * sar_water_frac)
        final_conf, status = calibrate_confidence(model_confidence, fused_frac, is_specialist)
        return VerificationResult(
            claim_type="water",
            modality_used="OPTICAL_SAR_FUSION",
            raw_model_confidence=model_confidence,
            verified_confidence=final_conf,
            status=status,
            supporting_pixel_fraction=round(fused_frac, 3),
            physical_metrics=metrics,
            audit_explanation=(
                f"Cross-verified via Optical NDWI ({opt_water_frac:.1%} > {ndwi_threshold}) and "
                f"SAR specular backscatter ({sar_water_frac:.1%} < {sar_water_db_threshold} dB)."
            ),
        )

    # Case 2: SAR-Only Water Verification (Closes the Round-1 SAR verification gap)
    if has_sar:
        final_conf, status = calibrate_confidence(model_confidence, sar_water_frac, is_specialist)
        return VerificationResult(
            claim_type="water",
            modality_used="SAR",
            raw_model_confidence=model_confidence,
            verified_confidence=final_conf,
            status=status,
            supporting_pixel_fraction=round(sar_water_frac, 3),
            physical_metrics=metrics,
            audit_explanation=(
                f"SAR-only physics check: {sar_water_frac:.1%} of grounded pixels exhibit specular reflection "
                f"(Sigma0 VV < {sar_water_db_threshold} dB, mean={metrics['mean_sar_vv_db']} dB)."
            ),
        )

    # Case 3: Optical-Only NDWI Verification
    if has_opt:
        final_conf, status = calibrate_confidence(model_confidence, opt_water_frac, is_specialist)
        return VerificationResult(
            claim_type="water",
            modality_used="OPTICAL",
            raw_model_confidence=model_confidence,
            verified_confidence=final_conf,
            status=status,
            supporting_pixel_fraction=round(opt_water_frac, 3),
            physical_metrics=metrics,
            audit_explanation=(
                f"Optical NDWI check: {opt_water_frac:.1%} of grounded pixels exceed NDWI > {ndwi_threshold} "
                f"(mean NDWI={metrics['mean_ndwi']})."
            ),
        )

    raise ValueError("At least one of (green, nir) or (sar_vv_db) must be provided.")


def verify_landcover_claim(
    claim_type: str,
    model_confidence: float,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    polygon_mask: Optional[np.ndarray] = None,
    red: Optional[np.ndarray] = None,
    nir: Optional[np.ndarray] = None,
    swir: Optional[np.ndarray] = None,
    sar_vv_db: Optional[np.ndarray] = None,
    sar_rvi: Optional[np.ndarray] = None,
    is_specialist: bool = False
) -> VerificationResult:
    """
    Verifies 'vegetation' (NDVI + Dual-Pol RVI) and 'built_up' (NDBI + SAR Double-Bounce > -6 dB).
    """
    claim = claim_type.lower().strip()
    metrics: Dict[str, float] = {}

    if claim == "vegetation":
        fracs = []
        if nir is not None and red is not None:
            ndvi_roi = _extract_roi(compute_ndvi(nir, red), bbox, polygon_mask)
            opt_f = float(np.mean(ndvi_roi > 0.30))
            metrics["mean_ndvi"] = round(float(np.mean(ndvi_roi)), 3)
            fracs.append(opt_f)
        if sar_rvi is not None:
            rvi_roi = _extract_roi(sar_rvi, bbox, polygon_mask)
            sar_f = float(np.mean(rvi_roi > 0.45))
            metrics["mean_sar_rvi"] = round(float(np.mean(rvi_roi)), 3)
            fracs.append(sar_f)
        support = float(np.mean(fracs)) if fracs else 0.0
        final_conf, status = calibrate_confidence(model_confidence, support, is_specialist)
        return VerificationResult(
            claim_type="vegetation",
            modality_used="OPTICAL_SAR_FUSION" if len(fracs) == 2 else ("SAR" if sar_rvi is not None else "OPTICAL"),
            raw_model_confidence=model_confidence,
            verified_confidence=final_conf,
            status=status,
            supporting_pixel_fraction=round(support, 3),
            physical_metrics=metrics,
            audit_explanation=f"Vegetation verified via NDVI/RVI volume scattering (support={support:.1%}).",
        )

    if claim in ("built_up", "urban", "building"):
        fracs = []
        if swir is not None and nir is not None:
            ndbi_roi = _extract_roi(compute_ndbi(swir, nir), bbox, polygon_mask)
            opt_f = float(np.mean(ndbi_roi > 0.05))
            metrics["mean_ndbi"] = round(float(np.mean(ndbi_roi)), 3)
            fracs.append(opt_f)
        if sar_vv_db is not None:
            vv_roi = _extract_roi(sar_vv_db, bbox, polygon_mask)
            # Corner reflectors / dihedral double-bounce from buildings produce high backscatter (> -7 dB)
            sar_f = float(np.mean(vv_roi > -7.0))
            metrics["mean_sar_vv_db"] = round(float(np.mean(vv_roi)), 2)
            fracs.append(sar_f)
        support = float(np.mean(fracs)) if fracs else 0.0
        final_conf, status = calibrate_confidence(model_confidence, support, is_specialist)
        return VerificationResult(
            claim_type="built_up",
            modality_used="OPTICAL_SAR_FUSION" if len(fracs) == 2 else ("SAR" if sar_vv_db is not None else "OPTICAL"),
            raw_model_confidence=model_confidence,
            verified_confidence=final_conf,
            status=status,
            supporting_pixel_fraction=round(support, 3),
            physical_metrics=metrics,
            audit_explanation=f"Built-up area verified via NDBI and/or SAR dihedral double-bounce > -7 dB (support={support:.1%}).",
        )

    raise ValueError(f"Unsupported claim_type: {claim_type}")