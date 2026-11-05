from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, Any
import numpy as np
from scipy.ndimage import uniform_filter
from skimage.feature import graycomatrix, graycoprops
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift as nd_shift
import rasterio
from rasterio.warp import reproject, Resampling


@dataclass
class SARPhysicsOutput:
    sigma0_vv_db: np.ndarray          # Calibrated VV backscatter in dB (float32)
    sigma0_vh_db: Optional[np.ndarray] # Calibrated VH backscatter in dB (float32)
    rvi: Optional[np.ndarray]          # Dual-Pol Radar Vegetation Index [0, 1]
    glcm_homogeneity: np.ndarray       # Local texture smoothness [0, 1] (high = water/flat)
    pseudo_rgb_uint8: np.ndarray       # 3-channel physics tensor (H, W, 3) for VLM/Backbone
    metadata: Dict[str, Any]


def classify_modality(
    image: np.ndarray,
    band_descriptions: Optional[list[str]] = None
) -> Dict[str, Any]:
    """
    Deterministic modality classifier upgrading the Round-1 speckle check.
    Expects image in (C, H, W) or (H, W, C) or (H, W).
    """
    if image.ndim == 2:
        bands, h, w = 1, image.shape[0], image.shape[1]
        arr = image[np.newaxis, ...]
    elif image.shape[0] <= 16 and image.shape[0] < image.shape[-1]:
        bands, h, w = image.shape
        arr = image
    else:
        h, w, bands = image.shape
        arr = np.transpose(image, (2, 0, 1))

    # 1. Check metadata band names first
    if band_descriptions:
        joined = " ".join(b.lower() for b in band_descriptions if b)
        if any(k in joined for k in ("vv", "vh", "hh", "hv", "sar", "sigma0")):
            return {"modality": "SAR", "confidence": 0.98, "reason": "metadata_band_names"}
        if any(k in joined for k in ("red", "green", "blue", "nir", "swir", "b02", "b03", "b04", "b08")):
            return {"modality": "OPTICAL", "confidence": 0.98, "reason": "metadata_band_names"}

    # 2. Compute coefficient of variation (speckle statistic: var / mean^2)
    sample = arr[0].astype(np.float64)
    mean_val = np.mean(sample) + 1e-8
    cv2_stat = np.var(sample) / (mean_val ** 2)

    if bands <= 2 and cv2_stat > 0.15:
        return {"modality": "SAR", "confidence": 0.90, "reason": f"bands={bands}, speckle_cv2={cv2_stat:.3f}"}
    if bands >= 3 and cv2_stat <= 0.15:
        return {"modality": "OPTICAL", "confidence": 0.90, "reason": f"bands={bands}, low_speckle_cv2={cv2_stat:.3f}"}

    # 3. Fallback when PNG/JPEG 3-channel SAR is uploaded (all channels nearly identical + high local variance)
    if bands >= 3:
        ch_diff = np.mean(np.abs(arr[0].astype(np.float32) - arr[1].astype(np.float32)))
        if ch_diff < 2.0 and cv2_stat > 0.12:
            return {"modality": "SAR", "confidence": 0.82, "reason": "grayscale_3ch_high_speckle"}

    fallback = "SAR" if bands <= 2 else "OPTICAL"
    return {"modality": fallback, "confidence": 0.60, "reason": f"fallback_band_count={bands}"}


def refined_lee_filter(img: np.ndarray, window_size: int = 7, enl: float = 4.4) -> np.ndarray:
    """
    Vectorized Lee Speckle Filter in linear power scale.
    Suppresses multiplicative radar speckle while preserving high-contrast edges (rivers, buildings).
    """
    eps = 1e-10
    img_f = np.maximum(img.astype(np.float64), eps)

    local_mean = uniform_filter(img_f, size=window_size)
    local_sq_mean = uniform_filter(img_f ** 2, size=window_size)
    local_var = np.maximum(local_sq_mean - local_mean ** 2, 0.0)

    # Noise variance constant from Equivalent Number of Looks (ENL)
    cu_sq = 1.0 / max(enl, 1.0)
    ci_sq = local_var / (local_mean ** 2 + eps)

    # Minimum Mean Square Error (MMSE) weight
    weight = np.clip((ci_sq - cu_sq) / (ci_sq + eps), 0.0, 1.0)
    filtered = local_mean + weight * (img_f - local_mean)
    return filtered.astype(np.float32)


def calibrate_to_sigma0_db(
    raw_band: np.ndarray,
    calibration_constant_db: float = 0.0,
    is_already_db: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts raw SAR Digital Numbers (DN) or amplitude to linear power and calibrated Sigma-0 (dB).
    Automatically detects if input is already in dB range [-40, +10] or 8-bit normalized [0, 255].
    """
    arr = raw_band.astype(np.float32)
    min_v, max_v = float(np.nanmin(arr)), float(np.nanmax(arr))

    if is_already_db or (min_v < 0.0 and max_v <= 15.0 and min_v >= -50.0):
        sigma0_db = np.clip(arr, -35.0, 5.0)
        sigma0_linear = np.power(10.0, sigma0_db / 10.0)
        return sigma0_linear, sigma0_db

    # If input is an 8-bit PNG demo image [0, 255], map linearly to physical SAR range [-25 dB, 0 dB]
    if max_v <= 255.0 and min_v >= 0.0 and raw_band.dtype == np.uint8:
        sigma0_db = (arr / 255.0) * 25.0 - 25.0
        sigma0_linear = np.power(10.0, sigma0_db / 10.0)
        return sigma0_linear, sigma0_db

    # Standard 16-bit / float amplitude DN -> linear power -> Lee filter -> dB
    power_linear = np.maximum(arr ** 2, 1e-10)
    filtered_linear = refined_lee_filter(power_linear, window_size=7)
    sigma0_db = 10.0 * np.log10(filtered_linear) - calibration_constant_db
    sigma0_db = np.clip(sigma0_db, -35.0, 5.0)
    return filtered_linear, sigma0_db


def compute_fast_glcm_homogeneity(sigma0_db: np.ndarray, patch_size: int = 16) -> np.ndarray:
    """
    Computes block-wise GLCM Homogeneity on quantized SAR dB values.
    Calm water exhibits specular reflection (low dB) AND high spatial homogeneity (> 0.65).
    """
    h, w = sigma0_db.shape
    # Quantize [-25, 0] dB into 16 gray levels for rapid GLCM calculation
    norm = np.clip((sigma0_db + 25.0) / 25.0, 0.0, 1.0)
    quantized = (norm * 15).astype(np.uint8)

    out_h = max(1, h // patch_size)
    out_w = max(1, w // patch_size)
    homogeneity_grid = np.zeros((out_h, out_w), dtype=np.float32)

    for i in range(out_h):
        for j in range(out_w):
            patch = quantized[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]
            glcm = graycomatrix(patch, distances=[1], angles=[0], levels=16, symmetric=True, normed=True)
            homogeneity_grid[i, j] = graycoprops(glcm, "homogeneity")[0, 0]

    # Upsample back to (H, W) using nearest neighbor repeat
    full_homogeneity = np.kron(homogeneity_grid, np.ones((patch_size, patch_size), dtype=np.float32))
    return full_homogeneity[:h, :w]


def process_sar_physics(
    vv_band: np.ndarray,
    vh_band: Optional[np.ndarray] = None,
    calibration_constant_db: float = 0.0
) -> SARPhysicsOutput:
    """
    Full SAR physics pipeline producing calibrated backscatter, Dual-Pol RVI, GLCM texture,
    and a 3-channel Physics Tensor ready for VLM / Grounding / Fusion encoders.
    """
    vv_lin, vv_db = calibrate_to_sigma0_db(vv_band, calibration_constant_db)
    vv_lin_filt = refined_lee_filter(vv_lin, window_size=7)
    vv_db_filt = np.clip(10.0 * np.log10(np.maximum(vv_lin_filt, 1e-10)), -30.0, 5.0)

    glcm_hom = compute_fast_glcm_homogeneity(vv_db_filt, patch_size=16)

    # Channel 1 (R): VV clipped to [-25, 0] dB (surface roughness / water vs urban)
    ch_r = np.clip((vv_db_filt + 25.0) / 25.0, 0.0, 1.0)

    if vh_band is not None:
        vh_lin, _ = calibrate_to_sigma0_db(vh_band, calibration_constant_db)
        vh_lin_filt = refined_lee_filter(vh_lin, window_size=7)
        vh_db_filt = np.clip(10.0 * np.log10(np.maximum(vh_lin_filt, 1e-10)), -35.0, 0.0)

        # Dual-Pol Radar Vegetation Index: 4 * VH / (VV + VH) in linear scale
        rvi = np.clip((4.0 * vh_lin_filt) / (vv_lin_filt + vh_lin_filt + 1e-10), 0.0, 1.0)

        # Channel 2 (G): VH clipped to [-30, -5] dB (volume scattering / vegetation)
        ch_g = np.clip((vh_db_filt + 30.0) / 25.0, 0.0, 1.0)
        # Channel 3 (B): Dual-Pol RVI
        ch_b = rvi
    else:
        vh_db_filt = None
        rvi = None
        # Single-pol fallback: Channel 2 = local spatial gradient, Channel 3 = GLCM homogeneity
        ch_g = np.clip((vv_db_filt + 20.0) / 20.0, 0.0, 1.0)
        ch_b = glcm_hom

    pseudo_rgb = (np.stack([ch_r, ch_g, ch_b], axis=-1) * 255.0).astype(np.uint8)

    meta = {
        "vv_db_mean": float(np.mean(vv_db_filt)),
        "vv_db_min": float(np.min(vv_db_filt)),
        "vv_db_max": float(np.max(vv_db_filt)),
        "water_pixel_fraction_est": float(np.mean((vv_db_filt < -15.0) & (glcm_hom > 0.60))),
        "dual_pol_available": vh_band is not None,
        "speckle_filter": "Refined_Lee_7x7",
    }
    return SARPhysicsOutput(
        sigma0_vv_db=vv_db_filt,
        sigma0_vh_db=vh_db_filt,
        rvi=rvi,
        glcm_homogeneity=glcm_hom,
        pseudo_rgb_uint8=pseudo_rgb,
        metadata=meta,
    )


def coregister_optical_sar(
    optical_luminance: np.ndarray,
    sar_vv_db: np.ndarray,
    upsample_factor: int = 10
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Sub-pixel translational co-registration using phase cross-correlation on structural edges.
    Aligns SAR array to the Optical reference grid.
    """
    opt_norm = (optical_luminance - np.mean(optical_luminance)) / (np.std(optical_luminance) + 1e-8)
    sar_norm = (sar_vv_db - np.mean(sar_vv_db)) / (np.std(sar_vv_db) + 1e-8)

    shift_vec, error, diffphase = phase_cross_correlation(
        opt_norm, sar_norm, upsample_factor=upsample_factor
    )
    dy, dx = float(shift_vec[0]), float(shift_vec[1])

    if abs(dy) < 50.0 and abs(dx) < 50.0:
        aligned_sar = nd_shift(sar_vv_db, shift=(dy, dx), mode="nearest")
        applied = True
    else:
        aligned_sar = sar_vv_db
        applied = False

    return aligned_sar, {
        "shift_y_px": round(dy, 2),
        "shift_x_px": round(dx, 2),
        "registration_error": round(float(error), 4),
        "subpixel_shift_applied": applied,
    }