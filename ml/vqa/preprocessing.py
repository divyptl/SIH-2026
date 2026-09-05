"""
Image loading & preprocessing for SatQuery VQA.

Pipeline:
    input file -> detect_format -> (GeoTIFF: read_geotiff via rasterio) ->
    inspect_bands -> normalize_image -> 3-channel PIL.Image -> model

Modality caveat (read before trusting SAR/multispectral results):
    Multispectral and SAR inputs are converted here to a *documented* RGB
    visualization so SkyEyeGPT (a conventional visual-image model) can
    consume them. This is a VISUALIZATION, not native multispectral/SAR
    understanding. Do not claim "the model supports SAR" on this basis —
    see README.md.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

from .config import SUPPORTED_EXTENSIONS, DEFAULT_MULTISPECTRAL_RGB_BANDS

logger = logging.getLogger(__name__)


class UnsupportedFormatError(ValueError):
    """Raised for file extensions this module does not (yet) support."""


class ImageReadError(ValueError):
    """Raised when a file exists but cannot be read/decoded."""


class UnsupportedBandCountError(ValueError):
    """Raised when band count/shape can't be mapped to an RGB composite."""


def detect_format(path: Path) -> str:
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFormatError(
            f"Unsupported file extension '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    return "geotiff" if ext in (".tif", ".tiff") else "standard"


def read_geotiff(path: Path) -> Tuple[np.ndarray, dict]:
    """Returns (array [bands, H, W], metadata dict). Requires `rasterio`."""
    try:
        import rasterio
    except ImportError as exc:
        raise ImportError(
            "rasterio is required for GeoTIFF support. Install with "
            "`pip install rasterio`."
        ) from exc

    try:
        with rasterio.open(path) as src:
            arr = src.read()  # (bands, H, W)
            # Metadata (CRS/transform/bounds) is preserved here rather than
            # discarded, in case downstream components (grounding, fusion)
            # need it later — per the project spec's note not to drop it.
            meta = {
                "crs": src.crs,
                "transform": src.transform,
                "bounds": src.bounds,
                "count": src.count,
                "dtype": src.dtypes[0],
            }
    except ImportError:
        raise
    except Exception as exc:  # rasterio raises its own varied exception types
        raise ImageReadError(f"Could not read GeoTIFF at {path}: {exc}") from exc

    return arr, meta


def inspect_bands(arr: np.ndarray) -> int:
    if arr.ndim == 2:
        return 1
    if arr.ndim == 3:
        return arr.shape[0]
    raise UnsupportedBandCountError(f"Unexpected array shape {arr.shape}")


def _percentile_stretch(band: np.ndarray, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    lo, hi = np.percentile(band, (low, high))
    if hi <= lo:
        hi = lo + 1e-6
    stretched = np.clip((band.astype(np.float32) - lo) / (hi - lo), 0, 1)
    return (stretched * 255).astype(np.uint8)


def normalize_image(
    arr: np.ndarray,
    modality: str = "optical",
    rgb_bands: Tuple[int, int, int] = DEFAULT_MULTISPECTRAL_RGB_BANDS,
) -> np.ndarray:
    """
    Convert an arbitrary-band array to an 8-bit 3-channel RGB array.

    - 1 band (e.g. SAR amplitude): grayscale, percentile-stretched, then
      replicated to 3 channels. Documented as a visualization, not native
      SAR understanding.
    - 3 bands, modality != "multispectral": assumed already RGB-ordered;
      each channel percentile-stretched independently.
    - >3 bands (multispectral): builds an RGB composite from `rgb_bands`
      (1-indexed positions), default (3, 2, 1) — see config.py. This
      mapping must stay documented wherever results are shown/reported.
    """
    n_bands = inspect_bands(arr)

    if n_bands == 1:
        band = arr if arr.ndim == 2 else arr[0]
        stretched = _percentile_stretch(band)
        rgb = np.stack([stretched] * 3, axis=-1)
        logger.info(
            "normalize_image: 1-band input (modality=%s) -> grayscale RGB visualization",
            modality,
        )
        return rgb

    if n_bands == 3 and modality != "multispectral":
        chans = [_percentile_stretch(arr[i]) for i in range(3)]
        return np.stack(chans, axis=-1)

    if n_bands >= max(rgb_bands):
        r, g, b = rgb_bands
        chans = [_percentile_stretch(arr[idx - 1]) for idx in (r, g, b)]
        logger.info(
            "normalize_image: %d-band multispectral input -> RGB composite using "
            "1-indexed bands %s (documented default mapping, see config.py)",
            n_bands, rgb_bands,
        )
        return np.stack(chans, axis=-1)

    raise UnsupportedBandCountError(
        f"Cannot build RGB composite: array has {n_bands} band(s) but "
        f"rgb_bands={rgb_bands} requires at least {max(rgb_bands)}."
    )


def load_image(path, modality: str = "optical") -> Image.Image:
    """
    Top-level entry point: file path -> model-ready PIL.Image (RGB).

    Raises UnsupportedFormatError / ImageReadError / UnsupportedBandCountError
    on bad input — callers (inference.py) are expected to catch these and
    turn them into a graceful ModelResponse error rather than crashing.
    """
    path = Path(path)
    if not path.exists():
        raise ImageReadError(f"Image path does not exist: {path}")

    fmt = detect_format(path)

    if fmt == "geotiff":
        arr, _meta = read_geotiff(path)
        rgb = normalize_image(arr, modality=modality)
        return Image.fromarray(rgb, mode="RGB")

    try:
        img = Image.open(path)
        img.load()
    except Exception as exc:
        raise ImageReadError(f"Could not read image at {path}: {exc}") from exc

    if img.mode != "RGB":
        img = img.convert("RGB")
    return img
