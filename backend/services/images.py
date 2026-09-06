"""Upload decoding, compatibility checking, and VLM-ready encoding.

Covers the problem statement's "input upload and compatibility checking" step:
formats are restricted to GeoTIFF/TIFF plus the benchmark PNG/JPEG formats,
pairs must be spatially corresponding, and every raster is normalised into an
8-bit RGB PNG data URI before it reaches a vision-language model.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass

from PIL import Image, ImageOps, UnidentifiedImageError

from schemas import ImageInfo, Modality

# GeoTIFF is a TIFF with extra tags, so Pillow reports both as "TIFF".
GEOSPATIAL_FORMATS = {"TIFF"}
BENCHMARK_FORMATS = {"PNG", "JPEG", "WEBP"}
ALLOWED_FORMATS = GEOSPATIAL_FORMATS | BENCHMARK_FORMATS

ALLOWED_EXTENSIONS = {
    ".tif",
    ".tiff",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
}

# GeoTIFF private tags. Presence of either means the raster carries a CRS and
# a pixel-to-world transform, i.e. it is georeferenced rather than a plain TIFF.
_TAG_GEO_KEY_DIRECTORY = 34735
_TAG_MODEL_PIXEL_SCALE = 33550
_TAG_MODEL_TIEPOINT = 33922

# Pair sizes are allowed to differ by this fraction and still count as
# co-registered; anything beyond it is very unlikely to be the same footprint.
_ASPECT_TOLERANCE = 0.02

# Longest edge of the JPEG preview handed back to the client.
PREVIEW_MAX_EDGE_PX = 768


class ImageValidationError(ValueError):
    """Raised when an upload cannot be used for analysis."""


@dataclass
class PreparedImage:
    """A validated upload plus its VLM-ready encoding."""

    info: ImageInfo
    data_uri: str


def _infer_modality(filename: str, image: Image.Image, declared: str | None) -> Modality:
    """Best-effort modality detection.

    A declared modality from the client always wins. Otherwise fall back to
    filename hints and then to a channel heuristic: SAR amplitude products are
    single-band, optical/multispectral products are not.
    """
    if declared:
        normalised = declared.strip().lower()
        if normalised in {"optical", "multispectral"}:
            return "optical"
        if normalised == "sar":
            return "sar"

    name = filename.lower()
    if any(hint in name for hint in ("sar", "s1", "sentinel1", "sentinel-1", "risat", "grd")):
        return "sar"
    if any(
        hint in name
        for hint in ("optical", "s2", "sentinel2", "sentinel-2", "cartosat", "rgb", "msi")
    ):
        return "optical"

    if image.mode in {"L", "I;16", "I", "F"}:
        return "sar"
    if image.mode in {"RGB", "RGBA", "P", "CMYK", "YCbCr"}:
        return "optical"
    return "unknown"


def _is_georeferenced(image: Image.Image) -> bool:
    tags = getattr(image, "tag_v2", None)
    if tags is None:
        return False
    return any(
        tag in tags
        for tag in (_TAG_GEO_KEY_DIRECTORY, _TAG_MODEL_PIXEL_SCALE, _TAG_MODEL_TIEPOINT)
    )


def _to_display_rgb(image: Image.Image) -> Image.Image:
    """Normalise any supported raster into 8-bit RGB.

    Single-band SAR and 16-bit panchromatic rasters are contrast-stretched to
    the full 8-bit range; without this a 16-bit GeoTIFF renders as near-black
    and the model has nothing to look at.
    """
    if image.mode in {"I;16", "I;16B", "I;16L", "I", "F"}:
        stretched = ImageOps.autocontrast(image.convert("F").convert("L"))
        return stretched.convert("RGB")
    if image.mode in {"L", "LA"}:
        return ImageOps.autocontrast(image.convert("L")).convert("RGB")
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def prepare_upload(
    *,
    index: int,
    filename: str,
    content_type: str | None,
    raw: bytes,
    declared_modality: str | None,
    max_bytes: int,
    max_edge_px: int,
) -> PreparedImage:
    """Validate one upload and encode it as a PNG data URI."""
    if not raw:
        raise ImageValidationError(f"'{filename}' is empty.")

    if len(raw) > max_bytes:
        limit_mb = max_bytes / (1024 * 1024)
        raise ImageValidationError(
            f"'{filename}' is {len(raw) / (1024 * 1024):.1f} MB, over the {limit_mb:.0f} MB limit."
        )

    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except UnidentifiedImageError as exc:
        raise ImageValidationError(
            f"'{filename}' is not a readable image. Supported: GeoTIFF/TIFF, PNG, JPEG, WEBP."
        ) from exc
    except OSError as exc:
        raise ImageValidationError(f"'{filename}' could not be decoded: {exc}") from exc

    detected = (image.format or "UNKNOWN").upper()
    if detected not in ALLOWED_FORMATS:
        raise ImageValidationError(
            f"'{filename}' is {detected}, which is not supported. "
            "Use GeoTIFF/TIFF for geospatial imagery, or PNG/JPEG for benchmark data."
        )

    notes: list[str] = []
    georeferenced = detected == "TIFF" and _is_georeferenced(image)
    if detected == "TIFF":
        notes.append(
            "GeoTIFF: georeferencing tags present."
            if georeferenced
            else "Plain TIFF: no georeferencing tags found, results are pixel-space only."
        )

    modality = _infer_modality(filename, image, declared_modality)
    width, height = image.size

    rgb = _to_display_rgb(image)
    if rgb is not image:
        notes.append(f"Normalised from mode '{image.mode}' to 8-bit RGB for the vision model.")

    if max(rgb.size) > max_edge_px:
        rgb = ImageOps.contain(rgb, (max_edge_px, max_edge_px))
        notes.append(f"Downscaled to {rgb.width}x{rgb.height} before upstream inference.")

    buffer = io.BytesIO()
    rgb.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

    # Smaller JPEG copy for the client. Derived from the same normalised raster
    # the model receives, so overlaid evidence boxes line up with what it saw.
    preview = ImageOps.contain(rgb, (PREVIEW_MAX_EDGE_PX, PREVIEW_MAX_EDGE_PX))
    preview_buffer = io.BytesIO()
    preview.save(preview_buffer, format="JPEG", quality=82, optimize=True)
    preview_encoded = base64.b64encode(preview_buffer.getvalue()).decode("ascii")

    return PreparedImage(
        info=ImageInfo(
            index=index,
            filename=filename,
            content_type=content_type,
            detected_format=detected,
            modality=modality,
            width=width,
            height=height,
            size_bytes=len(raw),
            is_georeferenced=georeferenced,
            notes=notes,
            preview_data_uri=f"data:image/jpeg;base64,{preview_encoded}",
        ),
        data_uri=f"data:image/png;base64,{encoded}",
    )


def check_pair_compatibility(images: list[PreparedImage]) -> list[str]:
    """Compatibility checks for a two-image submission.

    Returns non-fatal warnings. Genuine blockers raise instead, because sending
    mismatched footprints to a change or fusion model yields confident nonsense.
    """
    if len(images) != 2:
        return []

    first, second = images[0].info, images[1].info
    warnings: list[str] = []

    if (first.width, first.height) != (second.width, second.height):
        aspect_a = first.width / first.height
        aspect_b = second.width / second.height
        if abs(aspect_a - aspect_b) / max(aspect_a, aspect_b) > _ASPECT_TOLERANCE:
            raise ImageValidationError(
                f"Paired images must cover the same footprint, but "
                f"'{first.filename}' is {first.width}x{first.height} and "
                f"'{second.filename}' is {second.width}x{second.height}. "
                "Co-register and crop them to a matching extent first."
            )
        warnings.append(
            f"Pair resolutions differ ({first.width}x{first.height} vs "
            f"{second.width}x{second.height}) but aspect ratios match; assuming co-registration."
        )

    if first.modality == second.modality and first.modality != "unknown":
        warnings.append(
            f"Both images were detected as {first.modality}; treating this as a "
            "bi-temporal pair rather than a cross-modal pair."
        )

    if first.is_georeferenced != second.is_georeferenced:
        warnings.append(
            "Only one image of the pair is georeferenced; spatial evidence is reported "
            "in pixel space."
        )

    return warnings
