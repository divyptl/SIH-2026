"""Upload decoding, compatibility checking, and VLM-ready encoding.

Covers the problem statement's "input upload and compatibility checking" step:
uploads are GeoTIFF, TIFF, PNG or JPEG images, pairs must be spatially
corresponding, and every raster is normalised into an 8-bit RGB PNG data URI
before it reaches a vision-language model. Only a GeoTIFF carries
georeferencing; other formats are analysed in pixel space.
"""

from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass

from PIL import Image, ImageOps, UnidentifiedImageError

from schemas import ImageInfo, Modality

# GeoTIFF is a TIFF with extra tags, so Pillow reports both as "TIFF"; the
# georeferencing-tag check below is what separates a GeoTIFF from a plain TIFF.
ALLOWED_FORMATS = {"TIFF", "PNG", "JPEG"}

ALLOWED_EXTENSIONS = {
    ".tif",
    ".tiff",
    ".png",
    ".jpg",
    ".jpeg",
}

SUPPORTED_DESCRIPTION = "a GeoTIFF, TIFF, PNG or JPEG image"

# GeoTIFF private tags. Presence of either means the raster carries a CRS and
# a pixel-to-world transform, i.e. it is georeferenced rather than a plain TIFF.
_TAG_GEO_KEY_DIRECTORY = 34735
_TAG_MODEL_PIXEL_SCALE = 33550
_TAG_MODEL_TIEPOINT = 33922

# GeoKeys used to turn the pixel scale into metres.
_KEY_MODEL_TYPE = 1024          # 1 = projected, 2 = geographic (lat/lon)
_KEY_PROJ_LINEAR_UNITS = 3076   # EPSG unit code of a projected CRS
_MODEL_PROJECTED, _MODEL_GEOGRAPHIC = 1, 2
_UNIT_TO_METRES = {9001: 1.0, 9002: 0.3048, 9003: 0.3048006096}
_METRES_PER_DEGREE_LAT = 110_574.0
_METRES_PER_DEGREE_LON_EQUATOR = 111_320.0

# Pair sizes are allowed to differ by this fraction and still count as
# co-registered; anything beyond it is very unlikely to be the same footprint.
_ASPECT_TOLERANCE = 0.02

# Longest edge of the JPEG preview handed back to the client.
PREVIEW_MAX_EDGE_PX = 768

# Percentile trimmed from each end of the histogram when stretching a >8-bit
# raster. A handful of bright scatterers (urban double-bounce in SAR, specular
# water glint in optical) otherwise compress the whole scene into a few levels.
_STRETCH_CUTOFF_PCT = 1


class ImageValidationError(ValueError):
    """Raised when an upload cannot be used for analysis."""


@dataclass
class PreparedImage:
    """A validated upload plus its VLM-ready encoding."""

    info: ImageInfo
    data_uri: str
    # Metres per pixel of ``data_uri`` (after any downscaling), or None when
    # the upload carries no georeferencing to derive it from.
    analysis_gsd_m: float | None = None


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


def _geo_keys(tags) -> dict[int, int]:
    """GeoKey id -> inline value from the GeoKeyDirectory tag."""
    directory = tags.get(_TAG_GEO_KEY_DIRECTORY)
    if not directory or len(directory) < 4:
        return {}
    keys: dict[int, int] = {}
    for offset in range(4, 4 + 4 * int(directory[3]), 4):
        key_id, location, _count, value = directory[offset : offset + 4]
        if location == 0:  # value stored inline
            keys[int(key_id)] = int(value)
    return keys


def ground_sample_distance(image: Image.Image) -> float | None:
    """Metres per pixel of a GeoTIFF, from its pixel scale and CRS type.

    Projected CRSs give the scale in linear units (usually metres). Geographic
    CRSs give it in degrees, converted at the latitude of the tie point. Any
    other or incomplete georeferencing yields None rather than a guess.
    """
    tags = getattr(image, "tag_v2", None)
    scale = tags.get(_TAG_MODEL_PIXEL_SCALE) if tags is not None else None
    if not scale or len(scale) < 2 or scale[0] <= 0 or scale[1] <= 0:
        return None
    keys = _geo_keys(tags)
    model_type = keys.get(_KEY_MODEL_TYPE)
    if model_type == _MODEL_PROJECTED:
        factor = _UNIT_TO_METRES.get(keys.get(_KEY_PROJ_LINEAR_UNITS, 9001))
        return None if factor is None else (scale[0] + scale[1]) / 2 * factor
    if model_type == _MODEL_GEOGRAPHIC:
        tiepoint = tags.get(_TAG_MODEL_TIEPOINT)
        latitude = float(tiepoint[4]) if tiepoint and len(tiepoint) >= 6 else 0.0
        x = scale[0] * _METRES_PER_DEGREE_LON_EQUATOR * math.cos(math.radians(latitude))
        y = scale[1] * _METRES_PER_DEGREE_LAT
        return (x + y) / 2
    return None


def _stretch_high_bit_depth(image: Image.Image) -> Image.Image:
    """Linearly rescale a >8-bit raster into 8-bit greyscale.

    Converting straight to ``L`` *clips* at 255 rather than rescaling, so a
    16-bit Sentinel product — DN values run to five figures — arrives as a
    near-white blob, and a float sigma0 raster in dB (negative values) as pure
    black. Map the actual data range onto 0-255 first, then autocontrast with a
    small percentile cutoff to drop outliers.
    """
    # I;16 variants cannot be fed to point() directly; widening to "I" (32-bit
    # int) is lossless. Float rasters stay in "F" so that reflectance in 0..1
    # and backscatter in dB are not truncated to a couple of integer levels.
    wide = image if image.mode == "F" else image.convert("I")
    low, high = wide.getextrema()
    if high <= low:
        # Constant raster: nothing to stretch, and the scale below would divide
        # by zero.
        return Image.new("L", wide.size, 0)

    scale = 255.0 / (high - low)
    # point() keeps reporting mode "I" here, but the values it produces are
    # already inside 0..255, so the convert is lossless.
    rescaled = wide.point(lambda value: (value - low) * scale).convert("L")
    return ImageOps.autocontrast(rescaled, cutoff=_STRETCH_CUTOFF_PCT)


def _to_display_rgb(image: Image.Image) -> Image.Image:
    """Normalise any supported raster into 8-bit RGB.

    Single-band SAR and 16-bit panchromatic rasters are contrast-stretched to
    the full 8-bit range; without this a 16-bit GeoTIFF collapses to a flat
    single-colour image and the model has nothing to look at.
    """
    if image.mode in {"I;16", "I;16B", "I;16L", "I", "F"}:
        return _stretch_high_bit_depth(image).convert("RGB")
    if image.mode in {"L", "LA"}:
        return ImageOps.autocontrast(image.convert("L")).convert("RGB")
    if image.mode in {"RGBA", "P"}:
        # Flatten onto white rather than converting directly: dropping the
        # alpha channel via .convert("RGB") leaves transparent pixels at
        # whatever RGB value they happened to store (often black), which
        # shows up as a stray black wedge over no-data image edges.
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[3])
        return background
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
    """Validate one image upload and encode it as a PNG data URI."""
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
            f"'{filename}' is not a readable image. Uploads must be {SUPPORTED_DESCRIPTION}."
        ) from exc
    except OSError as exc:
        raise ImageValidationError(f"'{filename}' could not be decoded: {exc}") from exc

    detected = (image.format or "UNKNOWN").upper()
    if detected not in ALLOWED_FORMATS:
        raise ImageValidationError(
            f"'{filename}' is {detected}, which is not supported. "
            f"Uploads must be {SUPPORTED_DESCRIPTION}."
        )

    if detected == "JPEG":
        # Photos exported from phones and some viewers store rotation in EXIF.
        image = ImageOps.exif_transpose(image)

    georeferenced = _is_georeferenced(image)
    notes: list[str] = (
        ["GeoTIFF: georeferencing tags present."]
        if georeferenced
        else [f"{detected}: no georeferencing; locations are reported in image pixels."]
    )

    modality = _infer_modality(filename, image, declared_modality)
    width, height = image.size
    gsd_m = ground_sample_distance(image) if georeferenced else None

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
            ground_sample_distance_m=round(gsd_m, 3) if gsd_m is not None else None,
            notes=notes,
            preview_data_uri=f"data:image/jpeg;base64,{preview_encoded}",
        ),
        data_uri=f"data:image/png;base64,{encoded}",
        analysis_gsd_m=gsd_m * width / rgb.width if gsd_m is not None else None,
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
