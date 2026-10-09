"""PDF export of an analysis result.

The report is typeset with Typst (``assets/report.typ``) rather than printed
from HTML: it gets real pagination, running headers, PDF bookmarks and
HarfBuzz-grade shaping for every Indic script and right-to-left Urdu, Kashmiri
and Sindhi. Fonts ship in ``assets/fonts`` and system fonts are ignored, so a
report renders identically on any machine.

The endpoint is stateless: the client posts back the ``AnalysisResponse`` it
already holds, plus the UI strings in its current language.
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import shutil
import tempfile
from pathlib import Path

import typst
from PIL import Image

from schemas import ReportRequest
from services.translation import ENGLISH, LANGUAGES, dominant_script, resolve_source_language

ASSETS = Path(__file__).resolve().parent.parent / "assets"
TEMPLATE = ASSETS / "report.typ"
FONTS = ASSETS / "fonts"

# Urdu and Kashmiri are read in Nastaliq; Sindhi is written in Naskh.
_NASTALIQ_LANGUAGES = {"ur", "ks"}


class ReportError(RuntimeError):
    """The report could not be rendered."""


def _decode_preview(data_uri: str) -> tuple[bytes, str, float]:
    """Image bytes, file extension and height/width ratio of a preview data URI."""
    header, _, payload = data_uri.partition(",")
    if not header.startswith("data:image/") or ";base64" not in header:
        raise ReportError("Image preview is not a base64 image data URI.")
    try:
        raw = base64.b64decode(payload, validate=True)
    except binascii.Error as exc:
        raise ReportError("Image preview is not valid base64.") from exc
    try:
        with Image.open(io.BytesIO(raw)) as image:
            width, height = image.size
            kind = (image.format or "").lower()
    except OSError as exc:
        raise ReportError("Image preview could not be decoded.") from exc
    if kind not in {"jpeg", "png"}:
        raise ReportError(f"Unsupported preview format: {kind or 'unknown'}.")
    return raw, "jpg" if kind == "jpeg" else "png", height / width


def render_report(request: ReportRequest) -> bytes:
    """Typeset ``request`` into a PDF and return its bytes."""
    result = request.result
    language = LANGUAGES.get(request.language, LANGUAGES[ENGLISH])
    translation = result.translation
    # Only render the localised copy when it exists and matches the language
    # asked for; otherwise the report is the English original throughout.
    localised = (
        language.code != ENGLISH
        and translation is not None
        and translation.answer is not None
        and translation.target_language == language.code
    )
    query_language = resolve_source_language(request.query, request.language).language

    workdir = Path(tempfile.mkdtemp(prefix="satquery-report-"))
    try:
        images = []
        for info in result.inputs:
            if not info.preview_data_uri:
                continue
            raw, extension, aspect = _decode_preview(info.preview_data_uri)
            name = f"image-{info.index}.{extension}"
            (workdir / name).write_bytes(raw)
            images.append({"index": info.index, "path": name, "aspect": aspect})

        # The previews are already on disk; don't ship them to Typst twice.
        payload = result.model_dump(mode="json")
        for info in payload["inputs"]:
            info["preview_data_uri"] = None

        data = {
            "result": payload,
            "labels": request.labels.model_dump(mode="json"),
            "lang": language.code,
            "rtl": language.rtl,
            "nastaliq": language.code in _NASTALIQ_LANGUAGES,
            "localised": localised,
            "query": request.query,
            "query_lang": query_language.code,
            "query_rtl": dominant_script(request.query) == "Arab",
            "images": images,
        }
        (workdir / "data.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        shutil.copyfile(TEMPLATE, workdir / "report.typ")

        try:
            return typst.compile(
                str(workdir / "report.typ"),
                root=str(workdir),
                font_paths=[str(FONTS)],
                ignore_system_fonts=True,
            )
        except Exception as exc:  # typst raises its own TypstError / RuntimeError
            raise ReportError(f"Typst failed to render the report: {exc}") from exc
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
