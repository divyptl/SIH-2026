"""SatQuery AI backend.

FastAPI service exposing the agentic remote-sensing analysis workflow.

Run with:  uv run fastapi dev main.py
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from agent.controller import AgenticController, ControllerError
from agent.registry import describe_registry
from config import get_settings
from schemas import (
    AnalysisResponse,
    HealthResponse,
    LanguageInfo,
    LanguagesResponse,
    RegistryResponse,
    ReportRequest,
    TraceStep,
    TranslationInfo,
)
from services.images import (
    ImageValidationError,
    PreparedImage,
    prepare_upload,
)
from services.openrouter_client import OpenRouterClient, OpenRouterError
from services.report import ReportError, render_report
from services.translation import (
    ENGLISH,
    LANGUAGES,
    TranslationError,
    get_translator,
    resolve_source_language,
)

logger = logging.getLogger("satquery.api")

settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    if settings.translation_preload:
        # Load in the background so a slow model download never blocks startup.
        def _warm() -> None:
            try:
                get_translator().warm_up()
            except TranslationError as exc:
                logger.warning("Translation preload skipped: %s", exc)

        threading.Thread(target=_warm, name="translation-preload", daemon=True).start()
    yield


app = FastAPI(
    lifespan=lifespan,
    title="SatQuery AI",
    version="0.1.0",
    description=(
        "Agentic vision-language assistant for multimodal remote-sensing image "
        "analysis through text queries."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness plus whether the upstream model is actually configured."""
    return HealthResponse(
        status="ok" if settings.is_configured else "degraded",
        openrouter_configured=settings.is_configured,
        vision_model=settings.vision_model,
        router_model=settings.router_model,
    )


@app.get("/api/registry", response_model=RegistryResponse)
def registry() -> RegistryResponse:
    """The specialist-model registry the controller selects from."""
    return RegistryResponse(tools=describe_registry(settings.vision_model))


@app.get("/api/languages", response_model=LanguagesResponse)
def languages() -> LanguagesResponse:
    """Languages a query can be written in, and the translator's state."""
    translator = get_translator()
    state, detail = translator.status()
    return LanguagesResponse(
        languages=[
            LanguageInfo(code=lang.code, name=lang.name, native_name=lang.native_name, rtl=lang.rtl)
            for lang in LANGUAGES.values()
        ],
        engine=translator.engine,
        status=state,  # type: ignore[arg-type]
        detail=detail,
    )


@app.post(
    "/api/report",
    response_class=Response,
    responses={200: {"content": {"application/pdf": {}}, "description": "The report as a PDF."}},
    summary="Export an analysis result as a PDF report",
)
async def report(request: ReportRequest) -> Response:
    """Typeset a result the client already holds into a downloadable PDF."""
    try:
        pdf = await asyncio.to_thread(render_report, request)
    except ReportError as exc:
        logger.warning("Report %s failed: %s", request.result.request_id, exc)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    filename = f"satquery-report-{request.result.request_id[:8]}.pdf"
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post(
    "/api/analyse",
    response_model=AnalysisResponse,
    summary="Analyse one image, or a cross-modal / bi-temporal pair, against a query",
)
async def analyse(
    prompt: str = Form(..., description="Natural-language query about the imagery."),
    images: list[UploadFile] = File(
        ...,
        description=(
            "One georeferenced GeoTIFF, or two for a co-registered optical+SAR pair "
            "or a bi-temporal pair."
        ),
    ),
    modalities: str | None = Form(
        None,
        description=(
            "Optional comma-separated modality hints, one per image, e.g. 'optical,sar'. "
            "Overrides automatic detection."
        ),
    ),
    task: str | None = Form(
        None,
        description=(
            "Optional task override (vqa, caption, grounding, change_vqa, "
            "change_description, fusion). Omit to let the controller route."
        ),
    ),
    language: str | None = Form(
        None,
        description=(
            "Optional language code of the user (e.g. 'hi', 'ta'); see /api/languages. "
            "Disambiguates languages sharing a script and sets the answer language. "
            "Non-English queries are translated to English before analysis."
        ),
    ),
) -> AnalysisResponse:
    query = prompt.strip()
    if not query:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="A query is required.",
        )

    if not 1 <= len(images) <= 2:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Expected 1 or 2 images, received {len(images)}. Supported inputs are a "
                "single image, a co-registered optical+SAR pair, or a bi-temporal pair."
            ),
        )

    if not settings.is_configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OPENROUTER_API_KEY is not set on the server. Add it to backend/.env.",
        )

    hints = [part.strip() for part in modalities.split(",")] if modalities else []

    prepared: list[PreparedImage] = []
    try:
        for index, upload in enumerate(images):
            raw = await upload.read()
            await upload.close()
            prepared.append(
                prepare_upload(
                    index=index,
                    filename=upload.filename or f"image_{index}",
                    content_type=upload.content_type,
                    raw=raw,
                    declared_modality=hints[index] if index < len(hints) else None,
                    max_bytes=settings.max_image_bytes,
                    max_edge_px=settings.max_image_edge_px,
                )
            )
    except ImageValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    if language is not None and language not in LANGUAGES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported language '{language}'. See /api/languages.",
        )

    english_query, query_step = await _translate_query(query, language)

    controller = AgenticController(settings, OpenRouterClient(settings))
    try:
        response = await controller.run(
            query=english_query, images=prepared, task_override=task
        )
    except ImageValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except ControllerError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except OpenRouterError as exc:
        logger.exception("Upstream model call failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Upstream model call failed: {exc}",
        ) from exc

    if query_step is not None or (language or ENGLISH) != ENGLISH:
        await _localise_response(response, query, english_query, language, query_step)
    return response


async def _translate_query(query: str, hint: str | None) -> tuple[str, TraceStep | None]:
    """Translate a non-English query into English for the English-only models."""
    source = resolve_source_language(query, hint)
    if source.language.code == ENGLISH:
        return query, None

    started = time.perf_counter()
    try:
        (english,) = await asyncio.to_thread(
            get_translator().to_english, [query], source.flores
        )
    except TranslationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"The query is in {source.language.name} but could not be translated: {exc}",
        ) from exc

    step = TraceStep(
        stage="translate",
        tool="query-translator",
        model=get_translator().engine,
        params={"source": source.flores, "target": "eng_Latn", "detected": source.detected},
        detail=f"{source.language.name} → English: {english}",
        duration_ms=(time.perf_counter() - started) * 1000,
    )
    return english, step


async def _localise_response(
    response: AnalysisResponse,
    original_query: str,
    english_query: str,
    hint: str | None,
    query_step: TraceStep | None,
) -> None:
    """Attach the answer in the user's language, keeping the English original.

    The answer goes back in the language the user picked; if they left the UI in
    English but typed another language, it goes back in the query's language.
    A failed back-translation is only a warning — the English answer still stands.
    """
    source = resolve_source_language(original_query, hint)
    target = LANGUAGES[hint] if hint and hint != ENGLISH else source.language

    info = TranslationInfo(
        engine=get_translator().engine,
        source_language=source.language.code,
        source_detected=source.detected,
        target_language=target.code,
        original_query=original_query,
        english_query=english_query,
    )

    added_ms = 0.0
    if query_step is not None:
        response.trace.steps.insert(0, query_step)
        added_ms += query_step.duration_ms

    # Per-image notes repeat once per image ("GeoTIFF: georeferencing tags present").
    response.trace.warnings = list(dict.fromkeys(response.trace.warnings))

    if target.code != ENGLISH:
        started = time.perf_counter()
        # Everything the user reads, translated in one batch. Duplicate strings
        # (labels shared by several boxes, say) are translated once.
        labels = [item.label for item in response.evidence]
        rationale = response.trace.routing_rationale
        texts = list(
            dict.fromkeys(
                [
                    response.answer,
                    *(item.description for item in response.evidence),
                    *(label for label in labels if label),
                    *response.trace.warnings,
                    *([rationale] if rationale else []),
                ]
            )
        )
        try:
            translated = dict(
                zip(
                    texts,
                    await asyncio.to_thread(get_translator().from_english, texts, target.flores),
                    strict=True,
                )
            )
        except TranslationError as exc:
            response.trace.warnings.append(
                f"Answer could not be translated to {target.name}; showing English. ({exc})"
            )
        else:
            info.answer = translated[response.answer]
            info.evidence_descriptions = [translated[item.description] for item in response.evidence]
            info.evidence_labels = [translated[label] if label else None for label in labels]
            info.warnings = [translated[warning] for warning in response.trace.warnings]
            info.routing_rationale = translated[rationale] if rationale else None
            duration = (time.perf_counter() - started) * 1000
            added_ms += duration
            response.trace.steps.append(
                TraceStep(
                    stage="translate",
                    tool="answer-translator",
                    model=info.engine,
                    params={"source": "eng_Latn", "target": target.flores},
                    detail=f"English → {target.name} ({len(texts)} texts)",
                    duration_ms=duration,
                )
            )

    response.trace.total_duration_ms += added_ms
    response.execution_time_ms += added_ms
    response.translation = info
