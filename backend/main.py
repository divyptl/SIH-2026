"""SatQuery AI backend.

FastAPI service exposing the agentic remote-sensing analysis workflow.

Run with:  uv run fastapi dev main.py
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from agent.controller import AgenticController, ControllerError
from agent.registry import describe_registry
from config import get_settings
from schemas import (
    AnalysisResponse,
    HealthResponse,
    RegistryResponse,
)
from services.images import (
    ImageValidationError,
    PreparedImage,
    prepare_upload,
)
from services.openrouter_client import OpenRouterClient, OpenRouterError

logger = logging.getLogger("satquery.api")

settings = get_settings()

app = FastAPI(
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
            "One image, or two for a co-registered optical+SAR pair or a bi-temporal pair."
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

    controller = AgenticController(settings, OpenRouterClient(settings))
    try:
        return await controller.run(query=query, images=prepared, task_override=task)
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
