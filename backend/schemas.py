"""API-level schemas for the SatQuery AI backend.

These mirror the specialist-model contract in ``ml/controller/schema.py``
(``Evidence`` / ``ModelResponse``) so a local fine-tuned specialist and the
OpenRouter baseline can be serialised through exactly the same response shape.
The extra fields here (`inputs`, `trace`, `usage`) carry the auditable
execution summary the problem statement requires.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Task = Literal[
    "vqa",
    "caption",
    "grounding",
    "change_vqa",
    "change_description",
    "fusion",
]

Modality = Literal["optical", "sar", "unknown"]

InputConfiguration = Literal["single", "cross_modal_pair", "bi_temporal_pair"]


class Evidence(BaseModel):
    """Visual or numerical evidence backing an answer.

    Mirrors ``ml.controller.schema.Evidence``.
    """

    type: Literal["bbox", "mask", "heatmap", "observation"] = "observation"
    # bbox: a BoundingBox dict; mask: {"png": data URI, opaque where changed}.
    data: Any = None
    description: str

    label: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    image_index: int = Field(
        default=0,
        description="Which uploaded image this evidence refers to (0-based).",
    )


class BoundingBox(BaseModel):
    """Normalised box in [0, 1] relative to image width/height.

    Normalised rather than pixel coordinates so the frontend can overlay it on
    a resized preview without knowing the original raster size.
    """

    x_min: float = Field(ge=0.0, le=1.0)
    y_min: float = Field(ge=0.0, le=1.0)
    x_max: float = Field(ge=0.0, le=1.0)
    y_max: float = Field(ge=0.0, le=1.0)


class ImageInfo(BaseModel):
    """Result of input validation for one uploaded image."""

    index: int
    filename: str
    content_type: str | None = None
    detected_format: str
    modality: Modality
    width: int
    height: int
    size_bytes: int
    is_georeferenced: bool = False
    ground_sample_distance_m: float | None = Field(
        default=None,
        description="Metres per pixel of the upload, from its GeoTIFF tags; null when unknown.",
    )
    notes: list[str] = Field(default_factory=list)
    preview_data_uri: str | None = Field(
        default=None,
        description=(
            "Browser-renderable JPEG of exactly what the model was shown. Needed "
            "because browsers cannot display GeoTIFF, and it guarantees evidence "
            "boxes are overlaid on the same raster the model saw."
        ),
    )


class TraceStep(BaseModel):
    """One observable step of the agentic execution trace."""

    stage: Literal[
        "translate", "validate", "classify", "select", "execute", "narrate", "aggregate"
    ]
    tool: str
    model: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    detail: str = ""
    duration_ms: float = 0.0


class ExecutionTrace(BaseModel):
    """Auditable summary of what the controller selected and ran."""

    input_configuration: InputConfiguration
    task: Task
    task_source: Literal["auto", "override"] = "auto"
    routing_rationale: str = ""
    selected_tools: list[str] = Field(default_factory=list)
    steps: list[TraceStep] = Field(default_factory=list)
    total_duration_ms: float = 0.0
    domain_adapted: bool = Field(
        default=False,
        description=(
            "True only when a remote-sensing fine-tuned specialist produced the "
            "answer. False means the generic OpenRouter VLM baseline was used."
        ),
    )
    warnings: list[str] = Field(default_factory=list)


class Usage(BaseModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cost: float | None = None


class TranslationInfo(BaseModel):
    """What the translation layer did around the English-only models.

    The top-level ``answer`` and ``evidence`` always stay in English, exactly as
    the model produced them; the localised copies live here.
    """

    engine: str
    source_language: str = Field(description="Language code the query was read as.")
    source_detected: bool = Field(
        description="True when inferred from the script rather than the client's hint."
    )
    target_language: str = Field(description="Language code the answer was translated to.")
    original_query: str
    english_query: str
    answer: str | None = Field(
        default=None,
        description="Answer in the target language; null if back-translation failed.",
    )
    evidence_descriptions: list[str] | None = Field(
        default=None,
        description="Evidence descriptions in the target language, index-aligned with evidence.",
    )
    evidence_labels: list[str | None] | None = Field(
        default=None,
        description="Evidence labels in the target language, index-aligned with evidence.",
    )
    warnings: list[str] | None = Field(
        default=None,
        description="Trace warnings in the target language, index-aligned with trace.warnings.",
    )
    routing_rationale: str | None = Field(
        default=None,
        description="The router's rationale in the target language.",
    )


class Narration(BaseModel):
    """Set when a VLM reworded a specialist's result in plain language.

    The wording is the VLM's; every region and figure in it is the specialist's,
    checked before the text is accepted (agent/narration.py).
    """

    model: str = Field(description="The VLM that wrote the plain-language text.")
    specialist_answer: str = Field(description="The specialist's own answer, as it produced it.")
    regions_described: int = 0


class AnalysisResponse(BaseModel):
    """Evidence-grounded response returned to the client."""

    request_id: str
    task: Task
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[Evidence] = Field(default_factory=list)
    model_name: str = ""
    execution_time_ms: float = 0.0
    inputs: list[ImageInfo] = Field(default_factory=list)
    trace: ExecutionTrace
    usage: Usage | None = None
    translation: TranslationInfo | None = None
    narration: Narration | None = None


class ReportLabels(BaseModel):
    """Every string the PDF report prints, already localised by the client.

    The web app owns the UI translations, so it sends the exact wording it shows
    on screen; the report then reads the same as the page it was exported from.
    ``image`` and ``region`` contain an ``{{index}}`` placeholder.
    """

    title: str
    generated: str
    request: str
    question: str
    query_translated: str
    answer: str
    english_original: str
    task: str
    task_name: str
    input: str
    configuration_name: str
    confidence: str
    time: str
    tokens: str
    language: str
    language_name: str
    images: str
    image: str
    modality: dict[Modality, str]
    georeferenced: str
    evidence: str
    region: str
    no_spatial: str | None = None
    controller_notes: str
    trace: str
    routing: str
    tools: str


class ReportRequest(BaseModel):
    """Body of ``POST /api/report``: a result the client already holds."""

    result: AnalysisResponse
    query: str = Field(min_length=1, description="The question exactly as the user typed it.")
    language: str = Field(
        default="en",
        description="Language the report is written in; 'en' renders the English original only.",
    )
    labels: ReportLabels


class ToolInfo(BaseModel):
    """One entry of the specialist-model registry."""

    name: str
    task: Task
    description: str
    backend: Literal["openrouter", "local_specialist"]
    model: str | None = None
    available: bool
    domain_adapted: bool


class RegistryResponse(BaseModel):
    tools: list[ToolInfo]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    openrouter_configured: bool
    vision_model: str
    router_model: str


class LanguageInfo(BaseModel):
    code: str
    name: str
    native_name: str
    rtl: bool = False


class LanguagesResponse(BaseModel):
    languages: list[LanguageInfo]
    engine: str
    status: Literal["disabled", "unavailable", "idle", "ready"]
    detail: str | None = None


class ErrorResponse(BaseModel):
    detail: str
