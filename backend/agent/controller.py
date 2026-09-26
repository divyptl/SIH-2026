"""The agentic controller.

Implements the orchestration loop the problem statement specifies:

1. check the number, modality, format, and compatibility of the input images;
2. interpret the query and classify the requested task;
3. select one or more tools from the registry;
4. configure only permitted parameters and execute the workflow;
5. combine textual and spatial outputs and estimate confidence;
6. emit an auditable execution trace.

Internal planning is not part of the contract -- only the observable trace is.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

from agent.prompts import (
    ROUTER_SYSTEM_PROMPT,
    analysis_system_prompt,
    build_user_message,
)
from agent.registry import (
    DEFAULT_PAIR_TASK,
    DEFAULT_SINGLE_TASK,
    TOOL_REGISTRY,
    ToolEntry,
)
from config import Settings
from schemas import (
    AnalysisResponse,
    Evidence,
    ExecutionTrace,
    InputConfiguration,
    Task,
    TraceStep,
    Usage,
)
from services.images import PreparedImage, check_pair_compatibility
from services.openrouter_client import (
    OpenRouterClient,
    OpenRouterError,
    parse_json_object,
)

logger = logging.getLogger("satquery.controller")

# A specialist accepts imagery up to this factor finer or coarser than the
# range it was trained on; training applies resolution augmentation of up to 4x.
GSD_TOLERANCE = 2.0

VALID_TASKS: set[str] = set(TOOL_REGISTRY)


class ControllerError(RuntimeError):
    """Raised when the workflow cannot be completed."""


class _Timer:
    """Wall-clock timer for one trace step."""

    def __init__(self) -> None:
        self._start = time.perf_counter()

    def ms(self) -> float:
        return round((time.perf_counter() - self._start) * 1000, 2)


def classify_input_configuration(images: list[PreparedImage]) -> InputConfiguration:
    """Decide the input configuration from the uploads alone.

    Deterministic on purpose: the number and modality of images is a fact about
    the request, so it is never delegated to the model.
    """
    if len(images) == 1:
        return "single"
    modalities = {image.info.modality for image in images}
    if modalities == {"optical", "sar"}:
        return "cross_modal_pair"
    return "bi_temporal_pair"


def _fallback_task(configuration: InputConfiguration) -> Task:
    if configuration == "single":
        return DEFAULT_SINGLE_TASK
    if configuration == "cross_modal_pair":
        return "fusion"
    return DEFAULT_PAIR_TASK


def _image_summary(image: PreparedImage) -> str:
    info = image.info
    bits = [
        f"{info.filename}",
        f"modality={info.modality}",
        f"format={info.detected_format}",
        f"size={info.width}x{info.height}",
    ]
    if info.is_georeferenced:
        bits.append("georeferenced")
    return ", ".join(bits)


def _coerce_confidence(raw: Any) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.5
    if value > 1.0:
        # Some models answer on a 0-100 scale despite the instruction.
        value = value / 100.0
    return max(0.0, min(1.0, value))


def _coerce_evidence(raw: Any, image_count: int) -> list[Evidence]:
    """Normalise the model's evidence list, dropping anything malformed.

    Bad boxes are discarded rather than clamped: a box the model could not
    express correctly is not evidence, and silently reshaping it would make
    the overlay lie about where the model actually looked.
    """
    if not isinstance(raw, list):
        return []

    evidence: list[Evidence] = []
    for item in raw:
        if not isinstance(item, dict):
            continue

        description = str(item.get("description") or item.get("label") or "").strip()
        if not description:
            continue

        index = item.get("image_index", 0)
        try:
            image_index = int(index)
        except (TypeError, ValueError):
            image_index = 0
        if not 0 <= image_index < image_count:
            image_index = 0

        box = item.get("box")
        data: Any = None
        kind = "observation"
        if isinstance(box, dict):
            try:
                coords = {key: float(box[key]) for key in ("x_min", "y_min", "x_max", "y_max")}
            except (KeyError, TypeError, ValueError):
                coords = None
            if (
                coords
                and all(0.0 <= value <= 1.0 for value in coords.values())
                and coords["x_max"] > coords["x_min"]
                and coords["y_max"] > coords["y_min"]
            ):
                data = coords
                kind = "bbox"

        # A change mask from a specialist, as a PNG data URI (never from the VLM).
        mask = item.get("mask")
        if kind == "observation" and isinstance(mask, str) and mask.startswith("data:image/png;base64,"):
            data = {"png": mask}
            kind = "mask"

        label = item.get("label")
        confidence = item.get("confidence")
        evidence.append(
            Evidence(
                type=kind,
                data=data,
                description=description,
                label=str(label).strip() if label else None,
                confidence=(
                    _coerce_confidence(confidence) if confidence is not None else None
                ),
                image_index=image_index,
            )
        )
    return evidence


class AgenticController:
    """Routes a query and its imagery through the appropriate workflow."""

    def __init__(self, settings: Settings, client: OpenRouterClient) -> None:
        self._settings = settings
        self._client = client

    async def run(
        self,
        *,
        query: str,
        images: list[PreparedImage],
        task_override: str | None = None,
    ) -> AnalysisResponse:
        overall = _Timer()
        steps: list[TraceStep] = []
        warnings: list[str] = []

        # --- 1. Validate inputs and their compatibility ----------------------
        step_timer = _Timer()
        configuration = classify_input_configuration(images)
        warnings.extend(check_pair_compatibility(images))
        for image in images:
            warnings.extend(image.info.notes)
        steps.append(
            TraceStep(
                stage="validate",
                tool="input-validator",
                params={
                    "image_count": len(images),
                    "modalities": [image.info.modality for image in images],
                    "formats": [image.info.detected_format for image in images],
                },
                detail=f"Resolved input configuration: {configuration}.",
                duration_ms=step_timer.ms(),
            )
        )

        # --- 2. Classify the requested task ----------------------------------
        step_timer = _Timer()
        if task_override:
            if task_override not in VALID_TASKS:
                raise ControllerError(
                    f"Unknown task '{task_override}'. Valid tasks: {sorted(VALID_TASKS)}."
                )
            task: Task = task_override  # type: ignore[assignment]
            task_source: str = "override"
            rationale = "Task explicitly supplied by the client."
            router_model = None
        else:
            task, rationale = await self._classify(query, images, configuration)
            task_source = "auto"
            router_model = self._settings.router_model

        entry = TOOL_REGISTRY[task]
        if configuration not in entry.accepts:
            corrected = _fallback_task(configuration)
            warnings.append(
                f"Task '{task}' cannot run on a {configuration} input; "
                f"fell back to '{corrected}'."
            )
            task = corrected
            entry = TOOL_REGISTRY[task]

        steps.append(
            TraceStep(
                stage="classify",
                tool="query-router",
                model=router_model,
                params={"input_configuration": configuration},
                detail=f"Task '{task}' ({task_source}). {rationale}",
                duration_ms=step_timer.ms(),
            )
        )

        # --- 3. Select the tool ----------------------------------------------
        step_timer = _Timer()
        use_specialist = entry.specialist_available
        if use_specialist:
            use_specialist, note = await asyncio.to_thread(self._check_resolution, entry, images)
            if note:
                warnings.append(note)
        tool_name, backend, model_name, domain_adapted = (
            self._select(entry) if use_specialist else self._baseline(entry)
        )
        steps.append(
            TraceStep(
                stage="select",
                tool="tool-selector",
                model=model_name,
                params={"backend": backend, "permitted_params": list(entry.permitted_params)},
                detail=f"Selected '{tool_name}' for task '{task}'.",
                duration_ms=step_timer.ms(),
            )
        )

        # --- 4. Execute -------------------------------------------------------
        step_timer = _Timer()
        payload: dict[str, Any] | None = None
        specialist_payload: dict[str, Any] | None = None
        usage: dict[str, Any] = {}
        params: dict[str, Any]
        if domain_adapted and entry.runner is not None:
            params = {"checkpoint": entry.checkpoint}
            try:
                payload = await asyncio.to_thread(entry.runner, query, images)
            except Exception as exc:
                # A broken specialist must not take the request down with it; the
                # trace says plainly that the baseline answered instead.
                logger.exception("Specialist '%s' failed; using the baseline", tool_name)
                warnings.append(
                    f"The fine-tuned specialist for '{task}' failed ({exc}); "
                    "the generic VLM baseline answered instead."
                )
                tool_name, backend, model_name, domain_adapted = self._baseline(entry)

            # Hybrid execution: the specialist returned evidence but no answer.
            # Save the specialist payload and fall through to the VLM baseline
            # which will receive the specialist's context in its prompt.
            if payload is not None and not payload.get("answer"):
                specialist_payload = payload
                payload = None  # trigger VLM baseline below

        if payload is None:
            params = {"temperature": 0.2, "max_tokens": 1400}
            specialist_context = (
                specialist_payload.get("specialist_context")
                if specialist_payload
                else None
            )
            payload, usage = await self._execute_baseline(
                task=task,
                query=query,
                images=images,
                params=params,
                specialist_context=specialist_context,
            )
            # Merge specialist evidence (terrain probs, similarity) ahead of
            # any evidence the VLM itself produced, so the user sees both.
            if specialist_payload is not None:
                sp_evidence = specialist_payload.get("evidence", [])
                payload["evidence"] = sp_evidence + (payload.get("evidence") or [])
                # Prefer the specialist's confidence when the VLM omits one.
                if payload.get("confidence") is None and specialist_payload.get("confidence") is not None:
                    payload["confidence"] = specialist_payload["confidence"]

        steps.append(
            TraceStep(
                stage="execute",
                tool=tool_name,
                model=model_name,
                params={"backend": backend, **params},
                detail=f"Ran {task} over {len(images)} image(s).",
                duration_ms=step_timer.ms(),
            )
        )

        # --- 5. Aggregate outputs and confidence ------------------------------
        step_timer = _Timer()
        answer = str(payload.get("answer") or "").strip()
        if not answer:
            raise ControllerError("The model returned an empty answer.")

        # The model's own score, as reported; only a missing one is filled in.
        if payload.get("confidence") is None:
            warnings.append("The model did not report a confidence score; 0.5 is shown.")
        confidence = _coerce_confidence(payload.get("confidence"))
        evidence = _coerce_evidence(payload.get("evidence"), len(images))

        boxes = sum(1 for item in evidence if item.type == "bbox")
        steps.append(
            TraceStep(
                stage="aggregate",
                tool="response-aggregator",
                params={"evidence_items": len(evidence), "bounding_boxes": boxes},
                detail=f"Combined answer with {boxes} spatial evidence region(s).",
                duration_ms=step_timer.ms(),
            )
        )

        total_ms = overall.ms()
        return AnalysisResponse(
            request_id=uuid.uuid4().hex[:12],
            task=task,
            answer=answer,
            confidence=confidence,
            evidence=evidence,
            model_name=model_name,
            execution_time_ms=total_ms,
            inputs=[image.info for image in images],
            trace=ExecutionTrace(
                input_configuration=configuration,
                task=task,
                task_source=task_source,  # type: ignore[arg-type]
                routing_rationale=rationale,
                selected_tools=[tool_name],
                steps=steps,
                total_duration_ms=total_ms,
                domain_adapted=domain_adapted,
                warnings=warnings,
            ),
            usage=Usage(**usage) if usage else None,
        )

    async def _classify(
        self,
        query: str,
        images: list[PreparedImage],
        configuration: InputConfiguration,
    ) -> tuple[Task, str]:
        """Ask the router model to pick a task, with a rule-based fallback."""
        summaries = [_image_summary(image) for image in images]
        user_text = (
            f"Input configuration: {configuration}\n"
            f"Image count: {len(images)}\n"
            "Images:\n" + "\n".join(f"  [{i}] {s}" for i, s in enumerate(summaries)) + "\n\n"
            f"Query: {query}"
        )

        try:
            text, _ = await self._client.complete(
                model=self._settings.router_model,
                system_prompt=ROUTER_SYSTEM_PROMPT,
                user_text=user_text,
                json_object=True,
                temperature=0.0,
                max_tokens=200,
            )
            payload = parse_json_object(text)
        except OpenRouterError as exc:
            fallback = _fallback_task(configuration)
            return fallback, f"Routing model unavailable ({exc}); used rule-based default."

        task = payload.get("task")
        if task not in VALID_TASKS:
            fallback = _fallback_task(configuration)
            return fallback, f"Router returned unknown task '{task}'; used rule-based default."

        rationale = str(payload.get("rationale") or "").strip()
        return task, rationale  # type: ignore[return-value]

    def _select(self, entry: ToolEntry) -> tuple[str, str, str, bool]:
        """Choose between the fine-tuned specialist and the baseline."""
        if entry.specialist_available:
            return entry.name, "local_specialist", entry.specialist_model, True
        return self._baseline(entry)

    @staticmethod
    def _check_resolution(entry: ToolEntry, images: list[PreparedImage]) -> tuple[bool, str | None]:
        """Whether the specialist was trained on imagery at this resolution.

        Returns (use the specialist, note for the trace). A model shown imagery
        far outside its training resolution answers confidently and wrongly, so
        such inputs go to the baseline. Uploads without georeferencing have no
        known resolution; they still reach the specialist, with a caveat.
        """
        if entry.gsd_range is None:
            return True, None
        try:
            low, high = entry.gsd_range()
        except Exception:
            # Loading failed; the execute step reports it and falls back.
            logger.exception("Could not read the training resolution of '%s'", entry.name)
            return True, None
        trained = f"{low:g} m" if low == high else f"{low:g}-{high:g} m"

        known = [image.analysis_gsd_m for image in images if image.analysis_gsd_m is not None]
        if not known:
            return True, (
                f"The input resolution is unknown (no georeferencing); the fine-tuned "
                f"'{entry.task}' model is trained on {trained}/pixel imagery, so check "
                "its answer against the images."
            )
        gsd = max(known)
        if low / GSD_TOLERANCE <= gsd <= high * GSD_TOLERANCE:
            return True, None
        return False, (
            f"The fine-tuned '{entry.task}' model is trained on {trained}/pixel imagery; "
            f"this input is analysed at {gsd:.1f} m/pixel, so the general VLM answered instead."
        )

    def _baseline(self, entry: ToolEntry) -> tuple[str, str, str, bool]:
        return f"{entry.name}-baseline", "openrouter", self._settings.vision_model, False

    async def _execute_baseline(
        self,
        *,
        task: Task,
        query: str,
        images: list[PreparedImage],
        params: dict[str, Any],
        specialist_context: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run the task on the generic OpenRouter vision-language model.

        When ``specialist_context`` is provided (hybrid execution), the domain-
        adapted evidence is prepended to the user message so the VLM can
        incorporate it when answering.
        """
        user_text = build_user_message(
            query=query,
            image_summaries=[_image_summary(image) for image in images],
            task=task,
            specialist_context=specialist_context,
        )
        text, usage = await self._client.complete(
            model=self._settings.vision_model,
            system_prompt=analysis_system_prompt(task),
            user_text=user_text,
            image_data_uris=[image.data_uri for image in images],
            json_object=True,
            temperature=params["temperature"],
            max_tokens=params["max_tokens"],
        )
        return parse_json_object(text), usage
