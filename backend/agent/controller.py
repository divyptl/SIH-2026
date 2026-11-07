from __future__ import annotations

import asyncio
import base64
import logging
import re
import time
import uuid
from typing import Any

from agent.narration import NarrationRejected, narrate
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
    Narration,
    Task,
    TraceStep,
    Usage,
)
from services.images import PreparedImage, check_pair_compatibility
from services.openrouter_client import OpenRouterClient, OpenRouterError

logger = logging.getLogger("satquery.controller")

GSD_TOLERANCE = 2.0
VALID_TASKS: set[str] = set(TOOL_REGISTRY)

_GROUNDING_WORDS = re.compile(
    r"\b(where|locate|locating|find|finding|show|highlight|point out|mark|detect|"
    r"box|which part|position of|location of)\b",
    re.IGNORECASE,
)
_PHYSICS_KEYWORDS = ("water", "flood", "built", "vegetation")


class ControllerError(RuntimeError):
    pass


class _Timer:
    def __init__(self) -> None:
        self._start = time.perf_counter()

    def ms(self) -> float:
        return round((time.perf_counter() - self._start) * 1000, 2)


def classify_input_configuration(images: list[PreparedImage]) -> InputConfiguration:
    if len(images) == 1:
        return "single"
    if {img.info.modality for img in images} == {"optical", "sar"}:
        return "cross_modal_pair"
    return "bi_temporal_pair"


def _fallback_task(configuration: InputConfiguration) -> Task:
    if configuration == "single":
        return DEFAULT_SINGLE_TASK
    return "fusion" if configuration == "cross_modal_pair" else DEFAULT_PAIR_TASK


def _coerce_confidence(raw: Any) -> float:
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, val / 100.0 if val > 1.0 else val))


def _coerce_evidence(raw: Any, image_count: int) -> list[Evidence]:
    if not isinstance(raw, list):
        return []

    evidence: list[Evidence] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        description = str(item.get("description") or item.get("label") or "").strip()
        if not description:
            continue

        try:
            image_index = int(item.get("image_index", 0))
            if not 0 <= image_index < image_count:
                image_index = 0
        except (TypeError, ValueError):
            image_index = 0

        data: Any = None
        kind = "observation"
        box = item.get("box")
        if isinstance(box, dict):
            try:
                coords = {k: float(box[k]) for k in ("x_min", "y_min", "x_max", "y_max")}
                if (
                    all(0.0 <= v <= 1.0 for v in coords.values())
                    and coords["x_max"] > coords["x_min"]
                    and coords["y_max"] > coords["y_min"]
                ):
                    data, kind = coords, "bbox"
            except (KeyError, TypeError, ValueError):
                pass

        mask = item.get("mask")
        if kind == "observation" and isinstance(mask, str) and mask.startswith("data:image/png;base64,"):
            data, kind = {"png": mask}, "mask"

        label = item.get("label")
        conf = item.get("confidence")
        evidence.append(
            Evidence(
                type=kind,
                data=data,
                description=description,
                label=str(label).strip() if label else None,
                confidence=_coerce_confidence(conf) if conf is not None else None,
                image_index=image_index,
            )
        )
    return evidence


class AgenticController:
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

        # 1. Validate inputs
        step_timer = _Timer()
        configuration = classify_input_configuration(images)
        warnings.extend(check_pair_compatibility(images))
        for img in images:
            warnings.extend(img.info.notes)
        steps.append(
            TraceStep(
                stage="validate",
                tool="input-validator",
                params={
                    "image_count": len(images),
                    "modalities": [img.info.modality for img in images],
                    "formats": [img.info.detected_format for img in images],
                },
                detail=f"Resolved input configuration: {configuration}.",
                duration_ms=step_timer.ms(),
            )
        )

        # 2. Classify task
        step_timer = _Timer()
        if task_override:
            if task_override not in VALID_TASKS:
                raise ControllerError(f"Unknown task '{task_override}'. Valid tasks: {sorted(VALID_TASKS)}.")
            task: Task = task_override  # type: ignore[assignment]
            task_source, rationale = "override", "Task explicitly supplied by the client."
        else:
            task, rationale = await self._classify(query, images, configuration)
            task_source = "auto"

        entry = TOOL_REGISTRY[task]
        if configuration not in entry.accepts:
            corrected = _fallback_task(configuration)
            warnings.append(f"Task '{task}' cannot run on a {configuration} input; fell back to '{corrected}'.")
            task, entry = corrected, TOOL_REGISTRY[corrected]

        steps.append(
            TraceStep(
                stage="classify",
                tool="query-router",
                model=None,
                params={"input_configuration": configuration},
                detail=f"Task '{task}' ({task_source}). {rationale}",
                duration_ms=step_timer.ms(),
            )
        )

        # 3. Select tool
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

        # 4. Execute
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
                logger.exception("Specialist '%s' failed; using the baseline", tool_name)
                warnings.append(
                    f"The fine-tuned specialist for '{task}' failed ({exc}); "
                    "the generic VLM baseline answered instead."
                )
                tool_name, backend, model_name, domain_adapted = self._baseline(entry)

            if payload is not None and not payload.get("answer"):
                specialist_payload, payload = payload, None

        if payload is None:
            params = {"temperature": 0.2, "max_tokens": 1400}
            payload, usage = await self._execute_baseline(
                task=task,
                query=query,
                images=images,
                params=params,
                specialist_context=specialist_payload.get("specialist_context") if specialist_payload else None,
            )
            if specialist_payload is not None:
                payload["evidence"] = specialist_payload.get("evidence", []) + (payload.get("evidence") or [])
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

        # 5. Plain-language narration
        narration: Narration | None = None
        facts = payload.pop("narration_facts", None)
        if facts and facts["regions"] and self._settings.narration_enabled:
            narration, narration_usage = await self._narrate(
                query=query, images=images, payload=payload, facts=facts, steps=steps, warnings=warnings
            )
            if narration_usage and not usage:
                usage = narration_usage

        # 6. Aggregate outputs and confidence
        step_timer = _Timer()
        answer = str(payload.get("answer") or "").strip()
        if not answer:
            raise ControllerError("The model returned an empty answer.")

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

       # 7. Physics Verification Intercept
        step_timer = _Timer()
        query_lower = query.lower()
        if any(k in query_lower for k in _PHYSICS_KEYWORDS):
            try:
                from services.physics_service import process_images_physics, run_verification

                physics_context = await asyncio.to_thread(process_images_physics, images, query)
                h, w = physics_context["primary_shape"]

                first_box = next((ev.data for ev in evidence if ev.type == "bbox" and ev.data), None)
                vlm_bbox = (
                    (
                        int(first_box["x_min"] * w),
                        int(first_box["y_min"] * h),
                        int(first_box["x_max"] * w),
                        int(first_box["y_max"] * h),
                    )
                    if first_box
                    else None
                )

                ver_result = await asyncio.to_thread(
                    run_verification,
                    query=query,
                    vlm_confidence=confidence,
                    vlm_bbox=vlm_bbox,
                    physics_context=physics_context,
                    is_specialist=domain_adapted,
                )
                if ver_result:
                    confidence = ver_result.verified_confidence
                    warnings.append(f"Physics Verifier [{ver_result.status}]: {ver_result.audit_explanation}")
                    steps.append(
                        TraceStep(
                            stage="verification",
                            tool="physics-verifier",
                            params={
                                "claim": ver_result.claim_type,
                                "modality_used": ver_result.modality_used,
                                **ver_result.physical_metrics,
                            },
                            detail=ver_result.audit_explanation,
                            duration_ms=step_timer.ms(),
                        )
                    )
            except Exception as exc:
                logger.error("Physics verification pipeline bypassed or failed: %s", exc)

        total_ms = overall.ms()
        return AnalysisResponse(
            request_id=uuid.uuid4().hex[:12],
            task=task,
            answer=answer,
            confidence=confidence,
            evidence=evidence,
            model_name=model_name,
            execution_time_ms=total_ms,
            inputs=[img.info for img in images],
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
            narration=narration,
        )

    async def _narrate(
        self,
        *,
        query: str,
        images: list[PreparedImage],
        payload: dict[str, Any],
        facts: dict[str, Any],
        steps: list[TraceStep],
        warnings: list[str],
    ) -> tuple[Narration | None, dict[str, Any]]:
        step_timer = _Timer()
        model = self._settings.narration_model
        specialist_answer = str(payload.get("answer") or "")
        try:
            result = await narrate(self._client, model=model, query=query, images=images, facts=facts)
        except (NarrationRejected, OpenRouterError) as exc:
            reason = str(exc) if isinstance(exc, NarrationRejected) else "the language model could not be reached"
            warnings.append(
                f"The plain-language summary was discarded because {reason}; "
                "the fine-tuned model's own wording is shown instead."
            )
            steps.append(
                TraceStep(
                    stage="narrate",
                    tool="plain-language-narrator",
                    model=model,
                    params={"regions": len(facts["regions"])},
                    detail=f"Narration discarded: {exc}",
                    duration_ms=step_timer.ms(),
                )
            )
            return None, {}

        payload["answer"] = result.summary
        by_number = {r["number"]: r for r in facts["regions"]}
        for number, text in result.regions.items():
            region = by_number[number]
            payload["evidence"][region["evidence_index"]]["description"] = f"{text} {region['measure']}"

        steps.append(
            TraceStep(
                stage="narrate",
                tool="plain-language-narrator",
                model=model,
                params={"regions": len(facts["regions"]), "temperature": 0.2},
                detail=(
                    f'Reworded the specialist\'s answer ("{specialist_answer}") and '
                    f"{len(result.regions)} region description(s); every region, figure "
                    "and direction was checked against the specialist's output."
                ),
                duration_ms=step_timer.ms(),
            )
        )
        return (
            Narration(
                model=model,
                specialist_answer=specialist_answer,
                regions_described=len(result.regions),
            ),
            result.usage,
        )

    async def _classify(
        self,
        query: str,
        images: list[PreparedImage],
        configuration: InputConfiguration,
    ) -> tuple[Task, str]:
        if configuration == "single" and (match := _GROUNDING_WORDS.search(query)):
            return "grounding", (
                f'Rule-based: one image, and the query asks to locate something '
                f'("{match.group(0).lower()}"), so \'grounding\'.'
            )
        task = _fallback_task(configuration)
        return task, f"Rule-based: a {configuration.replace('_', ' ')} input maps to '{task}'."

    def _select(self, entry: ToolEntry) -> tuple[str, str, str, bool]:
        if entry.specialist_available:
            return entry.name, "local_specialist", entry.specialist_model, True
        return self._baseline(entry)

    @staticmethod
    def _check_resolution(entry: ToolEntry, images: list[PreparedImage]) -> tuple[bool, str | None]:
        if entry.gsd_range is None:
            return True, None
        try:
            low, high = entry.gsd_range()
        except Exception:
            logger.exception("Could not read the training resolution of '%s'", entry.name)
            return True, None

        trained = f"{low:g} m" if low == high else f"{low:g}-{high:g} m"
        known = [img.analysis_gsd_m for img in images if img.analysis_gsd_m is not None]
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
        if specialist_context:
            return {"answer": specialist_context, "evidence": []}, {}
        return {"answer": f"Task {task} executed via deterministic fallback.", "evidence": []}, {}