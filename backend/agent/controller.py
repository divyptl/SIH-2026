from __future__ import annotations

import asyncio
import base64
import copy
import io
import logging
import re
import time
import uuid
from typing import Any

from PIL import Image

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

# Detects VQA interrogatives that ask a question about a specific grounded region/object
_VQA_QUESTION_WORDS = re.compile(
    r"\b(how many|count|number of|what is|what are|is there|are there|is the|"
    r"condition|status|type of|kind of|damaged|intact|flooded|submerged)\b",
    re.IGNORECASE,
)

# Spatial scoping prepositions/regions that trigger 2-hop Ground -> Crop -> VQA chaining
_SPATIAL_SCOPE_WORDS = re.compile(
    r"\b(in the|near the|around the|next to|beside|along the|at the|on the|"
    r"top|bottom|left|right|centre|center|corner|bridge|river|water|lake|harbour|port|airport)\b",
    re.IGNORECASE,
)

_PHYSICS_KEYWORDS = (
    "water", "flood", "river", "lake", "inundat",
    "built", "urban", "building", "settlement",
    "vegetat", "forest", "crop", "green", "tree",
)


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


def _should_run_two_hop(
    configuration: InputConfiguration,
    task: Task,
    query: str,
) -> bool:
    """
    Determines if a single-image query requires 2-Hop Agentic Chaining:
    Hop 1 (rs-grounding) -> High-Res ROI Crop -> Hop 2 (rs-vqa).
    """
    if configuration != "single":
        return False
    g_entry = TOOL_REGISTRY.get("grounding")
    v_entry = TOOL_REGISTRY.get("vqa")
    if not (g_entry and v_entry and g_entry.specialist_available and v_entry.specialist_available):
        return False

    has_vqa_q = bool(_VQA_QUESTION_WORDS.search(query))
    has_grounding_trigger = bool(_GROUNDING_WORDS.search(query)) or bool(_SPATIAL_SCOPE_WORDS.search(query))
    return has_vqa_q and has_grounding_trigger


def _crop_prepared_image(
    image: PreparedImage,
    norm_box: dict[str, float],
    pad_ratio: float = 0.15,
) -> tuple[PreparedImage, tuple[int, int, int, int]]:
    """
    Crops a PreparedImage around a normalized bounding box with context padding,
    preserving native high-resolution pixels for downstream VQA inspection.
    """
    raw = base64.b64decode(image.data_uri.split(",", 1)[1])
    pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
    w, h = pil_img.size

    bw = norm_box["x_max"] - norm_box["x_min"]
    bh = norm_box["y_max"] - norm_box["y_min"]
    x1 = max(0, int(round((norm_box["x_min"] - pad_ratio * bw) * w)))
    y1 = max(0, int(round((norm_box["y_min"] - pad_ratio * bh) * h)))
    x2 = min(w, int(round((norm_box["x_max"] + pad_ratio * bw) * w)))
    y2 = min(h, int(round((norm_box["y_max"] + pad_ratio * bh) * h)))

    if x2 <= x1 + 4 or y2 <= y1 + 4:
        return image, (0, 0, w, h)

    cropped_pil = pil_img.crop((x1, y1, x2, y2))
    buf = io.BytesIO()
    cropped_pil.save(buf, format="PNG")
    cropped_uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    cropped_prep = copy.copy(image)
    object.__setattr__(cropped_prep, "data_uri", cropped_uri)
    return cropped_prep, (x1, y1, x2, y2)


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

        two_hop_active = (not task_override) and _should_run_two_hop(configuration, task, query)
        if two_hop_active:
            rationale += " Compound spatial-VQA query detected: chaining rs-grounding -> ROI crop -> rs-vqa."

        steps.append(
            TraceStep(
                stage="classify",
                tool="query-router",
                model=None,
                params={"input_configuration": configuration, "two_hop_chaining": two_hop_active},
                detail=f"Task '{task}' ({task_source}). {rationale}",
                duration_ms=step_timer.ms(),
            )
        )

        # 3. Select tool(s)
        step_timer = _Timer()
        use_specialist = entry.specialist_available
        if use_specialist:
            use_specialist, note = await asyncio.to_thread(self._check_resolution, entry, images)
            if note:
                warnings.append(note)
        tool_name, backend, model_name, domain_adapted = (
            self._select(entry) if use_specialist else self._baseline(entry)
        )
        selected_tools = [tool_name]
        if two_hop_active:
            g_entry = TOOL_REGISTRY["grounding"]
            v_entry = TOOL_REGISTRY["vqa"]
            selected_tools = [g_entry.name, "high-res-roi-cropper", v_entry.name]

        steps.append(
            TraceStep(
                stage="select",
                tool="tool-selector",
                model=model_name,
                params={"backend": backend, "selected_tools": selected_tools},
                detail=f"Selected {', '.join(selected_tools)} for task '{task}'.",
                duration_ms=step_timer.ms(),
            )
        )

        # 4. Execute (Single-Hop or 2-Hop Agentic Chain)
        payload: dict[str, Any] | None = None
        specialist_payload: dict[str, Any] | None = None
        usage: dict[str, Any] = {}
        params: dict[str, Any] = {}

        if two_hop_active:
            payload = await self._execute_two_hop_chain(query=query, images=images, steps=steps, warnings=warnings)
            if payload is not None:
                domain_adapted = True
                model_name = f"{TOOL_REGISTRY['grounding'].specialist_model} -> {TOOL_REGISTRY['vqa'].specialist_model}"

        if payload is None:
            step_timer = _Timer()
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
                selected_tools=selected_tools,
                steps=steps,
                total_duration_ms=total_ms,
                domain_adapted=domain_adapted,
                warnings=warnings,
            ),
            usage=Usage(**usage) if usage else None,
            narration=narration,
        )

    async def _execute_two_hop_chain(
        self,
        *,
        query: str,
        images: list[PreparedImage],
        steps: list[TraceStep],
        warnings: list[str],
    ) -> dict[str, Any] | None:
        """
        Executes 2-Hop Agentic Chaining:
        Hop 1: rs-grounding (with Physics Re-Ranker) to find Box 1.
        Crop : High-resolution ROI crop around Box 1 (+15% context pad).
        Hop 2: rs-vqa on the cropped ROI to answer the factual question.
        """
        g_entry = TOOL_REGISTRY["grounding"]
        v_entry = TOOL_REGISTRY["vqa"]
        if g_entry.runner is None or v_entry.runner is None:
            return None

        # Hop 1: Grounding
        hop1_timer = _Timer()
        try:
            g_payload = await asyncio.to_thread(g_entry.runner, query, images)
        except Exception as exc:
            logger.warning("2-Hop Hop-1 (grounding) failed (%s); falling back to single-hop", exc)
            return None

        g_evidence = g_payload.get("evidence") or []
        best_box = next((item["box"] for item in g_evidence if isinstance(item.get("box"), dict)), None)
        steps.append(
            TraceStep(
                stage="execute",
                tool=g_entry.name,
                model=g_entry.specialist_model,
                params={"hop": 1, "checkpoint": g_entry.checkpoint, "candidates_found": len(g_evidence)},
                detail=f"Hop 1 (Grounding): localised {len(g_evidence)} candidate region(s) with physics re-ranking.",
                duration_ms=hop1_timer.ms(),
            )
        )

        if not best_box:
            return g_payload

        # Intermediate Step: High-Resolution ROI Crop
        crop_timer = _Timer()
        cropped_img, px_box = await asyncio.to_thread(_crop_prepared_image, images[0], best_box, 0.15)
        steps.append(
            TraceStep(
                stage="crop",
                tool="high-res-roi-cropper",
                params={"bbox_px": list(px_box), "pad_ratio": 0.15},
                detail=f"Cropped high-resolution ROI {px_box} around Box 1 for targeted VQA inspection.",
                duration_ms=crop_timer.ms(),
            )
        )

        # Hop 2: Targeted VQA on the High-Res Crop
        hop2_timer = _Timer()
        try:
            v_payload = await asyncio.to_thread(v_entry.runner, query, [cropped_img])
        except Exception as exc:
            logger.warning("2-Hop Hop-2 (VQA) failed (%s); returning Hop-1 grounding result", exc)
            warnings.append(f"Targeted VQA on cropped ROI failed ({exc}); showing grounding result.")
            return g_payload

        steps.append(
            TraceStep(
                stage="execute",
                tool=v_entry.name,
                model=v_entry.specialist_model,
                params={"hop": 2, "checkpoint": v_entry.checkpoint},
                detail=f"Hop 2 (Targeted VQA): answered query on high-resolution ROI crop (conf={v_payload.get('confidence', 0.0):.2f}).",
                duration_ms=hop2_timer.ms(),
            )
        )

        vqa_answer = str(v_payload.get("answer") or "").strip().rstrip(".")
        g_answer = str(g_payload.get("answer") or "").strip()
        combined_conf = round(
            0.5 * float(g_payload.get("confidence", 0.5)) + 0.5 * float(v_payload.get("confidence", 0.5)),
            4,
        )

        return {
            "answer": f"{vqa_answer}. {g_answer}",
            "confidence": combined_conf,
            "evidence": g_evidence + (v_payload.get("evidence") or []),
        }

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