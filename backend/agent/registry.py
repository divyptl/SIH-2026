"""Registry of specialist models and tools the controller can select from.

Every task declares the fine-tuned specialist that *should* serve it plus the
OpenRouter baseline that serves it until that specialist is trained and wired
in. ``domain_adapted`` is what distinguishes the two: the problem statement is
explicit that a generic VLM alone does not satisfy the requirements, so the
flag is surfaced in the execution trace rather than hidden.

To promote a task to its specialist, add a runner for it to
``services/specialists.py`` and a checkpoint setting to ``config.py``, then set
both on the entry below.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config import get_settings
from schemas import InputConfiguration, Task, ToolInfo
from services import specialists
from services.images import PreparedImage

# Runs a specialist on (query, images) and returns the baseline's payload shape.
Runner = Callable[[str, list[PreparedImage]], dict[str, Any]]


@dataclass(frozen=True)
class ToolEntry:
    """One selectable tool in the registry."""

    name: str
    task: Task
    description: str
    accepts: tuple[InputConfiguration, ...]

    # Module path of the fine-tuned specialist that owns this task once trained.
    specialist_module: str

    # Runs the specialist. While None, the controller uses the OpenRouter baseline.
    runner: Runner | None = None

    # Human-readable weights the runner serves, for the trace and the registry.
    checkpoint: str | None = None

    # (min, max) metres per pixel the specialist was trained on, or None when it
    # declares no range. Called lazily, since reading it may load the model.
    gsd_range: Callable[[], tuple[float, float]] | None = None

    # Parameters the controller is permitted to configure for this tool.
    permitted_params: tuple[str, ...] = field(default=("temperature", "max_tokens"))

    @property
    def specialist_available(self) -> bool:
        return self.runner is not None

    @property
    def specialist_model(self) -> str:
        """The model name reported for a specialist answer."""
        return f"{self.specialist_module} ({self.checkpoint})"


def _specialist(
    runner: Runner, checkpoint: Path | None, describe: Callable[[Path], str] | None = None
) -> dict[str, Any]:
    """Entry fields that activate a specialist whose checkpoint is on disk."""
    settings = get_settings()
    if not settings.specialists_enabled or checkpoint is None or not checkpoint.is_file():
        return {}
    label = (describe or specialists.checkpoint_label)(checkpoint)
    return {"runner": runner, "checkpoint": label}


_settings = get_settings()


TOOL_REGISTRY: dict[Task, ToolEntry] = {
    "vqa": ToolEntry(
        name="rs-vqa",
        task="vqa",
        description="Answers a factual question about a single optical/multispectral or SAR image.",
        accepts=("single", "cross_modal_pair", "bi_temporal_pair"),
        specialist_module="ml.vqa",
    ),
    "caption": ToolEntry(
        name="rs-captioner",
        task="caption",
        description="Describes land cover, scene context, and major objects in a single image.",
        accepts=("single",),
        specialist_module="ml.vqa",
    ),
    "grounding": ToolEntry(
        name="rs-grounding",
        task="grounding",
        description="Localises the region referred to by the query and returns bounding boxes.",
        accepts=("single",),
        specialist_module="ml.grounding",
        **_specialist(
            specialists.run_grounding,
            _settings.grounding_checkpoint,
            specialists.describe_grounding,
        ),
    ),
    "change_vqa": ToolEntry(
        name="rs-change-vqa",
        task="change_vqa",
        description="Answers a question about what changed between two co-located dates.",
        accepts=("bi_temporal_pair",),
        specialist_module="ml.C_VQA",
        gsd_range=specialists.change_vqa_gsd_range,
        **_specialist(specialists.run_change_vqa, _settings.change_vqa_checkpoint),
    ),
    "change_description": ToolEntry(
        name="rs-change-describer",
        task="change_description",
        description="Describes and localises change between two co-located dates.",
        accepts=("bi_temporal_pair",),
        specialist_module="ml.C_VQA",
        gsd_range=specialists.change_vqa_gsd_range,
        **_specialist(specialists.run_change_vqa, _settings.change_vqa_checkpoint),
    ),
    "fusion": ToolEntry(
        name="rs-optical-sar-fusion",
        task="fusion",
        description=(
            "Extracts complementary information from a co-registered optical + SAR pair."
        ),
        accepts=("cross_modal_pair",),
        specialist_module="ml.fusion",
        **_specialist(specialists.run_fusion, _settings.fusion_checkpoint),
    ),
}

# Fallback when a task cannot serve the given input configuration.
DEFAULT_SINGLE_TASK: Task = "vqa"
DEFAULT_PAIR_TASK: Task = "change_vqa"


def describe_registry(baseline_model: str) -> list[ToolInfo]:
    """Registry contents for the ``/api/registry`` endpoint."""
    tools: list[ToolInfo] = []
    for entry in TOOL_REGISTRY.values():
        if entry.specialist_available:
            tools.append(
                ToolInfo(
                    name=entry.name,
                    task=entry.task,
                    description=entry.description,
                    backend="local_specialist",
                    model=entry.specialist_model,
                    available=True,
                    domain_adapted=True,
                )
            )
        else:
            tools.append(
                ToolInfo(
                    name=f"{entry.name}-baseline",
                    task=entry.task,
                    description=(
                        f"{entry.description} Served by the generic VLM baseline until "
                        f"'{entry.specialist_module}' is trained and registered."
                    ),
                    backend="openrouter",
                    model=baseline_model,
                    available=True,
                    domain_adapted=False,
                )
            )
    return tools
