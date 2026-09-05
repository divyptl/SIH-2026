"""
STUB — do not use until Phase 4 (small RS domain adaptation) begins.

Will load a small, documented subset of BigEarthNet.txt / VRSBench /
RSVQA / SkyEye-968k image-question-answer triples for LoRA fine-tuning
(Section 34-36). Reuses ml.vqa.preprocessing.load_image so training and
inference see images through the identical pipeline.

Intentionally unimplemented beyond the interface: fill this in only
after Phases 1-3 (baseline inference + evaluation) are done and a
concrete adaptation dataset subset has been selected.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, NamedTuple

from ..preprocessing import load_image


class TrainExample(NamedTuple):
    image_path: str
    question: str
    answer: str
    modality: str = "optical"


def load_training_subset(manifest_path: Path) -> List[TrainExample]:
    """
    TODO (Phase 4): implement once a concrete small subset (e.g. a few
    thousand BigEarthNet.txt / VRSBench samples) has been selected and
    flattened into a manifest (see evaluate.iter_dataset for the same
    JSONL convention used on the eval side).
    """
    raise NotImplementedError(
        "Training dataset loading is a Phase 4 stub. Implement this only "
        "after baseline SkyEyeGPT inference (Phases 1-3) is working and "
        "evaluated on RSVQA/VRSBench."
    )
