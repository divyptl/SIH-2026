"""
STUB — do not use until Phase 4 (small RS domain adaptation) begins.

Placeholder LoRA hyperparameters for a future SkyEyeGPT/MiniGPT-v2
adaptation experiment (Section 34-36 of the project spec). Intentionally
NOT wired into model.py yet: baseline inference (Phase 1-3) must be
working and evaluated first (RSVQA/VRSBench, Section 30-32) before any
fine-tuning is justified.

When this phase actually starts:
  1. Pick a SMALL subset of BigEarthNet.txt / VRSBench / RSVQA / SkyEye-968k
     (Section 36) — do not attempt the full dataset first.
  2. Fill in target_modules based on the actual LLM backbone used by the
     loaded MiniGPT-v2 config (inspect the model, don't guess names).
  3. Keep the eval split used in evaluate.py completely untouched.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class LoRAConfig:
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    # TODO: populate once the LLM backbone inside the loaded MiniGPT-v2
    # config is inspected — do not assume generic names like q_proj/v_proj
    # apply without checking.
    target_modules: List[str] = field(default_factory=list)
    num_train_epochs: int = 1
    learning_rate: float = 1e-4
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8


DEFAULT_LORA_CONFIG = LoRAConfig()
