"""
Confidence scoring for SatQuery VQA/Captioning.

WHY THIS FILE EXISTS:
    SkyEyeGPT does not natively output a calibrated probability just
    because it generated fluent text. We must NOT hard-code a number
    like 0.9 "because the answer looks good" (see project spec, Section
    23 — "DO NOT FABRICATE CONFIDENCE").

WHAT THIS ACTUALLY COMPUTES:
    confidence = f(generation_score, answer_text)

    - If the backend exposes a generation_score() (e.g. an average
      token log-probability from MiniGPT-v2's decoding), it is mapped
      into (0, 1) via a monotonic squashing function.
    - Otherwise, a coarse text-heuristic is used (empty/refusal ->
      low, hedged -> medium-low, direct non-empty -> medium).

WHAT THIS DOES NOT MEAN:
    Neither path produces a statistically calibrated "probability the
    answer is correct". Present this to users/evaluators as a triage
    signal only, until self-consistency checks, an answerability
    classifier, or specialist cross-validation are implemented (see
    Section 23/28 of the project spec).
"""
from __future__ import annotations

import math
import re
from typing import Optional

_HEDGE_PATTERNS = re.compile(
    r"\b(i am not sure|i'm not sure|unclear|cannot determine|can't determine|"
    r"not visible|hard to tell|possibly|maybe|might be|unable to)\b",
    re.IGNORECASE,
)

_REFUSAL_PATTERNS = re.compile(
    r"\b(i don't know|i do not know|no answer|n/a)\b", re.IGNORECASE
)


def score_to_confidence(log_prob: float) -> float:
    """
    Map an average token log-probability to (0, 1) via a logistic
    squashing function. This is a monotonic RESCALING, not a calibrated
    probability of correctness.
    """
    return 1.0 / (1.0 + math.exp(-log_prob))


def heuristic_confidence(answer: str) -> float:
    """
    Fallback used only when the backend exposes no generation score.

    Intentionally conservative and coarse: this is a triage signal for
    downstream aggregation/UI, not a correctness estimate.
    """
    text = (answer or "").strip()
    if not text:
        return 0.0
    if _REFUSAL_PATTERNS.search(text):
        return 0.1
    if _HEDGE_PATTERNS.search(text):
        return 0.35
    if len(text) < 3:
        return 0.2
    return 0.5


def compute_confidence(answer: str, generation_score: Optional[float]) -> float:
    """
    Main entry point used by inference.py.

    Returns a float in [0, 1]. Callers (API responses, frontend, docs)
    must not present this as a calibrated probability of correctness —
    document the caveat wherever this value is surfaced.
    """
    if generation_score is not None:
        try:
            mapped = score_to_confidence(float(generation_score))
            return max(0.0, min(1.0, mapped))
        except (TypeError, OverflowError, ValueError):
            pass
    return heuristic_confidence(answer)
