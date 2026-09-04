"""
Common Interface Schema for SatQuery AI Specialist Models.

This defines the exact contract that every specialist model (VQA, Grounding,
Change Detection, Fusion) must follow. The Agentic Controller relies on this
uniform interface to route queries and aggregate responses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol
from pathlib import Path


@dataclass
class Evidence:
    """Visual or numerical evidence backing up the model's answer."""
    
    # Type of evidence (e.g., "bbox", "mask", "heatmap", "embedding")
    type: str
    
    # Payload (e.g., [x1, y1, x2, y2, confidence, label] for bbox)
    data: Any
    
    # Human-readable explanation of what this evidence shows
    description: str


@dataclass
class ModelRequest:
    """Input request routed to a specialist model."""
    
    # The user's natural language question
    query: str
    
    # Paths or base64 strings of the images to analyze
    # (1 for VQA/Grounding, 2 for Change Detection/Fusion)
    images: list[str | Path]
    
    # The modalities of the images (e.g., ["optical"], ["optical", "sar"])
    modalities: list[str]
    
    # Optional explicitly parsed intent from the controller
    task_hint: str | None = None


@dataclass
class ModelResponse:
    """Standardized output from a specialist model."""
    
    # The text answer to the query
    answer: str
    
    # Confidence score from 0.0 to 1.0
    confidence: float
    
    # List of evidence supporting the answer
    evidence: list[Evidence] = field(default_factory=list)
    
    # Identifier for the model that generated this response
    model_name: str = ""
    
    # Execution time in milliseconds
    execution_time_ms: float = 0.0


class SpecialistModel(Protocol):
    """
    Protocol defining the mandatory interface for all specialist models.
    
    Every model in ml/vqa, ml/grounding, ml/change_detection, and ml/fusion
    MUST expose a class that implements this interface.
    """
    
    def predict(self, request: ModelRequest) -> ModelResponse:
        """Process a request and return a standardized response."""
        ...
