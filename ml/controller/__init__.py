"""
Agentic Controller Module

This module serves as the central orchestrator for SatQuery AI.
It handles intent parsing, routing requests to the appropriate specialist models,
and aggregating their responses.
"""

from ml.controller.schema import Evidence, ModelRequest, ModelResponse, SpecialistModel
from ml.controller.agent import AgentController

__all__ = [
    "Evidence",
    "ModelRequest", 
    "ModelResponse",
    "SpecialistModel",
    "AgentController"
]
