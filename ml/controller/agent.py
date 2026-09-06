import time
import re
from typing import Dict
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from ml.controller.schema import ModelRequest, ModelResponse, SpecialistModel

class AgentController:
    """
    Agentic Controller for SatQuery AI.
    Routes incoming natural language queries to the appropriate specialist model
    based on the query content and input image modalities.
    """

    def __init__(self, models: Dict[str, SpecialistModel]):
        """
        Initialize the AgentController with available specialist models.
        
        Args:
            models (Dict[str, SpecialistModel]): A dictionary mapping task names 
                                                 to model instances.
        """
        self.models = models

    def parse_intent(self, query: str, num_images: int, modalities: list[str]) -> str:
        """
        Heuristic rule-based intent parsing to route queries.
        
        Args:
            query (str): The natural language query.
            num_images (int): Number of input images.
            modalities (list[str]): The modalities of the images (e.g., ["optical", "sar"]).
            
        Returns:
            str: The name of the specialist model to route to (e.g., "change_vqa", "fusion", "grounding", "vqa").
        """
        query_lower = query.lower()
        
        # 1. Bi-temporal Change Detection (2 images of same modality)
        if num_images == 2 and len(set(modalities)) == 1:
            return "change_vqa"
            
        # 2. Optical-SAR Fusion (2 images, one optical, one SAR)
        if num_images == 2 and "optical" in modalities and "sar" in modalities:
            return "fusion"
            
        # 3. Grounding (Single image, keywords indicate localization)
        grounding_keywords = [
            "where", "highlight", "locate", "find", "bounding box", 
            "point out", "show me", "identify the location"
        ]
        if num_images == 1 and any(keyword in query_lower for keyword in grounding_keywords):
            return "grounding"
            
        # 4. Default: Standard VQA (Single image, general questions)
        return "vqa"

    def execute(self, query: str, images: list[str], modalities: list[str]) -> ModelResponse:
        """
        Execute the agentic flow: parse intent, route, and aggregate results.
        
        Args:
            query (str): The user's question.
            images (list[str]): List of image paths or base64 payloads.
            modalities (list[str]): List of image modalities.
            
        Returns:
            ModelResponse: The final answer with visual evidence and execution trace.
        """
        start_time = time.time()
        
        num_images = len(images)
        logger.info(f"Received query: '{query}' with {num_images} images ({modalities})")
        
        # 1. Intent Parsing
        task_hint = self.parse_intent(query, num_images, modalities)
        logger.info(f"Parsed intent: routing to '{task_hint}'")
        
        if task_hint not in self.models:
            raise ValueError(f"Specialist model for task '{task_hint}' is not available.")
            
        specialist = self.models[task_hint]
        
        # 2. Construct Model Request
        request = ModelRequest(
            query=query,
            images=images,
            modalities=modalities,
            task_hint=task_hint
        )
        
        # 3. Execute Specialist Model
        logger.info(f"Executing specialist model: {task_hint}")
        response = specialist.predict(request)
        
        # 4. Finalize Response (Add execution metadata)
        execution_time_ms = (time.time() - start_time) * 1000
        response.execution_time_ms = execution_time_ms
        
        if not response.model_name:
            response.model_name = task_hint
            
        logger.info(f"Execution completed in {execution_time_ms:.2f}ms")
        return response
