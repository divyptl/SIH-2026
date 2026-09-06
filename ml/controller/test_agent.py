import sys
import os

# Ensure the root directory is in sys.path for absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from ml.controller.agent import AgentController
from ml.controller.schema import ModelRequest, ModelResponse, SpecialistModel

class MockSpecialistModel(SpecialistModel):
    def __init__(self, name: str):
        self.name = name

    def predict(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            answer=f"Mock answer from {self.name}",
            confidence=0.95,
            model_name=self.name
        )

def test_routing():
    models = {
        "vqa": MockSpecialistModel("vqa"),
        "grounding": MockSpecialistModel("grounding"),
        "change_vqa": MockSpecialistModel("change_vqa"),
        "fusion": MockSpecialistModel("fusion"),
    }
    
    agent = AgentController(models=models)
    
    tests = [
        {
            "query": "What is the land cover type?",
            "images": ["img1.png"],
            "modalities": ["optical"],
            "expected_model": "vqa"
        },
        {
            "query": "Highlight the water body",
            "images": ["img1.png"],
            "modalities": ["optical"],
            "expected_model": "grounding"
        },
        {
            "query": "What changed between these dates?",
            "images": ["img1.png", "img2.png"],
            "modalities": ["optical", "optical"],
            "expected_model": "change_vqa"
        },
        {
            "query": "Find built up areas using both",
            "images": ["opt.png", "sar.png"],
            "modalities": ["optical", "sar"],
            "expected_model": "fusion"
        }
    ]
    
    for i, t in enumerate(tests):
        print(f"Test {i+1}: '{t['query']}'")
        res = agent.execute(t["query"], t["images"], t["modalities"])
        print(f"Routed to: {res.model_name} (Expected: {t['expected_model']})")
        assert res.model_name == t["expected_model"], f"Failed routing. Expected {t['expected_model']}, got {res.model_name}"
        print("Success!\n")
        
if __name__ == "__main__":
    test_routing()
    print("All tests passed!")
