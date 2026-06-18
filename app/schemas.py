from pydantic import BaseModel
from typing import List


class PredictionResponse(BaseModel):
    digit: int
    confidence: float
    scores: List[float]

    model_config = {
        "json_schema_extra": {
            "example": {
                "digit": 7,
                "confidence": 0.997,
                "scores": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.997, 0.0, 0.003],
            }
        }
    }


class HealthResponse(BaseModel):
    status: str
    model: str
    version: str
