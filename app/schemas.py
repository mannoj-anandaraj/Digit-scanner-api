from pydantic import BaseModel
from typing import List


class DigitResult(BaseModel):
    digit: int
    confidence: float
    scores: List[float]


class PredictionResponse(BaseModel):
    number: str
    count: int
    predictions: List[DigitResult]

    model_config = {
        "json_schema_extra": {
            "example": {
                "number": "27",
                "count": 2,
                "predictions": [
                    {
                        "digit": 2,
                        "confidence": 0.951,
                        "scores": [0.0, 0.0, 0.951, 0.0, 0.0, 0.0, 0.0, 0.049, 0.0, 0.0]
                    },
                    {
                        "digit": 7,
                        "confidence": 0.997,
                        "scores": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.997, 0.0, 0.003]
                    }
                ]
            }
        }
    }


class HealthResponse(BaseModel):
    status: str
    model: str
    version: str
