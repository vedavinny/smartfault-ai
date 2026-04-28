"""
SmartFault AI — API Request/Response Schemas
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional


class SensorReading(BaseModel):
    machine_id: str = Field(..., example="MACHINE_001", description="Unique machine identifier")
    temperature: float = Field(..., ge=0, le=300, example=87.4, description="Temperature in °C")
    vibration: float = Field(..., ge=0, le=20, example=0.83, description="Vibration amplitude in g")
    pressure: float = Field(..., ge=0, le=500, example=112.5, description="Pressure in bar")
    rpm: float = Field(..., ge=0, le=10000, example=1450.0, description="Motor RPM")
    timestamp: Optional[str] = Field(None, example="2024-04-28T14:30:00")

    model_config = {"json_schema_extra": {
        "example": {
            "machine_id": "MACHINE_001",
            "temperature": 87.4,
            "vibration": 0.83,
            "pressure": 112.5,
            "rpm": 1450.0,
        }
    }}


class BatchSensorReading(BaseModel):
    readings: List[SensorReading] = Field(..., min_length=1, max_length=100)


class PredictionResponse(BaseModel):
    machine_id: str
    failure_probability: float = Field(..., description="0.0 (safe) to 1.0 (certain failure)")
    risk_level: str = Field(..., description="LOW | MEDIUM | HIGH | CRITICAL")
    predicted_failure_in_hours: Optional[float] = Field(None, description="Estimated hours until failure")
    recommended_action: str
    models_used: List[str]
    confidence: float
    timestamp: Optional[str] = None


class BatchPredictionResponse(BaseModel):
    total: int
    high_risk_count: int
    predictions: List[PredictionResponse]
    batch_timestamp: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: List[str]
    timestamp: str
