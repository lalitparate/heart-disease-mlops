"""
Pydantic schemas for FastAPI request and response validation.
"""

from pydantic import BaseModel, Field
from typing import List


class PatientFeatures(BaseModel):
    age:      float = Field(..., ge=1,   le=120, description="Age in years")
    sex:      float = Field(..., ge=0,   le=1,   description="0 = female, 1 = male")
    cp:       float = Field(..., ge=0,   le=3,   description="Chest pain type (0-3)")
    trestbps: float = Field(..., ge=80,  le=250, description="Resting blood pressure (mmHg)")
    chol:     float = Field(..., ge=100, le=600, description="Serum cholesterol (mg/dl)")
    fbs:      float = Field(..., ge=0,   le=1,   description="Fasting blood sugar > 120 mg/dl")
    restecg:  float = Field(..., ge=0,   le=2,   description="Resting ECG results (0-2)")
    thalach:  float = Field(..., ge=60,  le=220, description="Max heart rate achieved")
    exang:    float = Field(..., ge=0,   le=1,   description="Exercise-induced angina")
    oldpeak:  float = Field(..., ge=0.0, le=10.0,description="ST depression (exercise vs rest)")
    slope:    float = Field(..., ge=0,   le=2,   description="Slope of peak exercise ST segment")
    ca:       float = Field(..., ge=0,   le=4,   description="Number of major vessels (0-4)")
    thal:     float = Field(..., ge=0,   le=3,   description="Thalassemia type (0-3)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 52, "sex": 1, "cp": 0, "trestbps": 125,
                "chol": 212, "fbs": 0, "restecg": 1, "thalach": 168,
                "exang": 0, "oldpeak": 1.0, "slope": 2, "ca": 2, "thal": 3,
            }
        }
    }


class BatchRequest(BaseModel):
    records: List[PatientFeatures] = Field(..., min_length=1, max_length=500)


class PredictionResponse(BaseModel):
    prediction:  int   = Field(..., description="0 = No Disease, 1 = Disease")
    probability: float = Field(..., description="Probability of heart disease (0-1)")
    label:       str   = Field(..., description="Human-readable prediction label")


class BatchResponse(BaseModel):
    predictions: List[PredictionResponse]
    count:       int


class HealthResponse(BaseModel):
    status:       str
    model_loaded: bool
    version:      str = "1.0.0"
