import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

app = FastAPI(title="Fraud Detection API")

# Load model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
model = joblib.load(MODEL_PATH)

EXPECTED_FEATURES = 30

class FraudRequest(BaseModel):
    features: List[float] = Field(
        ..., description=f"List of {EXPECTED_FEATURES} numerical features"
    )

class FraudResponse(BaseModel):
    fraud_prediction: int
    fraud_probability: float

@app.post("/predict", response_model=FraudResponse)
def predict(request: FraudRequest):
    if len(request.features) != EXPECTED_FEATURES:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {EXPECTED_FEATURES} features, received {len(request.features)}"
        )

    data = np.array(request.features).reshape(1, -1)
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    return {
        "fraud_prediction": int(prediction),
        "fraud_probability": round(float(probability), 6)
    }

# ✅ HEALTH CHECK ENDPOINT (NEW)
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None
    }
