
# import all necessary libraries
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from typing import List

# Create FastAPI app
app = FastAPI(title="Churn API", version="1.0.0")

# Load your trained model (make sure model.pkl is in same folder)
model = joblib.load("Top7_rf_model.pickle")

# Define input structure (Top 7 features)
class ChurnRequest(BaseModel):
    Day_Mins: float
    Night_Charge: float
    Day_Charge: float
    Eve_Mins: float
    Eve_Charge: float
    Night_Calls: int
    Vmail_Message: int

# Root endpoint
@app.get("/")
async def root():
    return {"message": " Churn Prediction API"}

# Prediction endpoint
@app.get("/predict/")
async def predict_churn(
    Day_Mins: float,
    Night_Charge: float,
    Day_Charge: float,
    Eve_Mins: float,
    Eve_Charge: float,
    Night_Calls: int,
    Vmail_Message: int
):
    # Create DataFrame from URL parameters
    data = [[Day_Mins, Night_Charge, Day_Charge, Eve_Mins, Eve_Charge, Night_Calls, Vmail_Message]]
    df = pd.DataFrame(data, columns=[
        "Day_Mins", "Night_Charge", "Day_Charge", "Eve_Mins",
        "Eve_Charge", "Night_Calls", "Vmail_Message"
    ])
    
    # Predict
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    
    return {
        "prediction": bool(prediction),
        "churn_probability": float(probability),
        "confidence": float(max(model.predict_proba(df)[0])),
        "message": "High risk" if probability > 0.7 else "Low risk"
    }


# Health check
@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": True}
