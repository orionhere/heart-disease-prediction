from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import os

app = FastAPI(title="Heart Disease Prediction API")

# Allow CORS for the local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "heart_disease_model.pkl"
model = None

@app.on_event("startup")
def load_ml_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"Loaded model from {MODEL_PATH}")
    else:
        print(f"Warning: Model file {MODEL_PATH} not found. Please train a model first.")

class PatientData(BaseModel):
    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

@app.post("/predict")
def predict(data: PatientData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Ensure heart_disease_model.pkl exists.")
    
    # Convert input to DataFrame as the model pipeline expects it
    input_data = pd.DataFrame([data.dict()])
    
    try:
        prediction = model.predict(input_data)[0]
        
        prob = None
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(input_data)[0][1]
        elif hasattr(model.steps[-1][1], "predict_proba"):
            prob = model.predict_proba(input_data)[0][1]

        return {
            "prediction": int(prediction),
            "probability": float(prob) if prob is not None else None,
            "risk_level": "High Risk" if prediction == 1 else "Low Risk"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

# To run: uvicorn backend_api:app --reload
