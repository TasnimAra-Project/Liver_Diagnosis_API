from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Load your trained model
model = joblib.load("liver_diagnosis_prediction.pkl")

app = FastAPI()

# Define expected input schema
class PatientData(BaseModel):
    Age: float
    Gender: int
    BMI: float
    AlcoholConsumption: float
    Smoking: int
    GeneticRisk: int
    PhysicalActivity: float
    Diabetes: int
    Hypertension: int
    LiverFunctionTest: float

@app.post("/predict")
def predict_disease(data: PatientData):
    input_data = pd.DataFrame([data.dict()])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    return {
        "prediction": int(prediction),
        "probability_of_disease": round(probability, 4)
    }
@app.get("/")
def root():
    return {"message": "Liver Diagnosis API is running"}

