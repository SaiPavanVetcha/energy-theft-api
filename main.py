from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

iso_model = joblib.load("iso_model.pkl")
rf_model = joblib.load("rf_model.pkl")

@app.get("/")
def home():
    return {"message":"Energy Theft Detection API running"}

@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])

    anomaly_score = iso_model.decision_function(df)[0]
    theft = rf_model.predict(df)[0]

    return {
        "household_id": data["householdid"],
        "anomaly_score": float(anomaly_score),
        "theft_detected": bool(theft)

    }
