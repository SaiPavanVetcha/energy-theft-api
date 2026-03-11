from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI(title="Energy Theft Detection API")

# Load ML models
iso_model = joblib.load("iso_model.pkl")
rf_model = joblib.load("rf_model.pkl")

# Season encoding (same as training)
season_map = {
    "winter": 0,
    "summer": 1,
    "monsoon": 2
}

@app.get("/")
def home():
    return {"message": "Energy Theft Detection API Running"}

@app.post("/predict")
def predict(data: dict):

    # Save household id separately
    household_id = data["household_id"]

    # Convert input to dataframe
    df = pd.DataFrame([data])

    # Encode season
    df["season"] = df["season"].map(season_map)

    # Remove household id from model input
    df_model = df.drop(columns=["household_id"])

    # -------- ISO MODEL --------
    iso_features = [
        "power_watts",
        "voltage_v",
        "current_a",
        "duration_minutes",
        "energy_kwh",
        "occupancy_count",
        "outside_temp_c",
        "daily_cumulative_kwh",
        "hour",
        "season"
    ]

    iso_input = df_model[iso_features]

    anomaly_score = iso_model.decision_function(iso_input)[0]

    # -------- ERROR CALCULATION --------
    predicted_energy = (df["power_watts"] * df["duration_minutes"]) / 60 / 1000

    prediction_error = abs(predicted_energy - df["energy_kwh"])[0]

    error_pct = (prediction_error / df["energy_kwh"])[0] * 100

    # -------- RF MODEL INPUT --------
    rf_input = pd.DataFrame({
        "power_watts": [df["power_watts"][0]],
        "voltage_v": [df["voltage_v"][0]],
        "current_a": [df["current_a"][0]],
        "energy_kwh": [df["energy_kwh"][0]],
        "daily_cumulative_kwh": [df["daily_cumulative_kwh"][0]],
        "occupancy_count": [df["occupancy_count"][0]],
        "hour": [df["hour"][0]],
        "outside_temp_c": [df["outside_temp_c"][0]],
        "anomaly_score": [anomaly_score],
        "prediction_error": [prediction_error],
        "error_pct": [error_pct]
    })

    theft = rf_model.predict(rf_input)[0]

    # Risk level
    if anomaly_score > 0.8:
        risk = "High"
    elif anomaly_score > 0.5:
        risk = "Medium"
    else:
        risk = "Low"

    return {
        "alert": "⚠ Possible Energy Theft Detected" if theft else "Normal Usage",
        "household_id": household_id,
        "anomaly_score": float(anomaly_score),
        "prediction_error_kwh": float(prediction_error),
        "error_percent": float(error_pct),
        "risk_level": risk
    }
