from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI(title="Energy Theft Detection API")

# Load models
iso_model = joblib.load("iso_model.pkl")
rf_model = joblib.load("rf_model.pkl")

# Season encoding (must match training)
season_map = {
    "winter": 0,
    "summer": 1,
    "monsoon": 2
}

@app.get("/")
def home():
    return {"message": "Energy Theft Detection API is running"}

@app.post("/predict")
def predict(data: dict):
    try:

        household_id = data["household_id"]
        df = pd.DataFrame([data])

        df["season"] = df["season"].map(season_map).fillna(0)

        df_model = df.drop(columns=["household_id"])

        iso_features = [
            "power_watts","voltage_v","current_a","duration_minutes",
            "energy_kwh","occupancy_count","outside_temp_c",
            "daily_cumulative_kwh","hour","season"
        ]

        iso_input = df_model[iso_features]

        anomaly_score = iso_model.decision_function(iso_input)[0]

        predicted_energy = (df["power_watts"] * df["duration_minutes"]) / 60 / 1000
        prediction_error = abs(predicted_energy - df["energy_kwh"])[0]
        error_pct = (prediction_error / df["energy_kwh"])[0] * 100

        rf_input = pd.DataFrame({
            "power_watts":[df["power_watts"][0]],
            "voltage_v":[df["voltage_v"][0]],
            "current_a":[df["current_a"][0]],
            "energy_kwh":[df["energy_kwh"][0]],
            "daily_cumulative_kwh":[df["daily_cumulative_kwh"][0]],
            "occupancy_count":[df["occupancy_count"][0]],
            "hour":[df["hour"][0]],
            "outside_temp_c":[df["outside_temp_c"][0]],
            "anomaly_score":[anomaly_score],
            "prediction_error":[prediction_error],
            "error_pct":[error_pct]
        })

        theft_prediction = rf_model.predict(rf_input)[0]

        return {
            "household_id": household_id,
            "anomaly_score": float(anomaly_score),
            "prediction_error": float(prediction_error),
            "risk": "High" if theft_prediction else "Low"
        }

    except Exception as e:
        return {"error": str(e)}
