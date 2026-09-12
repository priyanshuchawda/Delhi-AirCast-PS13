"""Small local API for AQI calculations and trained forecast artifacts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from .aqi import calculate_aqi, pm25_proxy


DATA_ROOT = Path(os.environ.get("DELHI_AIRCAST_DATA_ROOT", "."))
app = FastAPI(title="Delhi AirCast", version="0.1.0")


class AQIRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pm25: float | None = None
    pm10: float | None = None
    no2: float | None = None
    so2: float | None = None
    o3: float | None = None
    co: float | None = None
    nh3: float | None = None


def _station_summary_path() -> Path:
    return DATA_ROOT / "data/processed/cpcb_2024_25_hourly/station_summary.csv"


@app.get("/health")
def health() -> dict[str, Any]:
    summary = _station_summary_path()
    return {
        "status": "ok",
        "aqi_engine": "available",
        "station_summary": summary.exists(),
        "data_root": str(DATA_ROOT.resolve()),
    }


@app.post("/aqi")
def aqi(payload: AQIRequest) -> dict[str, Any]:
    pollutants = payload.model_dump(exclude_none=True)
    result = pm25_proxy(pollutants["pm25"]) if set(pollutants) == {"pm25"} else calculate_aqi(pollutants)
    return {
        "value": result.value,
        "category": result.category,
        "status": result.status,
        "subindices": result.subindices,
    }


@app.get("/stations")
def stations() -> dict[str, Any]:
    path = _station_summary_path()
    if not path.exists():
        raise HTTPException(status_code=503, detail="Run normalize_cpcb.py first")
    frame = pd.read_csv(path)
    return {"count": int(len(frame)), "stations": frame.to_dict(orient="records")}


@app.get("/forecast/{station_id}")
def forecast(station_id: str) -> dict[str, Any]:
    dataset = DATA_ROOT / f"data/processed/{station_id}_next_hour_forecast.parquet"
    model_path = DATA_ROOT / f"data/runs/{station_id}_next_hour_baseline/model.joblib"
    if not dataset.exists() or not model_path.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                f"Forecast artifact for {station_id} is unavailable; run "
                f"build_forecast_dataset.py and train_baseline.py first"
            ),
        )

    bundle = joblib.load(model_path)
    features = bundle["feature_columns"]
    frame = pd.read_parquet(dataset).sort_values("timestamp_utc")
    usable = frame.dropna(subset=features)
    if usable.empty:
        raise HTTPException(status_code=503, detail="No complete feature row is available")
    latest = usable.iloc[-1]
    prediction = float(bundle["model"].predict(latest[features].to_frame().T)[0])
    return {
        "station_id": station_id,
        "as_of_utc": latest["timestamp_utc"].isoformat(),
        "target_timestamp_utc": latest["target_timestamp_utc"].isoformat(),
        "current_pm25": float(latest["pm25"]),
        "forecast_pm25": prediction,
        "model": "HistGradientBoostingRegressor",
        "feature_count": len(features),
    }
