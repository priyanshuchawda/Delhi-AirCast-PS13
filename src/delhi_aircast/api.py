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
    multistation_dataset = DATA_ROOT / "data/processed/delhi_multistation_forecast.parquet"
    available_horizons = [
        horizon
        for horizon in (1, 3, 6, 12, 24)
        if (DATA_ROOT / f"data/runs/multistation_xgboost/model_{horizon}h.joblib").exists()
    ]
    return {
        "status": "ok",
        "aqi_engine": "available",
        "station_summary": summary.exists(),
        "multistation_dataset": multistation_dataset.exists(),
        "multistation_xgboost_horizons": available_horizons,
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


def _multistation_feature_row(row: pd.Series, station_id: str, feature_columns: list[str]) -> pd.DataFrame:
    values = {}
    for column in feature_columns:
        if column.startswith("station_"):
            values[column] = float(column == f"station_{station_id}")
        else:
            values[column] = row.get(column)
    return pd.DataFrame([values], columns=feature_columns)


def _multistation_forecast(station_id: str, horizon: int) -> dict[str, Any] | None:
    dataset = DATA_ROOT / "data/processed/delhi_multistation_forecast.parquet"
    model_path = DATA_ROOT / f"data/runs/multistation_xgboost/model_{horizon}h.joblib"
    if not dataset.exists() or not model_path.exists():
        return None
    try:
        frame = pd.read_parquet(dataset, filters=[("station_id", "=", station_id)])
    except (OSError, ValueError):
        frame = pd.read_parquet(dataset)
        frame = frame[frame["station_id"] == station_id]
    if frame.empty:
        raise HTTPException(status_code=404, detail=f"Unknown station: {station_id}")
    usable = frame.dropna(subset=["pm25_current"]).sort_values("timestamp_utc")
    if usable.empty:
        raise HTTPException(status_code=503, detail="No current PM2.5 feature row is available")
    latest = usable.iloc[-1]
    bundle = joblib.load(model_path)
    features = bundle["feature_columns"]
    vector = _multistation_feature_row(latest, station_id, features)
    prediction = max(0.0, float(bundle["model"].predict(vector)[0]))
    issued_at = pd.Timestamp(latest["timestamp_utc"])
    return {
        "station_id": station_id,
        "as_of_utc": issued_at.isoformat(),
        "target_timestamp_utc": (issued_at + pd.Timedelta(hours=horizon)).isoformat(),
        "current_pm25": float(latest["pm25_current"]),
        "forecast_pm25": prediction,
        "horizon_hours": horizon,
        "model": "global_xgboost",
        "model_artifact": str(model_path),
        "feature_count": len(features),
        "quality_status": "historical_artifact",
        "source_status": "offline_historical_panel",
    }


@app.get("/forecast/{station_id}")
def forecast(station_id: str, horizon: int = 1) -> dict[str, Any]:
    if horizon not in {1, 3, 6, 12, 24}:
        raise HTTPException(status_code=400, detail="horizon must be one of 1, 3, 6, 12, or 24 hours")
    multistation = _multistation_forecast(station_id, horizon)
    if multistation is not None:
        return multistation
    if horizon != 1:
        raise HTTPException(
            status_code=503,
            detail=f"Multi-station {horizon}h forecast artifact is unavailable; train it first",
        )
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
        "horizon_hours": 1,
        "quality_status": "historical_artifact",
        "source_status": "offline_station_panel",
    }


@app.get("/forecast/{station_id}/path")
def forecast_path(station_id: str) -> dict[str, Any]:
    forecasts = []
    for horizon in (1, 3, 6, 12, 24):
        result = _multistation_forecast(station_id, horizon)
        if result is not None:
            forecasts.append(result)
    if not forecasts:
        raise HTTPException(status_code=503, detail="No multi-horizon forecast artifacts are available")
    return {
        "station_id": station_id,
        "quality_status": forecasts[0]["quality_status"],
        "source_status": forecasts[0]["source_status"],
        "forecasts": forecasts,
    }
