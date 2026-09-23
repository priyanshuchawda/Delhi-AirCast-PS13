"""Small local API for AQI calculations and trained forecast artifacts."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict

from .aqi import calculate_aqi, pm25_proxy


DATA_ROOT = Path(os.environ.get("DELHI_AIRCAST_DATA_ROOT", "."))
app = FastAPI(title="Delhi AirCast", version="0.1.0")
origins = [origin.strip() for origin in os.environ.get("DELHI_AIRCAST_ALLOWED_ORIGINS", "http://localhost:3000").split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


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


def _horizon_artifact_path(horizon: int, run_dir: str) -> Path:
    return DATA_ROOT / f"data/runs/{run_dir}/model_{horizon}h.joblib"


@lru_cache(maxsize=16)
def _load_bundle(path: str) -> dict[str, Any]:
    return joblib.load(path)


def _serving_features_path() -> Path:
    return DATA_ROOT / "data/processed/delhi_serving_features.parquet"


@lru_cache(maxsize=1)
def _latest_features() -> pd.DataFrame:
    """Load compact deployment rows; fall back to the offline panel locally."""
    compact = _serving_features_path()
    if compact.exists():
        frame = pd.read_parquet(compact)
    else:
        panel = DATA_ROOT / "data/processed/delhi_unified_forecast.parquet"
        if not panel.exists():
            raise HTTPException(status_code=503, detail="Serving features are unavailable; build the offline bundle first")
        frame = pd.read_parquet(panel).sort_values("timestamp_utc")
        frame = frame.dropna(subset=["pm25_current"]).groupby("station_id", sort=False, as_index=False).tail(1)
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    return frame.reset_index(drop=True)


def _resolve_model_bundle(horizon: int) -> tuple[dict[str, Any], Path] | None:
    """Prefer the final unified-feature artifact, then the multi-station baseline."""
    for run_dir in ("final_xgboost", "multistation_xgboost"):
        path = _horizon_artifact_path(horizon, run_dir)
        if path.exists():
            return _load_bundle(str(path)), path
    return None


def _forecast_dataset(feature_set: str) -> Path:
    if feature_set == "unified-spatial-temporal":
        unified = DATA_ROOT / "data/processed/delhi_unified_forecast.parquet"
        if unified.exists():
            return unified
    return DATA_ROOT / "data/processed/delhi_multistation_forecast.parquet"


@app.get("/health")
def health() -> dict[str, Any]:
    summary = _station_summary_path()
    multistation_dataset = DATA_ROOT / "data/processed/delhi_multistation_forecast.parquet"
    unified_dataset = DATA_ROOT / "data/processed/delhi_unified_forecast.parquet"
    serving_features = _serving_features_path()

    def horizons(run_dir: str) -> list[int]:
        return [h for h in (1, 3, 6, 12, 24) if _horizon_artifact_path(h, run_dir).exists()]

    return {
        "status": "ok",
        "aqi_engine": "available",
        "station_summary": summary.exists() or (DATA_ROOT / "data/processed/cpcb_panel_quality/station_registry.csv").exists(),
        "multistation_dataset": multistation_dataset.exists(),
        "unified_dataset": unified_dataset.exists(),
        "serving_features": serving_features.exists(),
        "multistation_xgboost_horizons": horizons("multistation_xgboost"),
        "final_xgboost_horizons": horizons("final_xgboost"),
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
    path = DATA_ROOT / "data/processed/cpcb_panel_quality/station_registry.csv"
    if not path.exists():
        path = _station_summary_path()
    if not path.exists():
        raise HTTPException(status_code=503, detail="Station metadata unavailable; build the serving bundle")
    frame = pd.read_csv(path)
    try:
        latest = _latest_features()
        frame = frame.merge(latest[["station_id", "station_name", "timestamp_utc", "pm25_current"]], on="station_id", how="left", suffixes=("", "_latest"))
        if "station_name_latest" in frame:
            frame["station_name"] = frame["station_name"].fillna(frame["station_name_latest"])
            frame = frame.drop(columns="station_name_latest")
        frame["as_of_utc"] = frame["timestamp_utc"].astype(str)
        frame = frame.drop(columns="timestamp_utc")
        frame["current_aqi_proxy"] = frame["pm25_current"].map(lambda value: pm25_proxy(value).value if pd.notna(value) else None)
    except HTTPException:
        pass
    return {"count": int(len(frame)), "stations": frame.to_dict(orient="records")}


def _multistation_feature_row(row: pd.Series, station_id: str, feature_columns: list[str]) -> pd.DataFrame:
    values = {}
    for column in feature_columns:
        if column.startswith("station_"):
            values[column] = float(column == f"station_{station_id}")
        else:
            values[column] = row.get(column)
    return pd.DataFrame([values], columns=feature_columns)


def _forecast_rows(horizon: int) -> list[dict[str, Any]]:
    resolved = _resolve_model_bundle(horizon)
    if resolved is None:
        raise HTTPException(status_code=503, detail=f"No XGBoost model artifact for {horizon}h")
    bundle, model_path = resolved
    rows = _latest_features()
    station_ids = rows["station_id"].astype(str).tolist()
    matrix = pd.concat(
        [_multistation_feature_row(row, station_id, bundle["feature_columns"]) for (_, row), station_id in zip(rows.iterrows(), station_ids)],
        ignore_index=True,
    ).apply(pd.to_numeric, errors="coerce")
    predictions = np.maximum(0.0, bundle["model"].predict(matrix))

    composite_by_station: dict[str, tuple[dict[str, float], Any]] = {}
    if horizon == 6:
        pollutant_paths = {p: DATA_ROOT / f"data/runs/multipollutant_aqi_final/{p}_6h.joblib" for p in ("pm10", "no2", "co", "o3")}
        if all(path.exists() for path in pollutant_paths.values()):
            forecast_values: dict[str, np.ndarray] = {"pm25": predictions}
            for pollutant, path in pollutant_paths.items():
                pollutant_bundle = _load_bundle(str(path))
                x = pd.concat(
                    [_multistation_feature_row(row, station_id, pollutant_bundle["feature_columns"]) for (_, row), station_id in zip(rows.iterrows(), station_ids)],
                    ignore_index=True,
                ).apply(pd.to_numeric, errors="coerce")
                forecast_values[pollutant] = np.maximum(0.0, pollutant_bundle["model"].predict(x))
            for i, station_id in enumerate(station_ids):
                forecast_pollutants = {p: float(values[i]) for p, values in forecast_values.items()}
                composite_by_station[station_id] = (forecast_pollutants, calculate_aqi(forecast_pollutants))

    registry_path = DATA_ROOT / "data/processed/cpcb_panel_quality/station_registry.csv"
    if registry_path.exists():
        registry = pd.read_csv(registry_path)
    else:
        registry = pd.DataFrame(columns=["station_id", "latitude", "longitude"])
    coordinates = registry.set_index("station_id").to_dict(orient="index") if not registry.empty else {}
    result = []
    for i, row in rows.iterrows():
        station_id = str(row["station_id"])
        predicted = float(predictions[i])
        proxy = pm25_proxy(predicted)
        current_pollutants = {p: row.get(f"{p}_current") for p in ("pm25", "pm10", "no2", "so2", "o3", "co", "nh3")}
        current_aqi = calculate_aqi(current_pollutants)
        forecast_pollutants, forecast_aqi = composite_by_station.get(station_id, ({"pm25": predicted}, proxy))
        coords = coordinates.get(station_id, {})
        issued = pd.Timestamp(row["timestamp_utc"]).tz_convert("UTC")
        result.append({
            "station_id": station_id,
            "station_name": row.get("station_name") or station_id,
            "latitude": coords.get("latitude"),
            "longitude": coords.get("longitude"),
            "as_of_utc": issued.isoformat(),
            "target_timestamp_utc": (issued + pd.Timedelta(hours=horizon)).isoformat(),
            "current_pm25": float(row["pm25_current"]),
            "forecast_pm25": predicted,
            "current_aqi": current_aqi.value if current_aqi.value is not None else pm25_proxy(row["pm25_current"]).value,
            "current_aqi_status": current_aqi.status if current_aqi.value is not None else "pm25_derived_proxy",
            "forecast_aqi": forecast_aqi.value,
            "forecast_aqi_category": forecast_aqi.category,
            "forecast_aqi_status": "cpcb_five_pollutant_subset" if station_id in composite_by_station else "pm25_derived_proxy",
            "forecast_pollutants": forecast_pollutants,
            "horizon_hours": horizon,
            "model": "xgboost",
            "quality_status": "historical_artifact",
            "source_status": "offline_historical_panel",
        })
    return sorted(result, key=lambda station: station["station_name"].casefold())


def _multistation_forecast(station_id: str, horizon: int) -> dict[str, Any] | None:
    matches = [row for row in _forecast_rows(horizon) if row["station_id"] == str(station_id)]
    if not matches:
        raise HTTPException(status_code=404, detail=f"Unknown station: {station_id}")
    return matches[0]


@app.get("/stations/forecast")
def stations_forecast(horizon: int = 6) -> dict[str, Any]:
    if horizon not in {1, 3, 6, 12, 24}:
        raise HTTPException(status_code=400, detail="horizon must be one of 1, 3, 6, 12, or 24 hours")
    forecasts = _forecast_rows(horizon)
    return {
        "horizon_hours": horizon,
        "count": len(forecasts),
        "as_of_utc": max((row["as_of_utc"] for row in forecasts), default=None),
        "quality_status": "historical_artifact",
        "source_status": "offline_historical_panel",
        "forecasts": forecasts,
    }


@app.get("/stations/{station_id}/history")
def station_history(station_id: str, hours: int = 72) -> dict[str, Any]:
    if not 1 <= hours <= 336:
        raise HTTPException(status_code=400, detail="hours must be between 1 and 336")
    compact_history = DATA_ROOT / "data/processed/delhi_station_history.parquet"
    panel = compact_history if compact_history.exists() else DATA_ROOT / "data/processed/delhi_unified_forecast.parquet"
    if not panel.exists():
        raise HTTPException(status_code=503, detail="Historical panel is unavailable in this serving bundle")
    frame = pd.read_parquet(panel, columns=["station_id", "timestamp_utc", "pm25_current"])
    frame = frame[frame["station_id"].astype(str).eq(station_id)].dropna(subset=["pm25_current"]).sort_values("timestamp_utc").tail(hours)
    if frame.empty:
        raise HTTPException(status_code=404, detail=f"No history for station {station_id}")
    return {"station_id": station_id, "hours": hours, "history": [{"timestamp_utc": pd.Timestamp(row.timestamp_utc).tz_convert("UTC").isoformat(), "pm25": float(row.pm25_current)} for row in frame.itertuples()]}


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
    forecast_aqi = pm25_proxy(prediction)
    return {
        "station_id": station_id,
        "as_of_utc": latest["timestamp_utc"].isoformat(),
        "target_timestamp_utc": latest["target_timestamp_utc"].isoformat(),
        "current_pm25": float(latest["pm25"]),
        "forecast_pm25": prediction,
        "forecast_aqi": forecast_aqi.value,
        "forecast_aqi_category": forecast_aqi.category,
        "forecast_aqi_status": forecast_aqi.status,
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
