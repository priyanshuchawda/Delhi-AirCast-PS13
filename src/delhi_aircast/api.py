"""Small local API for AQI calculations and trained forecast artifacts."""

from __future__ import annotations

import os
import json
import time
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

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


LIVE_WAQI_STATIONS = {
    "site_107": {"name": "Pusa", "uid": 10124, "latitude": 28.639652, "longitude": 77.146275},
    "site_124": {"name": "R.K. Puram", "uid": 2556, "latitude": 28.563262, "longitude": 77.186937},
}
LIVE_MODEL_DIR = DATA_ROOT / "models/waqi_live_index"
WAQI_MAX_AGE_HOURS = 2
_WAQI_CACHE: dict[int, tuple[float, dict[str, Any]]] = {}
_WEATHER_CACHE: dict[str, tuple[float, list[dict[str, Any]]]] = {}
LIVE_HISTORY_PATH = Path(os.environ.get(
    "DELHI_AIRCAST_LIVE_HISTORY",
    str(DATA_ROOT / "data/raw/waqi_live/observations.jsonl"),
))
_LIVE_HISTORY_LOCK = threading.Lock()


def _record_live_observation(uid: int, observation: dict[str, Any]) -> None:
    """Append one newly observed WAQI sample, deduplicated by site and source time."""
    station_id = next((key for key, site in LIVE_WAQI_STATIONS.items() if site["uid"] == uid), None)
    if station_id is None:
        return
    point = {
        "station_id": station_id,
        "waqi_uid": uid,
        "observed_at": pd.Timestamp(observation["observed_at"]).isoformat(),
        "pm25_index": float(observation["pm25_index"]),
        "ingested_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "source": "WAQI station feed",
    }
    with _LIVE_HISTORY_LOCK:
        try:
            if LIVE_HISTORY_PATH.exists():
                for line in reversed(LIVE_HISTORY_PATH.read_text(encoding="utf-8").splitlines()):
                    if not line.strip():
                        continue
                    previous = json.loads(line)
                    if previous.get("station_id") == station_id:
                        if previous.get("observed_at") == point["observed_at"]:
                            return
                        break
            LIVE_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
            with LIVE_HISTORY_PATH.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(point, separators=(",", ":")) + "\n")
        except (OSError, json.JSONDecodeError):
            # Logging is best-effort: it must never take down live serving.
            return


def _waqi_api_key() -> str | None:
    token = os.environ.get("WAQI_API_KEY", "").strip()
    if token:
        return token
    env_file = DATA_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            name, separator, value = line.partition("=")
            if separator and name.strip() == "WAQI_API_KEY":
                return value.strip().strip("\"'") or None
    return None


def _waqi_observation(uid: int, force_refresh: bool = False) -> dict[str, Any]:
    """Fetch/cache one WAQI station feed without ever logging its token."""
    cached = _WAQI_CACHE.get(uid)
    if cached and not force_refresh and time.monotonic() - cached[0] < 300:
        return cached[1]
    token = _waqi_api_key()
    if not token:
        raise HTTPException(status_code=503, detail="WAQI_API_KEY is not configured")
    query = urlencode({"token": token})
    request = Request(
        f"https://api.waqi.info/feed/@{uid}/?{query}",
        headers={"Accept": "application/json", "User-Agent": "Delhi-AirCast/1.0"},
    )
    try:
        with urlopen(request, timeout=12) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, TimeoutError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=502, detail="WAQI station feed could not be reached") from error
    if payload.get("status") != "ok":
        raise HTTPException(status_code=502, detail="WAQI did not return a station observation")
    data = payload.get("data") or {}
    index_value = (data.get("iaqi") or {}).get("pm25", {}).get("v")
    try:
        index_value = float(index_value)
        observed_at = pd.Timestamp(data["time"]["iso"])
        if observed_at.tzinfo is None:
            observed_at = observed_at.tz_localize("Asia/Kolkata")
        observed_at = observed_at.tz_convert("UTC")
    except (TypeError, ValueError, KeyError):
        raise HTTPException(status_code=502, detail="WAQI response is missing PM2.5 index or timestamp")
    if not np.isfinite(index_value) or index_value < 0:
        raise HTTPException(status_code=502, detail="WAQI returned an invalid PM2.5 index")
    observation = {"pm25_index": index_value, "observed_at": observed_at}
    _WAQI_CACHE[uid] = (time.monotonic(), observation)
    _record_live_observation(uid, observation)
    return observation


def _open_meteo_weather(
    station_id: str,
    as_of: pd.Timestamp,
    force_refresh: bool = False,
) -> dict[str, Any] | None:
    """Get cached hourly weather not later than the WAQI sensor issue time."""
    station = LIVE_WAQI_STATIONS.get(station_id)
    if station is None:
        return None
    cached = _WEATHER_CACHE.get(station_id)
    if not cached or force_refresh or time.monotonic() - cached[0] >= 900:
        params = urlencode({
            "latitude": station["latitude"],
            "longitude": station["longitude"],
            "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,wind_direction_10m,surface_pressure,cloud_cover",
            "past_hours": 24,
            "forecast_hours": 1,
            "timezone": "UTC",
        })
        request = Request(
            f"https://api.open-meteo.com/v1/forecast?{params}",
            headers={"Accept": "application/json", "User-Agent": "Delhi-AirCast/1.0"},
        )
        try:
            with urlopen(request, timeout=12) as response:
                payload = json.loads(response.read().decode("utf-8"))
            hourly = payload["hourly"]
            rows = []
            for idx, timestamp in enumerate(hourly["time"]):
                row = {key: values[idx] for key, values in hourly.items() if key != "time"}
                weather_time = pd.Timestamp(timestamp)
                if weather_time.tzinfo is None:
                    weather_time = weather_time.tz_localize("UTC")
                row["timestamp_utc"] = weather_time.tz_convert("UTC")
                rows.append(row)
            _WEATHER_CACHE[station_id] = (time.monotonic(), rows)
        except (OSError, URLError, TimeoutError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None
    else:
        rows = cached[1]

    as_of = pd.Timestamp(as_of).tz_convert("UTC")
    eligible = [row for row in rows if row["timestamp_utc"] <= as_of]
    if not eligible:
        return None
    row = max(eligible, key=lambda value: value["timestamp_utc"])
    age_hours = (as_of - row["timestamp_utc"]).total_seconds() / 3600
    if age_hours < 0 or age_hours > 2:
        return None
    try:
        speed_mps = float(row["wind_speed_10m"]) / 3.6
        direction = np.deg2rad(float(row["wind_direction_10m"]))
        values = {
            "wx_temperature_2m": float(row["temperature_2m"]),
            "wx_relative_humidity_2m": float(row["relative_humidity_2m"]),
            "wx_precipitation": float(row["precipitation"]),
            "wx_wind_u": float(-speed_mps * np.sin(direction)),
            "wx_wind_v": float(-speed_mps * np.cos(direction)),
            "wx_surface_pressure": float(row["surface_pressure"]),
            "wx_cloud_cover": float(row["cloud_cover"]),
        }
        if not all(np.isfinite(value) for value in values.values()):
            return None
    except (TypeError, ValueError, KeyError):
        return None
    return {"values": values, "observed_at": row["timestamp_utc"], "age_minutes": round(age_hours * 60, 1)}


def _live_index_features(
    index_value: float,
    station_id: str,
    observed_at: pd.Timestamp,
    columns: list[str],
    history: list[dict[str, Any]] | None = None,
    neighbor_index: float | None = None,
    weather: dict[str, Any] | None = None,
) -> pd.DataFrame:
    local = observed_at.tz_convert("Asia/Kolkata")
    values = {
        "current_pm25_index": index_value,
        "hour_sin": float(np.sin(2 * np.pi * local.hour / 24)),
        "hour_cos": float(np.cos(2 * np.pi * local.hour / 24)),
        "weekday_sin": float(np.sin(2 * np.pi * local.dayofweek / 7)),
        "weekday_cos": float(np.cos(2 * np.pi * local.dayofweek / 7)),
        "year_day_sin": float(np.sin(2 * np.pi * local.dayofyear / 365.25)),
        "year_day_cos": float(np.cos(2 * np.pi * local.dayofyear / 365.25)),
        **{f"station_{site}": float(site == station_id) for site in LIVE_WAQI_STATIONS},
    }
    if "neighbor_current_index" in columns:
        values["neighbor_current_index"] = neighbor_index
        values["neighbor_minus_current"] = (
            neighbor_index - index_value if neighbor_index is not None else np.nan
        )
    if history:
        by_timestamp = {
            pd.Timestamp(point["observed_at"]).tz_convert("UTC"): float(point["pm25_index"])
            for point in history
        }
        for lag in (1, 3, 6, 12, 24, 48, 72):
            values[f"lag_{lag}h"] = by_timestamp.get(observed_at - pd.Timedelta(hours=lag))
        for window in (3, 6, 12, 24):
            samples = [by_timestamp.get(observed_at - pd.Timedelta(hours=lag)) for lag in range(1, window + 1)]
            valid = np.asarray([value for value in samples if value is not None], dtype=float)
            minimum = max(1, window // 2)
            values[f"mean_{window}h"] = float(np.mean(valid)) if len(valid) >= minimum else np.nan
            values[f"std_{window}h"] = float(np.std(valid, ddof=1)) if len(valid) >= max(2, window // 2) else np.nan
    if weather:
        values.update(weather.get("values", {}))
    return pd.DataFrame([[values.get(column, np.nan) for column in columns],], columns=columns)


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

    live_horizons = [h for h in (1, 3, 6, 12, 24) if (LIVE_MODEL_DIR / f"model_{h}h.joblib").exists()]

    return {
        "status": "ok",
        "aqi_engine": "available",
        "station_summary": summary.exists() or (DATA_ROOT / "data/processed/cpcb_panel_quality/station_registry.csv").exists(),
        "multistation_dataset": multistation_dataset.exists(),
        "unified_dataset": unified_dataset.exists(),
        "serving_features": serving_features.exists(),
        "multistation_xgboost_horizons": horizons("multistation_xgboost"),
        "final_xgboost_horizons": horizons("final_xgboost"),
        "waqi_live_index_horizons": live_horizons,
        "waqi_api_key_configured": _waqi_api_key() is not None,
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
    target_times = [pd.Timestamp(row["target_timestamp_utc"]) for row in forecasts]
    target_is_future = bool(target_times) and max(target_times) > pd.Timestamp.now(tz="UTC")
    return {
        "horizon_hours": horizon,
        "count": len(forecasts),
        "as_of_utc": max((row["as_of_utc"] for row in forecasts), default=None),
        "quality_status": "historical_artifact",
        "source_status": "offline_historical_panel",
        "target_is_future": target_is_future,
        "forecast_mode": "forecast" if target_is_future else "historical_backtest",
        "forecasts": forecasts,
    }


@app.get("/live/stations/forecast")
def live_stations_forecast(horizon: int = 6, refresh: bool = False) -> dict[str, Any]:
    """Issue live WAQI PM2.5-index forecasts for the two validated pilot sites.

    This intentionally does not feed WAQI index values into the existing CPCB
    concentration model. The separately trained pilot predicts an explicitly
    labelled PM2.5-only US-EPA-scale index proxy.
    """
    if horizon not in {1, 3, 6, 12, 24}:
        raise HTTPException(status_code=400, detail="horizon must be one of 1, 3, 6, 12, or 24 hours")
    challenger_path = LIVE_MODEL_DIR / f"model_{horizon}h.joblib"
    incumbent_path = LIVE_MODEL_DIR / f"incumbent_model_{horizon}h.joblib"
    weather_path = LIVE_MODEL_DIR / f"weather_model_{horizon}h.joblib"
    if not challenger_path.exists() and not incumbent_path.exists():
        raise HTTPException(status_code=503, detail="Live-index model artifacts are not installed")
    challenger_bundle = _load_bundle(str(challenger_path)) if challenger_path.exists() else None
    incumbent_bundle = _load_bundle(str(incumbent_path)) if incumbent_path.exists() else None
    weather_bundle = _load_bundle(str(weather_path)) if weather_path.exists() else None
    if weather_bundle and not weather_bundle.get("promotion_eligible", False):
        weather_bundle = None
    issued_at = pd.Timestamp.now(tz="UTC")
    forecasts = []
    observations_by_uid: dict[int, dict[str, Any]] = {}
    for station_id, station in LIVE_WAQI_STATIONS.items():
        try:
            observation = _waqi_observation(station["uid"], force_refresh=refresh)
        except HTTPException as error:
            forecasts.append({
                "station_id": station_id,
                "station_name": station["name"],
                "waqi_uid": station["uid"],
                "quality_status": "unavailable",
                "error": error.detail,
            })
            continue
        observations_by_uid[station["uid"]] = observation
        observed_at = observation["observed_at"]
        age_hours = (issued_at - observed_at).total_seconds() / 3600
        fresh = 0 <= age_hours <= WAQI_MAX_AGE_HOURS
        predicted = None
        model_used = None
        feature_mode = None
        weather_info = None
        if fresh:
            other_station_id = next(site_id for site_id in LIVE_WAQI_STATIONS if site_id != station_id)
            other_uid = LIVE_WAQI_STATIONS[other_station_id]["uid"]
            other_observation = observations_by_uid.get(other_uid) or _WAQI_CACHE.get(other_uid, (0.0, {}))[1]
            neighbor_index = None
            if other_observation:
                neighbor_age = abs((observed_at - other_observation["observed_at"]).total_seconds()) / 3600
                if neighbor_age <= 2:
                    neighbor_index = float(other_observation["pm25_index"])

            # Live snapshots do not provide historical hourly lags. The
            # history challenger has NaN-safe trees, but only use it after a
            # persistent collector has supplied meaningful lag coverage.
            candidate = challenger_bundle
            if candidate and candidate.get("feature_version") == "history_spatial_v1":
                history_path = LIVE_HISTORY_PATH
                history: list[dict[str, Any]] = []
                if history_path.exists():
                    try:
                        history = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                    except (OSError, json.JSONDecodeError):
                        history = []
                site_history = [
                    point for point in history
                    if point.get("station_id") == station_id
                    and 0 <= (observed_at - pd.Timestamp(point["observed_at"]).tz_convert("UTC")).total_seconds() <= 72 * 3600
                ]
                site_history.append({"observed_at": observed_at, "pm25_index": observation["pm25_index"]})
                features = _live_index_features(
                    observation["pm25_index"], station_id, observed_at,
                    candidate["feature_columns"], site_history, neighbor_index,
                )
                lag_count = features[[f"lag_{lag}h" for lag in (1, 3, 6, 12, 24, 48, 72)]].notna().sum(axis=1).iloc[0]
                if lag_count >= 3:
                    if weather_bundle:
                        weather_info = _open_meteo_weather(station_id, observed_at, force_refresh=refresh)
                    if weather_bundle and weather_info:
                        weather_features = _live_index_features(
                            observation["pm25_index"], station_id, observed_at,
                            weather_bundle["feature_columns"], site_history, neighbor_index,
                            weather_info,
                        )
                        predicted = float(np.clip(weather_bundle["model"].predict(weather_features)[0], 0.0, 500.0))
                        model_used, feature_mode = "weather + history/spatial HistGradientBoosting", "live_history_weather"
                    else:
                        predicted = float(np.clip(candidate["model"].predict(features)[0], 0.0, 500.0))
                        model_used, feature_mode = "history-spatial HistGradientBoosting", "live_history"
            if predicted is None and incumbent_bundle:
                features = _live_index_features(
                    observation["pm25_index"], station_id, observed_at,
                    incumbent_bundle["feature_columns"], neighbor_index=neighbor_index,
                )
                predicted = float(np.clip(incumbent_bundle["model"].predict(features)[0], 0.0, 500.0))
                model_used, feature_mode = "single-observation HistGradientBoosting", "history_warming_up"
            elif predicted is None and challenger_bundle:
                features = _live_index_features(
                    observation["pm25_index"], station_id, observed_at,
                    challenger_bundle["feature_columns"], neighbor_index=neighbor_index,
                )
                predicted = float(np.clip(challenger_bundle["model"].predict(features)[0], 0.0, 500.0))
                model_used, feature_mode = "history-spatial HistGradientBoosting", "history_unavailable"
        forecasts.append({
            "station_id": station_id,
            "station_name": station["name"],
            "waqi_uid": station["uid"],
            "as_of_utc": observed_at.isoformat(),
            "age_minutes": round(age_hours * 60, 1),
            "current_pm25_index": observation["pm25_index"],
            "forecast_pm25_index": predicted,
            "horizon_hours": horizon,
            "target_timestamp_utc": (issued_at + pd.Timedelta(hours=horizon)).isoformat(),
            "quality_status": "live" if fresh else "stale",
            "aqi_scale": "WAQI individual PM2.5 index (US-EPA scale)",
            "aqi_semantics": "PM2.5-only index proxy; not CPCB composite AQI",
            "model": model_used,
            "feature_mode": feature_mode,
            "history_max_lag_hours": 72,
            "weather_as_of_utc": weather_info["observed_at"].isoformat() if weather_info else None,
            "weather_age_minutes": weather_info["age_minutes"] if weather_info else None,
        })
    return {
        "issued_at_utc": issued_at.isoformat(),
        "horizon_hours": horizon,
        "source_status": "waqi_station_feed",
        "quality_status": "live" if any(row.get("quality_status") == "live" for row in forecasts) else "unavailable_or_stale",
        "attribution": "World Air Quality Index (WAQI), https://waqi.info/",
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
