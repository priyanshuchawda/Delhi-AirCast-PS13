#!/usr/bin/env python3
"""Train a compact, two-station PM2.5-index forecaster for WAQI live inputs.

This is deliberately separate from the CPCB-concentration XGBoost artifacts.
WAQI's ``iaqi.pm25`` field is an index value, so the target here is an explicitly
labelled PM2.5 AQI proxy derived from CPCB PM2.5 using the legacy US-EPA
breakpoints documented by WAQI. It is not a CPCB composite AQI forecast.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


DEFAULT_INPUT = Path("data/processed/cpcb_2024_25_hourly")
DEFAULT_WEATHER = Path("data/raw/weather/delhi-central-hourly-2017-2025.json")
DEFAULT_OUTPUT = Path("models/waqi_live_index")
STATIONS = {"site_107": "Pusa", "site_124": "R.K. Puram"}
HORIZONS = (1, 3, 6, 12, 24)
TRAIN_END = pd.Timestamp("2025-07-01", tz="UTC")
VALIDATION_END = pd.Timestamp("2025-10-01", tz="UTC")

# WAQI documents its AQI display as following the US-EPA scale. These legacy
# PM2.5 breakpoints are used only to derive a comparable model target; they do
# not turn the result into an official CPCB AQI or regulatory exposure metric.
EPA_PM25_BREAKPOINTS = (
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
)


def pm25_to_us_aqi_proxy(value: float | int | None) -> float:
    """Interpolate the legacy US-EPA PM2.5 index, preserving missing values."""
    if value is None or pd.isna(value) or not np.isfinite(float(value)):
        return float("nan")
    concentration = max(0.0, min(500.4, np.floor(float(value) * 10) / 10))
    for low_c, high_c, low_i, high_i in EPA_PM25_BREAKPOINTS:
        if low_c <= concentration <= high_c:
            return float(round((high_i - low_i) / (high_c - low_c) * (concentration - low_c) + low_i))
    return 500.0


def _load(input_dir: Path) -> pd.DataFrame:
    parts = []
    for station_id in STATIONS:
        path = input_dir / f"{station_id}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Required CPCB station panel is missing: {path}")
        frame = pd.read_parquet(path, columns=["station_id", "timestamp_utc", "pm25"])
        parts.append(frame)
    data = pd.concat(parts, ignore_index=True)
    data["timestamp_utc"] = pd.to_datetime(data["timestamp_utc"], utc=True, errors="coerce")
    data["pm25"] = pd.to_numeric(data["pm25"], errors="coerce")
    data = data.dropna(subset=["timestamp_utc", "pm25"])
    data = data[data["pm25"].between(0, 1000)].copy()
    data["station_id"] = data["station_id"].astype(str)
    return data.sort_values(["station_id", "timestamp_utc"]).reset_index(drop=True)


def _features(frame: pd.DataFrame) -> pd.DataFrame:
    local = frame["timestamp_utc"].dt.tz_convert("Asia/Kolkata")
    features = pd.DataFrame(index=frame.index)
    features["current_pm25_index"] = frame["pm25"].map(pm25_to_us_aqi_proxy)
    features["hour_sin"] = np.sin(2 * np.pi * local.dt.hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * local.dt.hour / 24)
    features["weekday_sin"] = np.sin(2 * np.pi * local.dt.dayofweek / 7)
    features["weekday_cos"] = np.cos(2 * np.pi * local.dt.dayofweek / 7)
    features["year_day_sin"] = np.sin(2 * np.pi * local.dt.dayofyear / 365.25)
    features["year_day_cos"] = np.cos(2 * np.pi * local.dt.dayofyear / 365.25)
    for station_id in STATIONS:
        features[f"station_{station_id}"] = frame["station_id"].eq(station_id).astype(float)
    return features


def _history_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create issue-time-only temporal and two-site context features.

    Inputs are aligned on the hourly clock. Every lag/rolling feature is shifted
    so it contains observations available at or before the issue time; no
    future target value can leak into the feature row.
    """
    base = frame[["station_id", "timestamp_utc", "pm25"]].copy()
    base["current_pm25_index"] = base["pm25"].map(pm25_to_us_aqi_proxy)
    base = base.sort_values(["station_id", "timestamp_utc"]).reset_index(drop=True)
    pieces = []
    for station_id, group in base.groupby("station_id", sort=False):
        group = group.copy()
        index = group["current_pm25_index"]
        for lag in (1, 3, 6, 12, 24, 48, 72):
            group[f"lag_{lag}h"] = index.shift(lag)
        for window in (3, 6, 12, 24):
            past = index.shift(1)
            group[f"mean_{window}h"] = past.rolling(window, min_periods=max(1, window // 2)).mean()
            group[f"std_{window}h"] = past.rolling(window, min_periods=max(2, window // 2)).std()
        pieces.append(group)
    result = pd.concat(pieces, ignore_index=True).sort_values(["timestamp_utc", "station_id"])

    # Point-in-time neighbor context. Align only same timestamps and use the
    # other station's contemporaneous observation, which is also available at
    # issue time (unlike its future label).
    pivot = result.pivot(index="timestamp_utc", columns="station_id", values="current_pm25_index")
    for station_id, other_id in (("site_107", "site_124"), ("site_124", "site_107")):
        other = pivot[other_id] if other_id in pivot else pd.Series(dtype=float)
        result.loc[result.station_id.eq(station_id), "neighbor_current_index"] = (
            result.loc[result.station_id.eq(station_id), "timestamp_utc"].map(other)
        )
    result["neighbor_minus_current"] = result["neighbor_current_index"] - result["current_pm25_index"]
    features = _features(result)
    history_columns = [column for column in result if column.startswith(("lag_", "mean_", "std_"))]
    features = pd.concat([features, result[history_columns + ["neighbor_current_index", "neighbor_minus_current"]]], axis=1)
    # Return rows in the caller's original order so horizon labels align.
    ordered_index = frame.sort_values(["timestamp_utc", "station_id"]).index
    return features.set_axis(ordered_index).reindex(frame.index)


def _load_issue_weather(path: Path) -> pd.DataFrame:
    """Load hourly Open-Meteo weather with the same UTC/units as the main panel."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    hourly = payload["hourly"]
    weather = pd.DataFrame({
        "timestamp_utc": pd.to_datetime(hourly["time"], errors="coerce")
        .tz_localize("Asia/Kolkata").tz_convert("UTC"),
        "temperature_2m": pd.to_numeric(hourly.get("temperature_2m"), errors="coerce"),
        "relative_humidity_2m": pd.to_numeric(hourly.get("relative_humidity_2m"), errors="coerce"),
        "precipitation": pd.to_numeric(hourly.get("precipitation"), errors="coerce"),
        "wind_speed_10m": pd.to_numeric(hourly.get("wind_speed_10m"), errors="coerce") / 3.6,
        "wind_direction_10m": pd.to_numeric(hourly.get("wind_direction_10m"), errors="coerce"),
        "surface_pressure": pd.to_numeric(hourly.get("surface_pressure"), errors="coerce"),
        "cloud_cover": pd.to_numeric(hourly.get("cloud_cover"), errors="coerce"),
    }).dropna(subset=["timestamp_utc"]).drop_duplicates("timestamp_utc").sort_values("timestamp_utc")
    radians = np.deg2rad(weather["wind_direction_10m"])
    weather["wind_u"] = -weather["wind_speed_10m"] * np.sin(radians)
    weather["wind_v"] = -weather["wind_speed_10m"] * np.cos(radians)
    return weather[[
        "timestamp_utc", "temperature_2m", "relative_humidity_2m", "precipitation",
        "wind_u", "wind_v", "surface_pressure", "cloud_cover",
    ]]


def _asof_weather_features(frame: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Join only the latest weather timestamp <= issue time (never future weather)."""
    left = frame[["timestamp_utc"]].copy().reset_index(names="row_id").sort_values("timestamp_utc")
    joined = pd.merge_asof(left, weather.sort_values("timestamp_utc"), on="timestamp_utc", direction="backward")
    columns = [column for column in weather.columns if column != "timestamp_utc"]
    joined = joined.set_index("row_id").reindex(frame.index)
    return joined[columns].rename(columns={column: f"wx_{column}" for column in columns})


def _metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae_index_points": float(mean_absolute_error(actual, predicted)),
        "rmse_index_points": float(mean_squared_error(actual, predicted) ** 0.5),
    }


def _weather_candidate_promoted(
    baseline: dict[str, dict[str, float]],
    candidate: dict[str, dict[str, float]],
) -> bool:
    """Require validation improvement and at least 1% test MAE improvement."""
    return (
        candidate["validation"]["mae_index_points"] < baseline["validation"]["mae_index_points"]
        and candidate["test"]["mae_index_points"] <= 0.99 * baseline["test"]["mae_index_points"]
    )


def train(input_dir: Path, output_dir: Path, weather_path: Path | None = DEFAULT_WEATHER) -> dict[str, object]:
    frame = _load(input_dir)
    frame["target_pm25_index"] = frame["pm25"].map(pm25_to_us_aqi_proxy)
    grouped = frame.groupby("station_id", sort=False)
    features = _history_features(frame)
    weather = _load_issue_weather(weather_path) if weather_path and weather_path.exists() else None
    weather_features = _asof_weather_features(frame, weather) if weather is not None else None
    weather_candidate_features = (
        pd.concat([features, weather_features], axis=1) if weather_features is not None else None
    )
    results: dict[str, object] = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    for horizon in HORIZONS:
        target = grouped["target_pm25_index"].shift(-horizon)
        # Eliminate gaps: a row is eligible only when issue and target are
        # exactly the requested number of hours apart for this same station.
        target_time = grouped["timestamp_utc"].shift(-horizon)
        exact = (target_time - frame["timestamp_utc"]) == pd.Timedelta(hours=horizon)
        usable = target.notna() & exact
        train_mask = usable & (frame["timestamp_utc"] < TRAIN_END)
        valid_mask = usable & frame["timestamp_utc"].between(TRAIN_END, VALIDATION_END, inclusive="left")
        test_mask = usable & (frame["timestamp_utc"] >= VALIDATION_END)
        if not train_mask.any() or not valid_mask.any() or not test_mask.any():
            raise ValueError(f"Not enough rows for chronological split at {horizon}h")

        model = HistGradientBoostingRegressor(
            max_iter=180, learning_rate=0.06, max_leaf_nodes=15,
            l2_regularization=2.0, random_state=42,
        )
        # Compare the incumbent's single-observation feature set with the
        # history/spatial challenger on identical chronological windows.
        baseline_features = _features(frame)
        baseline_model = HistGradientBoostingRegressor(
            max_iter=180, learning_rate=0.06, max_leaf_nodes=15,
            l2_regularization=2.0, random_state=42,
        )
        baseline_model.fit(baseline_features.loc[train_mask], target.loc[train_mask])
        model.fit(features.loc[train_mask], target.loc[train_mask])
        weather_model = None
        weather_scores = None
        if weather_candidate_features is not None:
            weather_model = HistGradientBoostingRegressor(
                max_iter=180, learning_rate=0.06, max_leaf_nodes=15,
                l2_regularization=2.0, random_state=42,
            )
            weather_model.fit(weather_candidate_features.loc[train_mask], target.loc[train_mask])
            weather_scores = {}
            for split, mask in (("validation", valid_mask), ("test", test_mask)):
                weather_prediction = np.clip(weather_model.predict(weather_candidate_features.loc[mask]), 0, 500)
                weather_scores[split] = _metrics(target.loc[mask], weather_prediction)
        scores = {}
        for split, mask in (("validation", valid_mask), ("test", test_mask)):
            prediction = np.clip(model.predict(features.loc[mask]), 0, 500)
            baseline_prediction = np.clip(baseline_model.predict(baseline_features.loc[mask]), 0, 500)
            actual = target.loc[mask]
            persistence = features.loc[mask, "current_pm25_index"]
            scores[split] = {
                "rows": int(mask.sum()),
                "model": _metrics(actual, prediction),
                "history_spatial_challenger": _metrics(actual, prediction),
                "single_observation_incumbent": _metrics(actual, baseline_prediction),
                "persistence": _metrics(actual, persistence.to_numpy()),
            }
            if weather_model is not None:
                scores[split]["weather_candidate"] = weather_scores[split]
        if weather_model is not None:
            baseline_scores = {
                split: scores[split]["history_spatial_challenger"]
                for split in ("validation", "test")
            }
            weather_promoted = _weather_candidate_promoted(baseline_scores, weather_scores)
            import joblib
            joblib.dump({
                "model": weather_model,
                "feature_columns": list(weather_candidate_features.columns),
                "station_ids": list(STATIONS),
                "horizon_hours": horizon,
                "target": "legacy_us_epa_pm25_index_proxy",
                "feature_version": "history_spatial_weather_v1",
                "promotion_eligible": weather_promoted,
                "weather_source": str(weather_path),
                "source": "CPCB historical station PM2.5 + issue-time Open-Meteo weather",
            }, output_dir / f"weather_model_{horizon}h.joblib")
            scores["weather_promotion_eligible"] = weather_promoted
        else:
            scores["weather_promotion_eligible"] = False
        results[str(horizon)] = scores
        import joblib

        joblib.dump(
            {
                "model": model,
                "feature_columns": list(features.columns),
                "station_ids": list(STATIONS),
                "horizon_hours": horizon,
                "target": "legacy_us_epa_pm25_index_proxy",
                "target_breakpoints": EPA_PM25_BREAKPOINTS,
                "source": "CPCB historical station PM2.5; site_107 and site_124",
                "feature_version": "history_spatial_v1",
            },
            output_dir / f"model_{horizon}h.joblib",
        )
        joblib.dump(
            {
                "model": baseline_model,
                "feature_columns": list(baseline_features.columns),
                "station_ids": list(STATIONS),
                "horizon_hours": horizon,
                "target": "legacy_us_epa_pm25_index_proxy",
                "feature_version": "single_observation_v1",
                "source": "CPCB historical station PM2.5; site_107 and site_124",
            },
            output_dir / f"incumbent_model_{horizon}h.joblib",
        )

    report = {
        "model_family": "HistGradientBoostingRegressor",
        "stations": STATIONS,
        "training_period_end_utc": TRAIN_END.isoformat(),
        "validation_period_utc": [TRAIN_END.isoformat(), VALIDATION_END.isoformat()],
        "test_period_start_utc": VALIDATION_END.isoformat(),
        "feature_columns": list(features.columns),
        "target": "PM2.5-only legacy US-EPA AQI index proxy (not CPCB composite AQI)",
        "input_requirements_live": ["recent WAQI iaqi.pm25 history (up to 72h)", "same-time other pilot-site WAQI index when available", "local timestamp", "mapped station"],
        "feature_version": "history_spatial_v1",
        "incumbent_artifacts": "incumbent_model_{horizon}h.joblib",
        "weather_candidate_feature_columns": list(weather_candidate_features.columns) if weather_candidate_features is not None else [],
        "weather_policy": "latest Open-Meteo historical weather timestamp at or before issue time; no future weather values",
        "weather_promotion_gate": "validation MAE must improve and test MAE must improve by at least 1% over history/spatial model",
        "results": results,
    }
    (output_dir / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--weather", type=Path, default=DEFAULT_WEATHER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = train(args.input_dir, args.output_dir, args.weather)
    for horizon, result in report["results"].items():
        score = result["test"]
        print(
            f"{horizon}h test: model MAE={score['model']['mae_index_points']:.2f}; "
            f"persistence MAE={score['persistence']['mae_index_points']:.2f}; rows={score['rows']}"
        )


if __name__ == "__main__":
    main()
