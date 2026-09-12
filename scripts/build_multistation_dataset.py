#!/usr/bin/env python3
"""Build a point-in-time, multi-station PM2.5 forecasting feature table."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_STATION_DIR = Path("data/processed/cpcb_2024_25_hourly")
DEFAULT_REGISTRY = Path("data/processed/cpcb_panel_quality/station_registry.csv")
DEFAULT_OUTPUT = Path("data/processed/delhi_multistation_forecast.parquet")
DEFAULT_MANIFEST = Path("data/processed/delhi_multistation_forecast.manifest.json")

POLLUTION_COLUMNS = ("pm10", "no2", "so2", "co", "o3", "nh3")
WEATHER_COLUMNS = (
    "temperature_c",
    "relative_humidity_pct",
    "wind_speed_mps",
    "wind_direction_deg",
    "rain_mm",
)
LAGS = (1, 3, 6, 12, 24, 48, 72)
ROLLING_WINDOWS = (3, 6, 12, 24, 72)
HORIZONS = (1, 3, 6, 12, 24)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_panel(station_dir: Path) -> pd.DataFrame:
    paths = sorted(station_dir.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no station Parquets found under {station_dir}")
    columns = ["station_id", "station_name", "timestamp_utc", "pm25", *POLLUTION_COLUMNS, *WEATHER_COLUMNS]
    frames = []
    for path in paths:
        frame = pd.read_parquet(path, columns=columns)
        frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce")
        frames.append(frame)
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.dropna(subset=["station_id", "timestamp_utc"])
    for column in ["pm25", *POLLUTION_COLUMNS, *WEATHER_COLUMNS]:
        panel[column] = pd.to_numeric(panel[column], errors="coerce")
    return panel.sort_values(["station_id", "timestamp_utc"]).reset_index(drop=True)


def _add_temporal_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    grouped = frame.groupby("station_id", sort=False)
    frame["pm25_current"] = frame["pm25"]
    for lag in LAGS:
        frame[f"pm25_lag_{lag}h"] = grouped["pm25"].shift(lag)
    for window in ROLLING_WINDOWS:
        rolling = grouped["pm25"].rolling(window, min_periods=max(1, window // 2))
        frame[f"pm25_rolling_mean_{window}h"] = rolling.mean().reset_index(level=0, drop=True)
        frame[f"pm25_rolling_std_{window}h"] = rolling.std().reset_index(level=0, drop=True)
    for column in (*POLLUTION_COLUMNS, *WEATHER_COLUMNS):
        frame[f"{column}_current"] = frame[column]
        for lag in (1, 3, 6, 24):
            frame[f"{column}_lag_{lag}h"] = grouped[column].shift(lag)

    timestamp = frame["timestamp_utc"]
    frame["hour_sin"] = np.sin(2 * np.pi * timestamp.dt.hour / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * timestamp.dt.hour / 24)
    frame["weekday_sin"] = np.sin(2 * np.pi * timestamp.dt.dayofweek / 7)
    frame["weekday_cos"] = np.cos(2 * np.pi * timestamp.dt.dayofweek / 7)
    frame["month_sin"] = np.sin(2 * np.pi * (timestamp.dt.month - 1) / 12)
    frame["month_cos"] = np.cos(2 * np.pi * (timestamp.dt.month - 1) / 12)
    frame["is_weekend"] = (timestamp.dt.dayofweek >= 5).astype(float)

    for horizon in HORIZONS:
        frame[f"target_pm25_{horizon}h"] = grouped["pm25"].shift(-horizon)
    frame["target_timestamp_utc"] = frame["timestamp_utc"] + pd.Timedelta(hours=max(HORIZONS))
    return frame


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    value = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return float(2 * radius * np.arcsin(np.sqrt(value)))


def _nearest_stations(registry: pd.DataFrame, station_ids: list[str], k: int) -> dict[str, list[str]]:
    coordinates = registry.set_index("station_id")
    result: dict[str, list[str]] = {}
    for station_id in station_ids:
        if station_id not in coordinates.index:
            result[station_id] = []
            continue
        row = coordinates.loc[station_id]
        distances = []
        for other in station_ids:
            if other == station_id or other not in coordinates.index:
                continue
            candidate = coordinates.loc[other]
            if pd.isna(row.latitude) or pd.isna(row.longitude) or pd.isna(candidate.latitude) or pd.isna(candidate.longitude):
                continue
            distances.append((_haversine_km(row.latitude, row.longitude, candidate.latitude, candidate.longitude), other))
        result[station_id] = [other for _, other in sorted(distances)[:k]]
    return result


def _add_neighbour_features(
    frame: pd.DataFrame, registry_path: Path, neighbour_count: int = 5
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    frame = frame.copy()
    registry = pd.read_csv(registry_path) if registry_path.exists() else pd.DataFrame()
    station_ids = sorted(frame["station_id"].astype(str).unique())
    if registry.empty or not {"station_id", "latitude", "longitude"}.issubset(registry.columns):
        nearest = {station_id: [] for station_id in station_ids}
    else:
        nearest = _nearest_stations(registry, station_ids, neighbour_count)

    pivot = frame.pivot_table(index="timestamp_utc", columns="station_id", values="pm25", aggfunc="first")
    feature_frames = []
    for station_id in station_ids:
        neighbours = nearest.get(station_id, [])
        if neighbours:
            values = pivot.reindex(columns=neighbours)
            features = pd.DataFrame(
                {
                    "timestamp_utc": values.index,
                    "station_id": station_id,
                    "neighbor_pm25_mean_current": values.mean(axis=1),
                    "neighbor_pm25_max_current": values.max(axis=1),
                    "neighbor_pm25_std_current": values.std(axis=1),
                    "neighbor_pm25_available_count": values.notna().sum(axis=1),
                }
            )
        else:
            features = pd.DataFrame(
                {
                    "timestamp_utc": pivot.index,
                    "station_id": station_id,
                    "neighbor_pm25_mean_current": np.nan,
                    "neighbor_pm25_max_current": np.nan,
                    "neighbor_pm25_std_current": np.nan,
                    "neighbor_pm25_available_count": 0,
                }
            )
        feature_frames.append(features)
    neighbour_frame = pd.concat(feature_frames, ignore_index=True)
    frame = frame.merge(neighbour_frame, on=["timestamp_utc", "station_id"], how="left", validate="one_to_one")
    return frame, nearest


def build_dataset(
    station_dir: Path = DEFAULT_STATION_DIR,
    registry_path: Path = DEFAULT_REGISTRY,
    output_path: Path = DEFAULT_OUTPUT,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> dict[str, object]:
    panel = _load_panel(station_dir)
    features = _add_temporal_features(panel)
    features, nearest = _add_neighbour_features(features, registry_path)
    raw_columns = {"pm25", *POLLUTION_COLUMNS, *WEATHER_COLUMNS}
    features = features.drop(columns=sorted(raw_columns - {"pm25"}))
    features = features.sort_values(["station_id", "timestamp_utc"]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)
    manifest: dict[str, object] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "station_source": str(station_dir),
        "registry_source": str(registry_path),
        "point_in_time_policy": "current and lagged observations only; targets are shifted forward",
        "station_count": int(features["station_id"].nunique()),
        "rows_written": int(len(features)),
        "horizons_hours": list(HORIZONS),
        "feature_columns": [column for column in features.columns if not column.startswith("target_")],
        "target_columns": [f"target_pm25_{horizon}h" for horizon in HORIZONS],
        "neighbor_map": nearest,
        "start_utc": features["timestamp_utc"].min().isoformat(),
        "end_utc": features["timestamp_utc"].max().isoformat(),
        "output": {
            "path": str(output_path),
            "bytes": output_path.stat().st_size,
            "sha256": _sha256(output_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station-dir", type=Path, default=DEFAULT_STATION_DIR)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    print(json.dumps(build_dataset(args.station_dir, args.registry, args.output, args.manifest), indent=2))


if __name__ == "__main__":
    main()
