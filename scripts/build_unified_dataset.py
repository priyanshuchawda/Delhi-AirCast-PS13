#!/usr/bin/env python3
"""Build the unified, leakage-safe multi-source PM2.5 forecasting table.

The label authority remains the quality-controlled CPCB/OpenCity hourly panel.
Auxiliary sources are joined in strictly point-in-time fashion:

- central weather (Open-Meteo historical): observed values aligned at the issue
  hour ``t``, matching the panel's existing ``*_current`` convention;
- CAMS global air-quality (Open-Meteo/CAMS): because analysis products are
  published several hours after the nominal hour, every CAMS feature is
  *required* to be at least ``CAMS_PUBLISH_LAG`` hours old at issue time;
- FIRMS fire features: daily aggregates are only usable after the day ends, so
  every FIRMS feature refers to complete days strictly before the issue date.

OpenAQ, AirDelhi, legacy AQI, and the CPCB live snapshot are intentionally not
joined here because they do not overlap the 2024-25 CPCB training panel. Their
roles are recorded in the manifest for reviewers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_MULTISTATION = Path("data/processed/delhi_multistation_forecast.parquet")
DEFAULT_WEATHER = Path("data/raw/weather/delhi-central-hourly-2017-2025.json")
DEFAULT_CAMS = Path("data/raw/cams/delhi-central-cams-global-hourly-2023-2025.json")
DEFAULT_FIRMS = Path("data/processed/firms/firms_daily_features.parquet")
DEFAULT_OUTPUT = Path("data/processed/delhi_unified_forecast.parquet")
DEFAULT_MANIFEST = Path("data/processed/delhi_unified_forecast.manifest.json")
DEFAULT_ROLES = Path("data/processed/dataset_roles.json")

CAMS_PUBLISH_LAG_H = 6

WEATHER_RENAME = {
    "temperature_2m": "temperature_c",
    "relative_humidity_2m": "relative_humidity_pct",
    "precipitation": "rain_mm",
    "wind_speed_10m": "wind_speed_mps",
    "wind_direction_10m": "wind_direction_deg",
    "surface_pressure": "pressure_hpa",
    "cloud_cover": "cloud_cover_pct",
}
CAMS_RENAME = {
    "pm2_5": "pm25",
    "pm10": "pm10",
    "nitrogen_dioxide": "no2",
    "ozone": "o3",
    "carbon_monoxide": "co",
    "sulphur_dioxide": "so2",
}

DATASET_ROLES = {
    "CPCB/OpenCity 2024-25 hourly panel": {
        "role": "primary",
        "reason": "Label authority and station-level pollutant/meteorology features; 39 Delhi stations, full 2024-25 overlap.",
    },
    "Open-Meteo central weather (Delhi)": {
        "role": "auxiliary",
        "reason": "Central Delhi observed meteorology aligned at the issue hour; full 2024-25 overlap.",
    },
    "CAMS global air quality (Open-Meteo)": {
        "role": "auxiliary",
        "reason": "Regional chemical composition reanalysis; full 2024-25 overlap, joined with a required 6 h published lag.",
    },
    "NASA FIRMS active fires": {
        "role": "auxiliary",
        "reason": "Daily fire counts/FRP for Delhi and the Punjab/Haryana upwind belt; only completed pre-issue days are used.",
    },
    "OpenAQ Delhi/NCR hourly": {
        "role": "validation-only",
        "reason": "Reference/community sensor readings begin 2025-10-31; the 2024-25 training panel has no overlap, so they are not training features. Useful for late-window cross-checks.",
    },
    "AirDelhi hourly grid": {
        "role": "excluded (for this panel)",
        "reason": "2020-11/2021 grid measurements; no temporal overlap with the 2024-25 label panel. Kept for spatial-grid research only.",
    },
    "CPCB legacy AQI 2017-2023": {
        "role": "validation-only",
        "reason": "AQI-only history before the pollutant panel; not mixed into concentration targets.",
    },
    "CPCB live snapshot": {
        "role": "excluded (for this panel)",
        "reason": "Single current snapshot; reserved for the live-ingestion phase, not for offline training.",
    },
    "Kaggle/reference datasets": {
        "role": "validation-only",
        "reason": "Comparison and reproducibility sources; not part of the promoted feature contract.",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_openmeteo(path: Path) -> pd.DataFrame:
    """Load an Open-Meteo JSON export and return UTC-indexed hourly rows."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    hourly = payload["hourly"]
    series = {}
    for key, values in hourly.items():
        if key == "time":
            continue
        series[key] = pd.to_numeric(values, errors="coerce")
    frame = pd.DataFrame(series)
    frame["timestamp_utc"] = pd.to_datetime(hourly["time"], errors="coerce")
    frame["timestamp_utc"] = frame["timestamp_utc"].dt.tz_localize(
        "Asia/Kolkata"
    ).dt.tz_convert(UTC)
    frame = frame.dropna(subset=["timestamp_utc"]).drop_duplicates(subset=["timestamp_utc"])
    frame["timestamp_utc"] = frame["timestamp_utc"].dt.as_unit("ns")
    return frame.sort_values("timestamp_utc").reset_index(drop=True)


def _wind_uv(speed: pd.Series, direction: pd.Series) -> tuple[pd.Series, pd.Series]:
    radians = np.deg2rad(direction).astype(float)
    u = -speed * np.sin(radians)
    v = -speed * np.cos(radians)
    return u, v


def _asof_grid(series: pd.Series, target: pd.Series) -> pd.DataFrame:
    """Forward-fill `series` onto the newest available value at each target time.

    Weather/CAMS sources report IST-hour averages whose UTC time stamps fall on
    the half-hour. For a UTC issue hour ``t`` this returns the most recent
    reading with a stamp not later than ``t`` (no future information).
    """
    grid = pd.Series(series.values, index=series.index).sort_index()
    frame = grid.reindex(grid.index.union(target.sort_values().drop_duplicates()))
    frame = frame.ffill().reindex(target.sort_values().drop_duplicates())
    frame.name = series.name
    out = frame.to_frame()
    out.index.name = "timestamp_utc"
    return out


def _join_at(current: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    """Join a per-timestamp lookup onto the panel, fanning out duplicates safely."""
    index = current.index
    joined = current.join(lookup, on="timestamp_utc")
    return joined.reindex(index)


def _add_weather_features(base: pd.DataFrame, weather_path: Path) -> pd.DataFrame:
    weather = _load_openmeteo(weather_path)
    weather = weather.rename(columns=WEATHER_RENAME)
    weather["wind_speed_mps"] = weather["wind_speed_mps"] / 3.6  # km/h -> m/s
    u, v = _wind_uv(weather["wind_speed_mps"], weather["wind_direction_deg"])
    weather["wind_u"] = u
    weather["wind_v"] = v
    columns = [
        "temperature_c", "relative_humidity_pct", "rain_mm", "wind_u", "wind_v",
        "pressure_hpa", "cloud_cover_pct",
    ]
    target = base["timestamp_utc"]
    series = {column: weather.set_index("timestamp_utc")[column].astype(float) for column in columns}
    aligned = pd.concat(
        [_asof_grid(series[column], target)[column].rename(f"wx_{column}") for column in columns],
        axis=1,
    )
    for lag in (1, 6, 24):
        for column in columns:
            aligned[f"wx_{column}_lag_{lag}h"] = aligned[f"wx_{column}"].shift(lag)
    return _join_at(base, aligned)


def _add_cams_features(base: pd.DataFrame, cams_path: Path) -> pd.DataFrame:
    cams = _load_openmeteo(cams_path)
    cams = cams.rename(columns=CAMS_RENAME)
    cams["cams_pm25"] = cams.pop("pm25").astype(float)
    cams["cams_pm10"] = cams.pop("pm10").astype(float)
    target = base["timestamp_utc"]
    pm25 = _asof_grid(cams.set_index("timestamp_utc")["cams_pm25"], target)
    pm25 = pm25.rename(columns={"cams_pm25": "cams_pm25_asof"})
    pm10 = _asof_grid(cams.set_index("timestamp_utc")["cams_pm10"], target)
    pm10 = pm10.rename(columns={"cams_pm10": "cams_pm10_asof"})
    aligned = pd.concat([pm25["cams_pm25_asof"], pm10["cams_pm10_asof"]], axis=1)
    for lag in (CAMS_PUBLISH_LAG_H, 12, 24):
        aligned[f"cams_pm25_lag{lag}h"] = aligned["cams_pm25_asof"].shift(lag)
        aligned[f"cams_pm10_lag{lag}h"] = aligned["cams_pm10_asof"].shift(lag)
    aligned["cams_pm25_rolling_24h_mean"] = (
        aligned["cams_pm25_asof"].rolling(24, min_periods=12).mean().shift(CAMS_PUBLISH_LAG_H)
    )
    return _join_at(base, aligned)


def _add_firms_features(data: pd.DataFrame, firms_path: Path) -> pd.DataFrame:
    daily = pd.read_parquet(firms_path)
    daily["date_utc"] = pd.to_datetime(daily["date_utc"], utc=True).dt.normalize()
    daily = daily.sort_values("date_utc").reset_index(drop=True)
    for window in (3, 7):
        for column in ("fire_count", "fire_frp_sum", "fire_count_northwest", "fire_frp_sum_northwest"):
            daily[f"{column}_sum_{window}d"] = daily[column].rolling(window, min_periods=1).sum()

    lookup = daily.set_index("date_utc")
    previous_day = data["timestamp_utc"].dt.normalize() - pd.Timedelta(days=1)
    joined = lookup.reindex(previous_day)
    columns = [
        "fire_count", "fire_frp_sum", "fire_frp_max", "fire_high_confidence_count",
        "fire_count_delhi", "fire_frp_sum_delhi", "fire_count_northwest",
        "fire_frp_sum_northwest", "fire_count_west", "fire_frp_sum_west",
        "fire_count_east", "fire_frp_sum_east",
        "fire_count_sum_3d", "fire_frp_sum_sum_3d", "fire_count_northwest_sum_3d",
        "fire_frp_sum_northwest_sum_3d",
        "fire_count_sum_7d", "fire_frp_sum_sum_7d", "fire_count_northwest_sum_7d",
        "fire_frp_sum_northwest_sum_7d",
    ]
    feature_frame = pd.DataFrame(index=data.index, dtype=float)
    for column in columns:
        feature_frame[f"firms_{column}_prev_day"] = joined[column].to_numpy(dtype=float)
    return feature_frame


def build_dataset(
    multistation_path: Path = DEFAULT_MULTISTATION,
    weather_path: Path = DEFAULT_WEATHER,
    cams_path: Path = DEFAULT_CAMS,
    firms_path: Path = DEFAULT_FIRMS,
    output_path: Path = DEFAULT_OUTPUT,
    manifest_path: Path = DEFAULT_MANIFEST,
    roles_path: Path = DEFAULT_ROLES,
) -> dict[str, object]:
    base = pd.read_parquet(multistation_path)
    base["timestamp_utc"] = pd.to_datetime(base["timestamp_utc"], utc=True).dt.as_unit("ns")

    features = base.copy()
    features = _add_weather_features(features, weather_path)
    features = _add_cams_features(features, cams_path)
    firms_features = _add_firms_features(features, firms_path)
    features = pd.concat([features.reset_index(drop=True), firms_features.reset_index(drop=True)], axis=1)

    features = features.sort_values(["station_id", "timestamp_utc"]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)

    non_target = [c for c in features.columns if not c.startswith("target_")]
    aux_columns = [c for c in non_target if c.startswith(("wx_", "cams_", "firms_"))]
    manifest: dict[str, object] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "base_dataset": str(multistation_path),
        "base_rows": int(len(base)),
        "source_files": {
            "station_panel": str(multistation_path),
            "weather": str(weather_path),
            "cams": str(cams_path),
            "firms": str(firms_path),
        },
        "point_in_time_policy": (
            "CPCB panel and central weather observed at issue hour t; CAMS required to be at "
            f"least {CAMS_PUBLISH_LAG_H}h old (published-lag rule); FIRMS only complete days "
            "strictly before the issue date."
        ),
        "cams_publish_lag_hours": CAMS_PUBLISH_LAG_H,
        "station_count": int(features["station_id"].nunique()),
        "rows_written": int(len(features)),
        "start_utc": features["timestamp_utc"].min().isoformat(),
        "end_utc": features["timestamp_utc"].max().isoformat(),
        "feature_columns": non_target,
        "auxiliary_feature_columns": aux_columns,
        "auxiliary_fill_fraction": {
            column: float(features[column].notna().mean()) for column in aux_columns
        },
        "target_columns": [f"target_pm25_{horizon}h" for horizon in (1, 3, 6, 12, 24)],
        "dataset_roles": DATASET_ROLES,
        "output": {
            "path": str(output_path),
            "bytes": output_path.stat().st_size,
            "sha256": _sha256(output_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    roles_path.write_text(json.dumps(DATASET_ROLES, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multistation", type=Path, default=DEFAULT_MULTISTATION)
    parser.add_argument("--weather", type=Path, default=DEFAULT_WEATHER)
    parser.add_argument("--cams", type=Path, default=DEFAULT_CAMS)
    parser.add_argument("--firms", type=Path, default=DEFAULT_FIRMS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--roles", type=Path, default=DEFAULT_ROLES)
    args = parser.parse_args()
    print(json.dumps(build_dataset(args.multistation, args.weather, args.cams,
                                   args.firms, args.output, args.manifest, args.roles), indent=2))


if __name__ == "__main__":
    main()