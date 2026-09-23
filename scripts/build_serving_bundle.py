#!/usr/bin/env python3
"""Assemble a compact, offline-serving bundle from local trained artifacts.

The bundle contains one historical feature row per station, coordinates,
forecast models, and optional six-hour pollutant models. It deliberately does
not copy raw or training datasets.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import joblib
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
HORIZONS = (1, 3, 6, 12, 24)
POLLUTANTS = ("pm10", "no2", "co", "o3")


def build_bundle(data_root: Path, output: Path) -> dict[str, object]:
    """Create the compact root expected by DELHI_AIRCAST_DATA_ROOT."""
    dataset = data_root / "data/processed/delhi_unified_forecast.parquet"
    registry = data_root / "data/processed/cpcb_panel_quality/station_registry.csv"
    model_root = data_root / "data/runs/final_xgboost"
    pollutant_root = data_root / "data/runs/multipollutant_aqi_final"
    if not dataset.exists() or not registry.exists():
        raise FileNotFoundError("Unified panel and station_registry.csv are required")

    output = output.resolve()
    feature_columns: set[str] = set()
    models: list[tuple[Path, Path]] = []
    for horizon in HORIZONS:
        source = model_root / f"model_{horizon}h.joblib"
        if not source.exists():
            continue
        bundle = joblib.load(source)
        feature_columns.update(bundle["feature_columns"])
        models.append((source, output / f"data/runs/final_xgboost/{source.name}"))
    for pollutant in POLLUTANTS:
        source = pollutant_root / f"{pollutant}_6h.joblib"
        if source.exists():
            bundle = joblib.load(source)
            feature_columns.update(bundle["feature_columns"])
            models.append((source, output / f"data/runs/multipollutant_aqi_final/{source.name}"))
    if not models:
        raise FileNotFoundError("No trained final XGBoost artifacts found")

    metadata_columns = ["station_id", "station_name", "timestamp_utc"]
    available = set(pq.read_schema(dataset).names)
    columns = metadata_columns + sorted(feature_columns - set(metadata_columns))
    # Station one-hot columns live in the model contract, not the Parquet
    # table; the API reconstructs those from station_id at inference time.
    source = pd.read_parquet(dataset, columns=[c for c in columns if c in available])
    source["timestamp_utc"] = pd.to_datetime(source["timestamp_utc"], utc=True)
    source = source.sort_values("timestamp_utc")
    current_col = "pm25_current"
    if current_col in source:
        source = source.dropna(subset=[current_col])
    latest = source.groupby("station_id", sort=False, as_index=False).tail(1).copy()
    latest = latest.drop(columns=[c for c in latest if c.startswith("target_")], errors="ignore")

    stations = pd.read_csv(registry)
    stations = stations[["station_id", "station_name", "latitude", "longitude"]]
    latest = latest.drop(columns=["station_name"], errors="ignore").merge(stations, on="station_id", how="left", suffixes=("", "_registry"))
    latest["station_name"] = latest["station_name_registry"].fillna(latest["station_name"]) if "station_name_registry" in latest else latest["station_name"]
    latest = latest.drop(columns=["station_name_registry"], errors="ignore")

    # Keep a tiny 72-hour sparkline per station so deployments need not ship
    # the full source panel just to render historical context.
    history = pd.read_parquet(dataset, columns=["station_id", "timestamp_utc", "pm25_current"])
    history["timestamp_utc"] = pd.to_datetime(history["timestamp_utc"], utc=True)
    history = history.dropna(subset=["pm25_current"]).sort_values("timestamp_utc").groupby("station_id", sort=False).tail(72)

    feature_path = output / "data/processed/delhi_serving_features.parquet"
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    latest.to_parquet(feature_path, index=False)
    history_path = output / "data/processed/delhi_station_history.parquet"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history.to_parquet(history_path, index=False)
    registry_path = output / "data/processed/cpcb_panel_quality/station_registry.csv"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    stations.to_csv(registry_path, index=False)
    for original, destination in models:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, destination)

    manifest = {
        "stations": len(latest),
        "horizons": sorted(int(p.stem.split("_")[1][:-1]) for _, p in models if p.parent.name == "final_xgboost"),
        "six_hour_multipollutant": all((output / f"data/runs/multipollutant_aqi_final/{p}_6h.joblib").exists() for p in POLLUTANTS),
        "latest_observation_utc": latest["timestamp_utc"].max().isoformat(),
        "feature_rows_path": "data/processed/delhi_serving_features.parquet",
        "warning": "Historical offline snapshot; not live data.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT, help="Project root containing ignored data/")
    parser.add_argument("--output", type=Path, required=True, help="Serving bundle root; not committed")
    args = parser.parse_args()
    print(json.dumps(build_bundle(args.data_root, args.output), indent=2))


if __name__ == "__main__":
    main()
