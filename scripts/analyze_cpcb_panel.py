#!/usr/bin/env python3
"""Create an auditable quality and coverage report for the CPCB station panel."""

from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT_DIR = Path("data/processed/cpcb_2024_25_hourly")
DEFAULT_LIVE_SNAPSHOT = Path("data/raw/cpcb_live/cpcb_live_delhi.parquet")
DEFAULT_OUTPUT_DIR = Path("data/processed/cpcb_panel_quality")

QUALITY_COLUMNS = (
    "pm25",
    "pm10",
    "no2",
    "so2",
    "co",
    "o3",
    "nh3",
    "temperature_c",
    "relative_humidity_pct",
    "wind_speed_mps",
    "wind_direction_deg",
    "rain_mm",
)


def _station_key(value: object) -> str:
    """Build a conservative key for historical/live station-name matching."""

    text = str(value).lower().replace("_", " ")
    text = re.sub(r"\b(delhi|cpcb|dpcc|iitm|imd)\b", " ", text)
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _finite_stats(values: pd.Series) -> dict[str, float | None]:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return {name: None for name in ("min", "median", "p95", "max")}
    return {
        "min": float(clean.min()),
        "median": float(clean.median()),
        "p95": float(clean.quantile(0.95)),
        "max": float(clean.max()),
    }


def station_quality(frame: pd.DataFrame) -> dict[str, object]:
    """Summarize one normalized station frame without imputing observations."""

    if frame.empty:
        raise ValueError("station frame is empty")
    frame = frame.copy()
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc")
    timestamps = frame["timestamp_utc"]
    expected = pd.date_range(timestamps.min().floor("h"), timestamps.max().floor("h"), freq="h", tz="UTC")
    observed = pd.DatetimeIndex(timestamps.dt.floor("h").drop_duplicates())
    missing_hours = int(expected.difference(observed).size)
    station_id = str(frame["station_id"].iloc[0])
    station_name = str(frame["station_name"].iloc[0])
    quality: dict[str, object] = {
        "station_id": station_id,
        "station_name": station_name,
        "rows": int(len(frame)),
        "start_utc": timestamps.min().isoformat(),
        "end_utc": timestamps.max().isoformat(),
        "expected_hourly_rows": int(len(expected)),
        "missing_hour_rows": missing_hours,
        "coverage_fraction": float(len(observed) / len(expected)) if len(expected) else 0.0,
        "duplicate_timestamp_rows": int(timestamps.duplicated().sum()),
        "pm25_non_null_rows": int(frame.get("pm25", pd.Series(dtype=float)).notna().sum()),
        "pm25_missing_fraction": float(frame["pm25"].isna().mean()) if "pm25" in frame else 1.0,
        "pm25_stats": _finite_stats(frame.get("pm25", pd.Series(dtype=float))),
        "missing_fraction": {},
    }
    quality["missing_fraction"] = {
        column: float(frame[column].isna().mean()) if column in frame else 1.0
        for column in QUALITY_COLUMNS
    }
    return quality


def _load_live_coordinates(path: Path) -> pd.DataFrame:
    columns = ["station", "latitude", "longitude"]
    if not path.exists():
        return pd.DataFrame(columns=columns + ["station_key"])
    frame = pd.read_parquet(path)
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"live snapshot missing coordinate columns: {missing}")
    frame = frame[columns].drop_duplicates().copy()
    frame["latitude"] = pd.to_numeric(frame["latitude"], errors="coerce")
    frame["longitude"] = pd.to_numeric(frame["longitude"], errors="coerce")
    frame["station_key"] = frame["station"].map(_station_key)
    return frame.dropna(subset=["station_key"]).drop_duplicates("station_key")


def build_report(
    input_dir: Path = DEFAULT_INPUT_DIR,
    live_snapshot: Path = DEFAULT_LIVE_SNAPSHOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, object]:
    """Write station, network, and registry reports and return the summary."""

    paths = sorted(input_dir.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no station Parquets found under {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    live = _load_live_coordinates(live_snapshot)
    station_summaries: list[dict[str, object]] = []
    registry_rows: list[dict[str, object]] = []
    coverage_frames: list[pd.DataFrame] = []
    panel_rows = 0

    for path in paths:
        frame = pd.read_parquet(path)
        summary = station_quality(frame)
        station_summaries.append(summary)
        panel_rows += int(summary["rows"])
        station_key = _station_key(summary["station_name"])
        match = live[live["station_key"] == station_key]
        coordinate = match.iloc[0] if len(match) == 1 else None
        registry_rows.append(
            {
                "station_id": summary["station_id"],
                "station_name": summary["station_name"],
                "station_key": station_key,
                "latitude": float(coordinate["latitude"]) if coordinate is not None else np.nan,
                "longitude": float(coordinate["longitude"]) if coordinate is not None else np.nan,
                "coordinate_match": "unique_live_name" if coordinate is not None else "unmatched",
                "source_file": str(path),
            }
        )
        timestamps = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce")
        valid = frame.loc[timestamps.notna()].copy()
        valid["timestamp_utc"] = timestamps[timestamps.notna()].dt.floor("h").to_numpy()
        coverage_frames.append(
            valid.assign(pm25_available=valid["pm25"].notna())
            .groupby("timestamp_utc", as_index=False)["pm25_available"]
            .sum()
            .rename(columns={"pm25_available": "stations_with_pm25"})
        )

    coverage = (
        pd.concat(coverage_frames, ignore_index=True)
        .groupby("timestamp_utc", as_index=False)["stations_with_pm25"]
        .sum()
        .sort_values("timestamp_utc")
    )
    coverage["stations_total"] = len(paths)
    coverage["network_coverage_fraction"] = coverage["stations_with_pm25"] / len(paths)
    coverage.to_csv(output_dir / "network_hourly_coverage.csv", index=False)

    station_table = pd.DataFrame(station_summaries).sort_values("station_id")
    station_table.to_csv(output_dir / "station_quality.csv", index=False)
    registry = pd.DataFrame(registry_rows).sort_values("station_id")
    registry.to_csv(output_dir / "station_registry.csv", index=False)

    summary: dict[str, object] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "input_dir": str(input_dir),
        "live_snapshot": str(live_snapshot),
        "station_count": len(paths),
        "panel_rows": panel_rows,
        "network_start_utc": coverage["timestamp_utc"].min().isoformat() if len(coverage) else None,
        "network_end_utc": coverage["timestamp_utc"].max().isoformat() if len(coverage) else None,
        "coordinate_matches": int((registry["coordinate_match"] == "unique_live_name").sum()),
        "coordinate_unmatched": int((registry["coordinate_match"] == "unmatched").sum()),
        "stations_below_60_percent_coverage": int((station_table["coverage_fraction"] < 0.60).sum()),
        "network_mean_coverage_fraction": float(coverage["network_coverage_fraction"].mean()) if len(coverage) else 0.0,
        "outputs": {
            "station_quality": str(output_dir / "station_quality.csv"),
            "station_registry": str(output_dir / "station_registry.csv"),
            "network_hourly_coverage": str(output_dir / "network_hourly_coverage.csv"),
        },
    }
    (output_dir / "panel_quality.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--live-snapshot", type=Path, default=DEFAULT_LIVE_SNAPSHOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    summary = build_report(args.input_dir, args.live_snapshot, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
