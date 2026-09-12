#!/usr/bin/env python3
"""Normalize the detailed 2024-25 CPCB/OpenCity station files to hourly Parquet."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "opencity"
OUT = ROOT / "data" / "processed" / "cpcb_2024_25_hourly"
MANIFEST = RAW / "manifest.json"


ALIASES = {
    "Station ID": "station_id",
    "State": "state",
    "City": "city",
    "Station Name": "station_name",
    "Timestamp": "timestamp",
    "PM2.5 (µg/m³)": "pm25",
    "PM10 (µg/m³)": "pm10",
    "NO (µg/m³)": "no",
    "NO2 (µg/m³)": "no2",
    "NOx (ppb)": "nox",
    "NH3 (µg/m³)": "nh3",
    "SO2 (µg/m³)": "so2",
    "CO (mg/m³)": "co",
    "Ozone (µg/m³)": "o3",
    "Benzene (µg/m³)": "benzene",
    "Toluene (µg/m³)": "toluene",
    "Xylene (µg/m³)": "xylene",
    "O Xylene (µg/m³)": "o_xylene",
    "Eth-Benzene (µg/m³)": "eth_benzene",
    "MP-Xylene (µg/m³)": "mp_xylene",
    "AT (°C)": "temperature_c",
    "RH (%)": "relative_humidity_pct",
    "WS (m/s)": "wind_speed_mps",
    "WD (deg)": "wind_direction_deg",
    "RF (mm)": "rain_mm",
    "TOT-RF (mm)": "total_rain_mm",
    "SR (W/mt2)": "solar_wm2",
    "BP (mmHg)": "pressure_mmhg",
    "VWS (m/s)": "vertical_wind_mps",
}


def clean_column(column: str) -> str:
    return re.sub(r"\s+", " ", str(column).replace("μ", "µ").strip())


def normalize_frame(frame: pd.DataFrame, resource: dict) -> tuple[pd.DataFrame, dict]:
    frame.columns = [clean_column(column) for column in frame.columns]
    rename = {column: ALIASES[column] for column in frame.columns if column in ALIASES}
    frame = frame.rename(columns=rename)
    required = {"station_id", "station_name", "timestamp", "pm25", "pm10"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing expected columns {missing}; got {list(frame.columns)}")

    frame["timestamp_utc"] = pd.to_datetime(frame.pop("timestamp"), utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp_utc"])
    numeric = [column for column in frame.columns if column not in {"station_id", "state", "city", "station_name", "timestamp_utc"}]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame[column] = frame[column].replace([999, 998, 985, 1999, -9999, -999], np.nan)

    first = frame.iloc[0]
    station_id = str(first["station_id"])
    station_name = str(first["station_name"])
    frame = frame.set_index("timestamp_utc").sort_index()
    numeric_columns = [column for column in numeric if column in frame.columns]
    hourly = frame[numeric_columns].resample("1h").mean()
    coverage = frame[["pm25", "pm10"]].resample("1h").count().rename(columns={"pm25": "pm25_samples", "pm10": "pm10_samples"})
    hourly = hourly.join(coverage)
    hourly.loc[hourly["pm25_samples"] < 2, "pm25"] = np.nan
    hourly.loc[hourly["pm10_samples"] < 2, "pm10"] = np.nan
    hourly = hourly.reset_index()
    hourly.insert(0, "station_id", station_id)
    hourly.insert(1, "station_name", station_name)
    hourly.insert(2, "source", "OpenCity/CPCB")
    hourly.insert(3, "source_resource_id", resource.get("resource_id", ""))
    hourly["timestamp_ist"] = hourly["timestamp_utc"].dt.tz_convert("Asia/Kolkata")
    ordered = [
        "station_id", "station_name", "source", "source_resource_id",
        "timestamp_utc", "timestamp_ist", "pm25", "pm10", "no", "no2",
        "nox", "nh3", "so2", "co", "o3", "benzene", "toluene", "xylene",
        "o_xylene", "eth_benzene", "mp_xylene", "temperature_c",
        "relative_humidity_pct", "wind_speed_mps", "wind_direction_deg",
        "rain_mm", "total_rain_mm", "solar_wm2", "pressure_mmhg",
        "vertical_wind_mps", "pm25_samples", "pm10_samples",
    ]
    for column in ordered:
        if column not in hourly:
            hourly[column] = np.nan
    hourly = hourly[ordered]
    summary = {
        "station_id": station_id,
        "station_name": station_name,
        "raw_rows": int(len(frame)),
        "hourly_rows": int(len(hourly)),
        "start_utc": hourly["timestamp_utc"].min().isoformat() if len(hourly) else None,
        "end_utc": hourly["timestamp_utc"].max().isoformat() if len(hourly) else None,
        "pm25_non_null_hours": int(hourly["pm25"].notna().sum()),
        "pm25_missing_fraction": float(hourly["pm25"].isna().mean()) if len(hourly) else 1.0,
    }
    return hourly, summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    resources = [resource for resource in manifest["resources"] if "2024-25" in (resource.get("name") or "")]
    summaries = []
    errors = []
    for index, resource in enumerate(resources, start=1):
        source_path = ROOT / resource["local"]
        print(f"[{index}/{len(resources)}] {resource['name']}", flush=True)
        try:
            frame = pd.read_csv(source_path, low_memory=False)
            hourly, summary = normalize_frame(frame, resource)
            destination = OUT / f"{summary['station_id']}.parquet"
            hourly.to_parquet(destination, index=False)
            summary["output"] = str(destination.relative_to(ROOT))
            summaries.append(summary)
        except Exception as error:  # keep other stations moving; report every failure
            errors.append({"resource": resource.get("name"), "error": repr(error)})
            print(f"  ERROR: {error}")
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(MANIFEST.relative_to(ROOT)),
        "resource_count": len(resources),
        "station_count": len(summaries),
        "errors": errors,
        "stations": summaries,
    }
    (OUT / "manifest.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(summaries).sort_values("station_name").to_csv(OUT / "station_summary.csv", index=False)
    print(f"Wrote {len(summaries)} stations to {OUT}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
