"""Build a leakage-safe one-hour-ahead CPCB forecasting table."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_STATION_DIR = Path("data/processed/cpcb_2024_25_hourly")
DEFAULT_WEATHER = Path("data/raw/weather/delhi-central-hourly-2017-2025.json")
DEFAULT_CAMS = Path("data/raw/cams/delhi-central-cams-global-hourly-2023-2025.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _external_frame(path: Path, prefix: str) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    hourly = payload["hourly"]
    timestamp = pd.to_datetime(hourly.pop("time"), errors="coerce").tz_localize(
        payload.get("timezone", "Asia/Kolkata")
    ).tz_convert("UTC")
    frame = pd.DataFrame({"timestamp_utc": timestamp})
    for name, values in hourly.items():
        frame[f"{prefix}_{name}"] = pd.to_numeric(values, errors="coerce")
    return frame


def build_dataset(
    station_id: str,
    station_dir: Path,
    weather_path: Path,
    cams_path: Path,
    output_path: Path,
    manifest_path: Path,
) -> dict[str, object]:
    station_path = station_dir / f"{station_id}.parquet"
    if not station_path.exists():
        raise FileNotFoundError(f"No normalized CPCB station file: {station_path}")

    frame = pd.read_parquet(station_path).sort_values("timestamp_utc").reset_index(drop=True)
    frame["target_timestamp_utc"] = frame["timestamp_utc"] + pd.Timedelta(hours=1)
    frame["target_pm25_next_hour"] = frame["pm25"].shift(-1)

    for hours in (1, 3, 6, 24):
        frame[f"pm25_lag_{hours}h"] = frame["pm25"].shift(hours)
    for hours in (3, 6, 24):
        frame[f"pm25_rolling_mean_{hours}h"] = frame["pm25"].rolling(hours).mean().shift(1)

    frame["hour_sin"] = np.sin(2 * np.pi * frame["timestamp_utc"].dt.hour / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * frame["timestamp_utc"].dt.hour / 24)
    frame["weekday_sin"] = np.sin(2 * np.pi * frame["timestamp_utc"].dt.dayofweek / 7)
    frame["weekday_cos"] = np.cos(2 * np.pi * frame["timestamp_utc"].dt.dayofweek / 7)

    weather = _external_frame(weather_path, "weather")
    cams = _external_frame(cams_path, "cams")
    # Open-Meteo returns local clock-hour values for Asia/Kolkata. The CPCB
    # archive is hourly on UTC boundaries, which appear at :30 in local time.
    # Nearest-hour alignment keeps the source timestamps intact and avoids a
    # fabricated exact match between the two reporting clocks.
    frame = pd.merge_asof(
        frame.sort_values("timestamp_utc"),
        weather.sort_values("timestamp_utc"),
        on="timestamp_utc",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=31),
    )
    frame = pd.merge_asof(
        frame.sort_values("timestamp_utc"),
        cams.sort_values("timestamp_utc"),
        on="timestamp_utc",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=31),
    )
    frame = frame.sort_values("timestamp_utc").reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    manifest: dict[str, object] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "station_id": station_id,
        "station_source": str(station_path),
        "weather_source": str(weather_path),
        "cams_source": str(cams_path),
        "forecast_horizon_hours": 1,
        "feature_policy": "current and lagged observations only; target is shifted one hour forward",
        "rows_written": int(len(frame)),
        "columns": list(frame.columns),
        "target_non_null_rows": int(frame["target_pm25_next_hour"].notna().sum()),
        "start_utc": frame["timestamp_utc"].min().isoformat(),
        "end_utc": frame["timestamp_utc"].max().isoformat(),
        "sources": {
            "station_sha256": _sha256(station_path),
            "weather_sha256": _sha256(weather_path),
            "cams_sha256": _sha256(cams_path),
        },
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
    parser.add_argument("--station-id", default="site_105")
    parser.add_argument("--station-dir", type=Path, default=DEFAULT_STATION_DIR)
    parser.add_argument("--weather", type=Path, default=DEFAULT_WEATHER)
    parser.add_argument("--cams", type=Path, default=DEFAULT_CAMS)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or Path(f"data/processed/{args.station_id}_next_hour_forecast.parquet")
    manifest = args.manifest or Path(f"data/processed/{args.station_id}_next_hour_forecast.manifest.json")
    result = build_dataset(args.station_id, args.station_dir, args.weather, args.cams, output, manifest)
    print(f"Wrote {result['rows_written']:,} rows to {result['output']['path']}")


if __name__ == "__main__":
    main()
