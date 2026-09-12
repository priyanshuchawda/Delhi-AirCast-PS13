#!/usr/bin/env python3
"""Normalize NASA FIRMS CSV files into fire events and daily model features."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "firms"
OUTPUT_DIR = ROOT / "data" / "processed" / "firms"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _confidence_score(value: object) -> float:
    if pd.isna(value):
        return float("nan")
    text = str(value).strip().lower()
    if text in {"h", "high"}:
        return 100.0
    if text in {"n", "normal", "nominal"}:
        return 50.0
    if text in {"l", "low"}:
        return 25.0
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _read_event_file(path: Path, source: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        return pd.DataFrame()
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    required = {"latitude", "longitude", "acq_date", "acq_time"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path} lacks required FIRMS columns: {sorted(required)}")
    frame["source"] = source
    frame["source_file"] = str(path.relative_to(ROOT))
    frame["acq_time"] = frame["acq_time"].astype(str).str.zfill(4)
    frame["timestamp_utc"] = pd.to_datetime(
        frame["acq_date"].astype(str) + " " + frame["acq_time"],
        format="%Y-%m-%d %H%M",
        errors="coerce",
        utc=True,
    )
    frame["latitude"] = pd.to_numeric(frame["latitude"], errors="coerce")
    frame["longitude"] = pd.to_numeric(frame["longitude"], errors="coerce")
    frame["frp"] = pd.to_numeric(
        frame["frp"] if "frp" in frame else pd.Series(index=frame.index),
        errors="coerce",
    )
    frame["confidence_score"] = frame.get("confidence", pd.Series(index=frame.index)).map(_confidence_score)
    for column in ("satellite", "instrument", "daynight"):
        frame[column] = frame[column].astype(str) if column in frame else ""
    return frame[
        [
            "timestamp_utc",
            "latitude",
            "longitude",
            "frp",
            "confidence_score",
            "satellite",
            "instrument",
            "daynight",
            "source",
            "source_file",
        ]
    ].dropna(subset=["timestamp_utc", "latitude", "longitude"])


def _in_box(frame: pd.DataFrame, west: float, south: float, east: float, north: float) -> pd.Series:
    return (
        frame["longitude"].between(west, east, inclusive="both")
        & frame["latitude"].between(south, north, inclusive="both")
    )


def _daily_features(events: pd.DataFrame) -> pd.DataFrame:
    events = events.copy()
    events["date_utc"] = events["timestamp_utc"].dt.floor("D")
    events["is_high_confidence"] = events["confidence_score"].ge(80).fillna(False)
    regions = {
        "delhi": (76.8, 28.4, 77.4, 28.9),
        "northwest": (74.0, 29.0, 77.0, 32.0),
        "west": (74.0, 27.0, 77.0, 29.0),
        "east": (77.4, 27.5, 79.5, 31.5),
    }
    features = []
    for date_utc, group in events.groupby("date_utc", sort=True):
        row: dict[str, object] = {
            "date_utc": date_utc,
            "fire_count": int(len(group)),
            "fire_frp_sum": float(group["frp"].sum(min_count=1)),
            "fire_frp_max": float(group["frp"].max()),
            "fire_high_confidence_count": int(group["is_high_confidence"].sum()),
        }
        for region, (west, south, east, north) in regions.items():
            subset = group[_in_box(group, west, south, east, north)]
            row[f"fire_count_{region}"] = int(len(subset))
            row[f"fire_frp_sum_{region}"] = float(subset["frp"].sum(min_count=1))
        features.append(row)
    if not features:
        return pd.DataFrame()
    return pd.DataFrame(features).sort_values("date_utc").reset_index(drop=True)


def normalize(raw_root: Path = RAW, output_dir: Path = OUTPUT_DIR) -> dict[str, object]:
    files = sorted(raw_root.glob("**/*.csv"))
    if not files:
        raise RuntimeError("No FIRMS CSV files found; run download_firms.py first")
    frames = []
    for path in files:
        relative_parts = path.relative_to(raw_root).parts
        source = relative_parts[-2] if len(relative_parts) >= 2 else "unknown"
        frames.append(_read_event_file(path, source))
    events = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
    before = len(events)
    events = events.drop_duplicates(
        subset=["timestamp_utc", "latitude", "longitude", "satellite", "source"]
    ).sort_values("timestamp_utc").reset_index(drop=True)
    daily = _daily_features(events)
    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / "firms_fire_events.parquet"
    daily_path = output_dir / "firms_daily_features.parquet"
    events.to_parquet(events_path, index=False)
    daily.to_parquet(daily_path, index=False)
    manifest = {
        "source": "NASA FIRMS area CSV API",
        "input_files": len(files),
        "rows_before_deduplication": before,
        "event_rows": len(events),
        "daily_rows": len(daily),
        "start_utc": events["timestamp_utc"].min().isoformat(),
        "end_utc": events["timestamp_utc"].max().isoformat(),
        "outputs": {
            "events": {
                "path": str(events_path.relative_to(ROOT)),
                "bytes": events_path.stat().st_size,
                "sha256": sha256_file(events_path),
            },
            "daily_features": {
                "path": str(daily_path.relative_to(ROOT)),
                "bytes": daily_path.stat().st_size,
                "sha256": sha256_file(daily_path),
            },
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=RAW)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    print(json.dumps(normalize(args.raw_root, args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
