#!/usr/bin/env python3
"""Normalize OpenAQ hourly batch JSON into an analysis-ready Parquet file."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "openaq" / "hours"
OUTPUT = ROOT / "data" / "processed" / "openaq" / "openaq_delhi_ncr_hourly.parquet"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(batch_names: list[str] | None = None) -> dict:
    requested = set(batch_names or [])
    rows: list[dict] = []
    input_manifests: list[dict] = []
    for manifest_path in sorted(RAW.glob("*/manifest.json")):
        batch = manifest_path.parent.name
        if requested and batch not in requested:
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        input_manifests.append(
            {
                "batch": batch,
                "manifest": str(manifest_path.relative_to(ROOT)),
                "sensor_count": manifest.get("sensor_count"),
                "record_count": len(manifest.get("records", [])),
            }
        )
        for record in manifest.get("records", []):
            payload_path = ROOT / record["path"]
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
            for item in payload.get("results", []):
                period = item.get("period") or {}
                datetime_from = period.get("datetimeFrom") or {}
                datetime_to = period.get("datetimeTo") or {}
                coverage = item.get("coverage") or {}
                rows.append(
                    {
                        "batch": batch,
                        "sensor_id": record["sensor_id"],
                        "location_id": record["location_id"],
                        "location_name": record["location_name"],
                        "parameter_id": record["parameter_id"],
                        "parameter": item.get("parameter", {}).get("name", record["parameter"]),
                        "units": item.get("parameter", {}).get("units", record["units"]),
                        "value": item.get("value"),
                        "datetime_from_utc": datetime_from.get("utc"),
                        "datetime_to_utc": datetime_to.get("utc"),
                        "datetime_from_local": datetime_from.get("local"),
                        "datetime_to_local": datetime_to.get("local"),
                        "coverage_expected_count": coverage.get("expectedCount"),
                        "coverage_observed_count": coverage.get("observedCount"),
                        "coverage_percent": coverage.get("percentComplete"),
                    }
                )

    if not rows:
        raise RuntimeError("No OpenAQ hourly rows found; run download_openaq.py first")
    frame = pd.DataFrame(rows)
    frame["datetime_from_utc"] = pd.to_datetime(frame["datetime_from_utc"], utc=True)
    frame["datetime_to_utc"] = pd.to_datetime(frame["datetime_to_utc"], utc=True)
    frame = frame.sort_values(["datetime_from_utc", "location_id", "parameter_id", "sensor_id"])
    before = len(frame)
    frame = frame.drop_duplicates(
        subset=["sensor_id", "datetime_from_utc"], keep="first"
    ).reset_index(drop=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    partial = OUTPUT.with_suffix(OUTPUT.suffix + ".part")
    frame.to_parquet(partial, index=False)
    partial.replace(OUTPUT)
    result = {
        "source": "OpenAQ API v3 hourly averages",
        "input_manifests": input_manifests,
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "rows_before_deduplication": before,
        "rows_written": len(frame),
        "duplicate_rows_collapsed": before - len(frame),
        "sensor_count": int(frame["sensor_id"].nunique()),
        "location_count": int(frame["location_id"].nunique()),
        "parameters": sorted(frame["parameter"].dropna().unique().tolist()),
        "start_utc": frame["datetime_from_utc"].min().isoformat(),
        "end_utc": frame["datetime_to_utc"].max().isoformat(),
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_bytes": OUTPUT.stat().st_size,
        "output_sha256": sha256_file(OUTPUT),
    }
    manifest_path = OUTPUT.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", action="append", help="Batch directory name; repeatable")
    args = parser.parse_args()
    result = normalize(args.batch)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
