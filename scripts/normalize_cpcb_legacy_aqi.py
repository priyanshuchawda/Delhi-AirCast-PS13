#!/usr/bin/env python3
"""Normalize OpenCity's legacy 2017-2023 hourly AQI grids.

These files contain AQI values rather than the pollutant concentrations used
by the main CPCB forecast table, so they are deliberately emitted as a
separate validation dataset.
"""

from __future__ import annotations

import argparse
import calendar
import csv
import hashlib
import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESOURCE_MANIFEST = ROOT / "data" / "raw" / "opencity" / "manifest.json"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "cpcb_2017_2023_aqi.parquet"
MONTHS = {name: number for number, name in enumerate(calendar.month_name) if name}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_rows(rows: list[list[str]], station_id: str, station_name: str) -> list[dict[str, object]]:
    year: int | None = None
    month: int | None = None
    results: list[dict[str, object]] = []
    for row in rows:
        if not row:
            continue
        first = row[0].strip().strip('"')
        if first == "Year" and len(row) > 1:
            year = int(row[1])
            month = None
            continue
        match = re.fullmatch(r"([A-Za-z]+)-(\d{4})", first)
        if match:
            month = MONTHS.get(match.group(1))
            year = int(match.group(2))
            continue
        if year is None or month is None or not first.isdigit():
            continue
        day = int(first)
        if not 1 <= day <= calendar.monthrange(year, month)[1]:
            continue
        for hour, value in enumerate(row[1:25]):
            value = value.strip().strip('"')
            if not value:
                continue
            try:
                aqi = float(value)
            except ValueError:
                continue
            timestamp_ist = pd.Timestamp(year, month, day, hour, tz="Asia/Kolkata")
            results.append(
                {
                    "station_id": station_id,
                    "station_name": station_name,
                    "timestamp_ist": timestamp_ist,
                    "timestamp_utc": timestamp_ist.tz_convert("UTC"),
                    "aqi": aqi,
                    "source": "OpenCity/CPCB legacy AQI archive",
                }
            )
    return results


def normalize(resource_manifest: Path = DEFAULT_RESOURCE_MANIFEST, output: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    manifest = json.loads(resource_manifest.read_text(encoding="utf-8"))
    frames: list[pd.DataFrame] = []
    input_files = []
    for resource in manifest.get("resources", []):
        if "2017-2023" not in resource.get("name", ""):
            continue
        path = ROOT / resource["path"]
        station_name = resource["name"].replace(" AQI Data 2017-2023", "")
        station_id = "legacy_" + re.sub(r"[^a-z0-9]+", "_", station_name.lower()).strip("_")
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.reader(handle))
        parsed = parse_rows(rows, station_id, station_name)
        if parsed:
            frames.append(pd.DataFrame(parsed))
        input_files.append(
            {
                "name": resource["name"],
                "path": resource["path"],
                "sha256": _sha256(path),
                "rows_written": len(parsed),
            }
        )
    if not frames:
        raise RuntimeError("No legacy AQI rows were parsed")
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.drop_duplicates(
        subset=["station_id", "timestamp_utc"], keep="first"
    ).sort_values(["station_id", "timestamp_utc"]).reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    result = {
        "source": "OpenCity/CPCB legacy 2017-2023 hourly AQI archive",
        "input_files": input_files,
        "station_count": int(frame["station_id"].nunique()),
        "rows_written": len(frame),
        "start_utc": frame["timestamp_utc"].min().isoformat(),
        "end_utc": frame["timestamp_utc"].max().isoformat(),
        "output": str(output.relative_to(ROOT)),
        "bytes": output.stat().st_size,
        "sha256": _sha256(output),
        "warning": "AQI-only validation data; do not mix with pollutant concentration targets without auditing.",
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource-manifest", type=Path, default=DEFAULT_RESOURCE_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(normalize(args.resource_manifest, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
