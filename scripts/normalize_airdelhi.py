"""Normalize the IIT Delhi AirDelhi 1 km x 1 hour grid variant.

The Hugging Face mirror contains the same public AirDelhi research dataset as
the IIT Delhi download page, but is easier to reproduce in automation. This
script keeps the grid's spatial and hourly resolution, converts timestamps to
UTC, coerces measurements to numeric values, and writes a compact Parquet
table plus a local provenance manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path("data/raw/huggingface/DelhiPollDataset/4.Grid(PM+Met)")
DEFAULT_OUTPUT = Path("data/processed/airdelhi_grid_hourly.parquet")
DEFAULT_MANIFEST = Path("data/processed/airdelhi_grid_hourly.manifest.json")

NUMERIC_COLUMNS = (
    "latitude",
    "longitude",
    "pressure_hpa",
    "temperature_c",
    "humidity_pct",
    "pm1_0",
    "pm2_5",
    "pm10",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(input_dir: Path, output_path: Path, manifest_path: Path) -> dict[str, object]:
    sources = sorted(input_dir.glob("*.csv.gz"))
    if not sources:
        raise FileNotFoundError(f"No .csv.gz grid files found under {input_dir}")

    frames: list[pd.DataFrame] = []
    source_stats: list[dict[str, object]] = []
    for source in sources:
        frame = pd.read_csv(source, compression="gzip")
        frame = frame.drop(columns=[column for column in frame if column.startswith("Unnamed:")])
        frame = frame.rename(
            columns={
                "dateTime": "timestamp_utc",
                "lat": "latitude",
                "long": "longitude",
                "pressure": "pressure_hpa",
                "temperature": "temperature_c",
                "humidity": "humidity_pct",
            }
        )

        required = {"timestamp_utc", *NUMERIC_COLUMNS}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{source} is missing required columns: {missing}")

        frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce")
        for column in NUMERIC_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        frame = frame.dropna(subset=["timestamp_utc", "latitude", "longitude"])
        for column in ("pm1_0", "pm2_5", "pm10"):
            frame.loc[frame[column] < 0, column] = pd.NA
        frame["source_month"] = source.name.removesuffix(".csv.gz")
        frames.append(frame[["timestamp_utc", *NUMERIC_COLUMNS, "source_month"]])
        source_stats.append(
            {
                "path": str(source),
                "rows_read": int(len(frame)),
                "sha256": _sha256(source),
            }
        )

    combined = pd.concat(frames, ignore_index=True)
    key = ["timestamp_utc", "latitude", "longitude"]
    duplicate_rows = int(combined.duplicated(key).sum())
    aggregations: dict[str, str] = {column: "mean" for column in NUMERIC_COLUMNS if column not in key}
    aggregations["source_month"] = "first"
    normalized = combined.groupby(key, as_index=False, sort=True).agg(aggregations)
    normalized = normalized.sort_values(key).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_parquet(output_path, index=False)

    manifest: dict[str, object] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "source_dataset": "sachin-iitd/DelhiPollDataset",
        "source_variant": "4.Grid(PM+Met)",
        "input_directory": str(input_dir),
        "source_files": source_stats,
        "rows_before_deduplication": int(len(combined)),
        "duplicate_rows_collapsed": duplicate_rows,
        "rows_written": int(len(normalized)),
        "columns": list(normalized.columns),
        "start_utc": normalized["timestamp_utc"].min().isoformat(),
        "end_utc": normalized["timestamp_utc"].max().isoformat(),
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
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    manifest = normalize(args.input_dir, args.output, args.manifest)
    print(
        f"Wrote {manifest['rows_written']:,} normalized rows to {manifest['output']['path']} "
        f"({manifest['output']['bytes']:,} bytes)"
    )


if __name__ == "__main__":
    main()
