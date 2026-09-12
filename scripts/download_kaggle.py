"""Download the curated public Kaggle comparison datasets.

Authentication is intentionally external to this repository. The Kaggle CLI
may use ``KAGGLE_API_TOKEN`` or its own local configuration; no token is read
from or written to project files.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


DATASETS = (
    "kunshbhatia/delhi-air-quality-dataset",
    "deepaksirohiwal/delhi-air-quality",
    "anuragbantu/new-delhi-air-quality",
    "digantdixit/delhi-air-quality-index-data",
    "rohanrao/air-quality-data-in-india",
    "jatinkalra17/delhi-aqi",
    "abhisheksjha/time-series-air-quality-data-of-india-2010-2023",
    "aniket0712/delhi-ncr-hourly-air-quality-dataset-20192024",
    "sumanbera19/delhi-air-quality-dataset-20232025-aqi-data",
    "sohails07/delhi-weather-and-aqi-dataset-2025",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", dest="datasets", choices=DATASETS)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/kaggle"))
    args = parser.parse_args()

    kaggle = shutil.which("kaggle") or str(Path.home() / ".local/bin/kaggle")
    if not Path(kaggle).exists() and shutil.which("kaggle") is None:
        raise SystemExit("Kaggle CLI not found; install it with: uv tool install kaggle")

    selected = args.datasets or DATASETS
    args.output_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    for reference in selected:
        slug = reference.split("/", 1)[1]
        destination = args.output_dir / slug
        destination.mkdir(parents=True, exist_ok=True)
        command = [kaggle, "datasets", "download", "-d", reference, "-p", str(destination), "--unzip"]
        print(f"Downloading {reference} -> {destination}")
        subprocess.run(command, check=True, env=environment)


if __name__ == "__main__":
    main()
