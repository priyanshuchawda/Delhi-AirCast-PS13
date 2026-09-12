#!/usr/bin/env python3
"""Download public Delhi AirCast datasets with provenance and checksums.

The downloader deliberately keeps sources in separate directories.  Similar
looking pollutant columns from different providers must not be merged until
their station IDs, units, timestamps, and provenance have been audited.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
USER_AGENT = "Delhi-AirCast/0.1 (academic air-quality research; contact via repository)"
OPENCITY_API = "https://data.opencity.in/api/3/action/package_show?id=delhi-hourly-air-quality-reports"
OPENMETEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
OPENMETEO_AIR = "https://air-quality-api.open-meteo.com/v1/air-quality"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slug(value: str) -> str:
    value = value.lower().replace("&", "and")
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    return value[:120]


def fetch_bytes(url: str, *, timeout: int = 60) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_file(url: str, destination: Path, *, advertised_size: int | None = None) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    if destination.exists() and destination.stat().st_size > 0:
        return {
            "status": "exists",
            "path": str(destination.relative_to(ROOT)),
            "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
        }

    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=180) as response, partial.open("wb") as output:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    output.write(block)
            partial.replace(destination)
            size = destination.stat().st_size
            result = {
                "status": "downloaded",
                "path": str(destination.relative_to(ROOT)),
                "bytes": size,
                "sha256": sha256_file(destination),
            }
            if advertised_size and size != advertised_size:
                result["size_warning"] = {
                    "advertised": advertised_size,
                    "received": size,
                }
            return result
        except (HTTPError, URLError, TimeoutError, OSError) as error:
            last_error = error
            if partial.exists():
                partial.unlink()
            if attempt < 3:
                time.sleep(attempt * 2)
    raise RuntimeError(f"download failed after 3 attempts: {url}: {last_error}")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def download_opencity() -> None:
    target = DATA / "raw" / "opencity"
    metadata_path = target / "package_metadata.json"
    metadata = json.loads(fetch_bytes(OPENCITY_API))
    if not metadata.get("success"):
        raise RuntimeError("OpenCity CKAN API did not return success")
    write_json(metadata_path, metadata)

    resources = metadata["result"]["resources"]
    manifest = {
        "source": "OpenCity CKAN / CPCB Delhi Hourly Air Quality Reports",
        "source_url": OPENCITY_API,
        "downloaded_at": now(),
        "resource_count": len(resources),
        "resources": [],
    }
    for index, resource in enumerate(resources, start=1):
        resource_url = resource.get("url")
        if not resource_url:
            continue
        filename = Path(resource_url.split("/")[-1]).name or f"resource-{index}.csv"
        destination = target / filename
        print(f"[{index:02d}/{len(resources):02d}] {resource.get('name', filename)}", flush=True)
        record = {
            "name": resource.get("name"),
            "resource_id": resource.get("id"),
            "url": resource_url,
            "format": resource.get("format"),
            "advertised_size": resource.get("size"),
            "local": str(destination.relative_to(ROOT)),
        }
        try:
            record.update(download_file(resource_url, destination, advertised_size=resource.get("size")))
        except RuntimeError as error:
            record.update({"status": "error", "error": str(error)})
            print(f"  ERROR: {error}", file=sys.stderr, flush=True)
        manifest["resources"].append(record)
        write_json(target / "manifest.json", manifest)
    write_json(target / "manifest.json", manifest)


def download_openmeteo_weather() -> None:
    target = DATA / "raw" / "weather"
    query = {
        "latitude": "28.6139",
        "longitude": "77.2090",
        "start_date": "2017-01-01",
        "end_date": "2025-12-31",
        "hourly": ",".join(
            [
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation",
                "wind_speed_10m",
                "wind_direction_10m",
                "surface_pressure",
                "cloud_cover",
            ]
        ),
        "timezone": "Asia/Kolkata",
        "format": "json",
    }
    url = OPENMETEO_ARCHIVE + "?" + urlencode(query)
    destination = target / "delhi-central-hourly-2017-2025.json"
    print("Downloading Open-Meteo historical weather", flush=True)
    result = download_file(url, destination)
    write_json(target / "manifest.json", {"source_url": url, "downloaded_at": now(), **result})


def download_openmeteo_air_quality() -> None:
    target = DATA / "raw" / "cams"
    query = {
        "latitude": "28.6139",
        "longitude": "77.2090",
        "start_date": "2023-01-01",
        "end_date": "2025-12-31",
        "hourly": "pm2_5,pm10,nitrogen_dioxide,ozone,carbon_monoxide,sulphur_dioxide",
        "timezone": "Asia/Kolkata",
        "domains": "cams_global",
        "format": "json",
    }
    url = OPENMETEO_AIR + "?" + urlencode(query)
    destination = target / "delhi-central-cams-global-hourly-2023-2025.json"
    print("Downloading Open-Meteo CAMS air-quality history", flush=True)
    try:
        result = download_file(url, destination)
    except RuntimeError as error:
        result = {"status": "error", "error": str(error)}
        print(f"  WARNING: {error}", file=sys.stderr)
    write_json(target / "manifest.json", {"source_url": url, "downloaded_at": now(), **result})


def download_references() -> None:
    target = DATA / "reference"
    references = {
        "cpcb-national-aqi-calculation.pdf": "https://airquality.cpcb.gov.in/ccr_docs/How_AQI_Calculated.pdf",
        "cpcb-national-aqi-final-report.pdf": "https://airquality.cpcb.gov.in/ccr_docs/FINAL-REPORT_AQI_.pdf",
        "cpcb-delhi-ncr-stations.pdf": "https://airquality.cpcb.gov.in/ccr_docs/caaqms_list_NCR.pdf",
    }
    manifest = {"downloaded_at": now(), "files": []}
    for filename, url in references.items():
        print(f"Downloading reference {filename}", flush=True)
        try:
            result = download_file(url, target / filename)
        except RuntimeError as error:
            result = {"status": "error", "error": str(error)}
            print(f"  WARNING: {error}", file=sys.stderr)
        manifest["files"].append({"filename": filename, "url": url, **result})
    write_json(target / "manifest.json", manifest)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=("all", "opencity", "weather", "cams", "references"),
        default="all",
    )
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    if args.source in ("all", "opencity"):
        download_opencity()
    if args.source in ("all", "weather"):
        download_openmeteo_weather()
    if args.source in ("all", "cams"):
        download_openmeteo_air_quality()
    if args.source in ("all", "references"):
        download_references()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
