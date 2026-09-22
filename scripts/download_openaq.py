#!/usr/bin/env python3
"""Download reproducible OpenAQ Delhi/NCR batches.

The catalog and latest batches preserve the API responses. Historical data is
downloaded as hourly averages for the selected sensors, one JSON file per
sensor and time window, so interrupted runs can resume without re-downloading
completed files.

Authentication is read only from OPENAQ_API_KEY. It is never written to the
project or to a manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "data" / "raw" / "openaq"
BASE_URL = "https://api.openaq.org/v3"
USER_AGENT = "Delhi-AirCast/0.1 (OpenAQ batch acquisition; academic research)"
DELHI_NCR_BBOX = "76.8,28.4,77.4,28.9"
CORE_PARAMETER_IDS = (1, 2, 3, 4, 5, 6)  # PM10, PM2.5, O3, CO, NO2, SO2


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_datetime(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    partial.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    partial.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class OpenAQClient:
    def __init__(self, api_key: str, min_interval: float = 1.0) -> None:
        if len(api_key) != 64 or any(c not in "0123456789abcdef" for c in api_key):
            raise ValueError("OPENAQ_API_KEY must be the 64-character OpenAQ key")
        self.api_key = api_key
        self.min_interval = min_interval
        self.last_request = 0.0

    def get(self, path: str, params: dict[str, object] | None = None) -> tuple[dict, dict]:
        url = f"{BASE_URL}{path}"
        if params:
            url += "?" + urlencode(params)
        for attempt in range(1, 5):
            wait = self.min_interval - (time.monotonic() - self.last_request)
            if wait > 0:
                time.sleep(wait)
            request = Request(
                url,
                headers={
                    "Accept": "application/json",
                    "User-Agent": USER_AGENT,
                    "X-API-Key": self.api_key,
                },
            )
            try:
                # Pace request starts, so response latency does not get added
                # to the one-request-per-second account limit.
                self.last_request = time.monotonic()
                with urlopen(request, timeout=60) as response:
                    return json.loads(response.read()), dict(response.headers.items())
            except HTTPError as error:
                self.last_request = time.monotonic()
                body = error.read().decode("utf-8", "replace")
                if error.code in (429, 500, 502, 503, 504) and attempt < 4:
                    retry_after = error.headers.get("Retry-After")
                    delay = float(retry_after) if retry_after else attempt * 3
                    print(
                        f"  retry {attempt}/3 for HTTP {error.code} after {delay:.0f}s",
                        file=sys.stderr,
                        flush=True,
                    )
                    time.sleep(delay)
                    continue
                raise RuntimeError(f"GET {path} failed with HTTP {error.code}: {body[:300]}")
            except (OSError, URLError, TimeoutError, json.JSONDecodeError) as error:
                self.last_request = time.monotonic()
                if attempt < 4:
                    delay = attempt * 3
                    print(
                        f"  retry {attempt}/3 after transport error: {error}",
                        file=sys.stderr,
                        flush=True,
                    )
                    time.sleep(delay)
                    continue
                raise RuntimeError(f"GET {path} failed: {error}") from error
        raise AssertionError("unreachable")


def fetch_pages(
    client: OpenAQClient, path: str, params: dict[str, object], limit: int = 1000
) -> dict:
    all_results: list[dict] = []
    page = 1
    first_meta: dict = {}
    while True:
        payload, _headers = client.get(path, {**params, "limit": limit, "page": page})
        if page == 1:
            first_meta = payload.get("meta", {})
        results = payload.get("results", [])
        all_results.extend(results)
        if len(results) < limit:
            break
        page += 1
    return {"meta": {**first_meta, "pageCount": page}, "results": all_results}


def load_locations() -> dict:
    path = TARGET / "catalog" / "locations_delhi_ncr.json"
    if not path.exists():
        raise RuntimeError("Run --batch catalog first")
    return json.loads(path.read_text(encoding="utf-8"))


def selected_sensors(
    locations_payload: dict,
    parameter_ids: set[int],
    monitors_only: bool,
    start: datetime | None = None,
    end: datetime | None = None,
) -> list[dict]:
    sensors: dict[int, dict] = {}
    for location in locations_payload.get("results", []):
        if monitors_only and not location.get("isMonitor"):
            continue
        first = (location.get("datetimeFirst") or {}).get("utc")
        last = (location.get("datetimeLast") or {}).get("utc")
        try:
            first_dt = datetime.fromisoformat(first.replace("Z", "+00:00")) if first else None
            last_dt = datetime.fromisoformat(last.replace("Z", "+00:00")) if last else None
        except ValueError:
            first_dt = last_dt = None
        if start and last_dt and last_dt < start:
            continue
        if end and first_dt and first_dt > end:
            continue
        for sensor in location.get("sensors", []):
            parameter = sensor.get("parameter") or {}
            sensor_id = sensor.get("id")
            parameter_id = parameter.get("id")
            if sensor_id and parameter_id in parameter_ids:
                sensors[int(sensor_id)] = {
                    "sensor_id": int(sensor_id),
                    "location_id": location.get("id"),
                    "location_name": location.get("name"),
                    "parameter_id": parameter_id,
                    "parameter": parameter.get("name"),
                    "units": parameter.get("units"),
                }
    return sorted(sensors.values(), key=lambda row: (row["location_id"], row["parameter_id"], row["sensor_id"]))


def run_catalog(client: OpenAQClient, bbox: str) -> None:
    catalog = TARGET / "catalog"
    catalog.mkdir(parents=True, exist_ok=True)
    locations = fetch_pages(client, "/locations", {"bbox": bbox})
    write_json(catalog / "locations_delhi_ncr.json", locations)

    metadata_endpoints = {
        "parameters": "/parameters",
        "countries": "/countries",
        "providers": "/providers",
        "owners": "/owners",
        "manufacturers": "/manufacturers",
        "instruments": "/instruments",
        "licenses": "/licenses",
    }
    for name, path in metadata_endpoints.items():
        payload = fetch_pages(client, path, {})
        write_json(catalog / f"{name}.json", payload)
        print(f"catalog {name}: {len(payload.get('results', []))} records", flush=True)

    locations_rows = locations.get("results", [])
    sensors = selected_sensors(locations, set(CORE_PARAMETER_IDS), monitors_only=False)
    summary = {
        "generated_at": utc_now().isoformat(),
        "source": "OpenAQ API v3",
        "bbox": bbox,
        "location_count": len(locations_rows),
        "monitor_location_count": sum(1 for row in locations_rows if row.get("isMonitor")),
        "sensor_count": len({row["sensor_id"] for row in sensors}),
        "core_parameter_ids": list(CORE_PARAMETER_IDS),
        "parameters": sorted({row["parameter"] for row in sensors}),
        "coverage_first_utc": min(
            ((row.get("datetimeFirst") or {}).get("utc") for row in locations_rows if row.get("datetimeFirst")),
            default=None,
        ),
        "coverage_last_utc": max(
            ((row.get("datetimeLast") or {}).get("utc") for row in locations_rows if row.get("datetimeLast")),
            default=None,
        ),
    }
    write_json(catalog / "summary.json", summary)
    print(
        f"catalog locations: {summary['location_count']} locations, "
        f"{summary['monitor_location_count']} monitors, {summary['sensor_count']} core sensors",
        flush=True,
    )


def run_latest(client: OpenAQClient, monitors_only: bool) -> None:
    locations = load_locations()
    target = TARGET / "latest"
    manifest = {
        "generated_at": utc_now().isoformat(),
        "source": "OpenAQ API v3 / locations/{id}/latest",
        "monitors_only": monitors_only,
        "records": [],
    }
    selected = [
        row for row in locations.get("results", []) if not monitors_only or row.get("isMonitor")
    ]
    for index, location in enumerate(selected, start=1):
        location_id = location["id"]
        destination = target / f"location_{location_id}.json"
        print(f"latest [{index}/{len(selected)}] {location_id} {location.get('name')}", flush=True)
        payload, headers = client.get(f"/locations/{location_id}/latest", {"limit": 1000})
        write_json(destination, payload)
        manifest["records"].append(
            {
                "location_id": location_id,
                "location_name": location.get("name"),
                "result_count": len(payload.get("results", [])),
                "path": str(destination.relative_to(ROOT)),
                "bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
                "rate_limit_remaining": headers.get("x-ratelimit-remaining"),
            }
        )
        write_json(target / "manifest.json", manifest)
    write_json(target / "manifest.json", manifest)


def run_hourly(
    client: OpenAQClient,
    start: datetime,
    end: datetime,
    parameter_ids: set[int],
    monitors_only: bool,
) -> None:
    locations = load_locations()
    sensors = selected_sensors(locations, parameter_ids, monitors_only, start, end)
    batch_name = f"{start.date().isoformat()}_{end.date().isoformat()}"
    target = TARGET / "hours" / batch_name
    manifest = {
        "generated_at": utc_now().isoformat(),
        "source": "OpenAQ API v3 / sensors/{id}/hours",
        "start_utc": iso_datetime(start),
        "end_utc": iso_datetime(end),
        "parameter_ids": sorted(parameter_ids),
        "monitors_only": monitors_only,
        "sensor_count": len(sensors),
        "records": [],
    }
    for index, sensor in enumerate(sensors, start=1):
        sensor_id = sensor["sensor_id"]
        destination = target / f"sensor_{sensor_id}_p{sensor['parameter_id']}.json"
        if destination.exists() and destination.stat().st_size > 0:
            payload = json.loads(destination.read_text(encoding="utf-8"))
            print(f"hours [{index}/{len(sensors)}] exists sensor={sensor_id}", flush=True)
        else:
            print(
                f"hours [{index}/{len(sensors)}] sensor={sensor_id} "
                f"location={sensor['location_id']} parameter={sensor['parameter']}",
                flush=True,
            )
            payload = fetch_pages(
                client,
                f"/sensors/{sensor_id}/hours",
                {"datetime_from": iso_datetime(start), "datetime_to": iso_datetime(end)},
            )
            write_json(destination, payload)
        manifest["records"].append(
            {
                **sensor,
                "result_count": len(payload.get("results", [])),
                "path": str(destination.relative_to(ROOT)),
                "bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
            }
        )
        write_json(target / "manifest.json", manifest)
    write_json(target / "manifest.json", manifest)
    print(f"hourly batch complete: {target}", flush=True)


def parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", choices=("catalog", "latest", "hourly"), required=True)
    parser.add_argument("--bbox", default=DELHI_NCR_BBOX)
    parser.add_argument("--start", help="UTC ISO datetime for hourly batch")
    parser.add_argument("--end", help="UTC ISO datetime for hourly batch")
    parser.add_argument("--days", type=int, default=30, help="Days before now for hourly batch")
    parser.add_argument(
        "--parameter-ids",
        default=",".join(str(value) for value in CORE_PARAMETER_IDS),
        help="Comma-separated OpenAQ parameter IDs",
    )
    parser.add_argument("--all-locations", action="store_true", help="Include non-monitor locations")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAQ_API_KEY", "").strip()
    if not api_key:
        parser.error("OPENAQ_API_KEY is required")
    try:
        client = OpenAQClient(api_key)
        parameter_ids = {int(value) for value in args.parameter_ids.split(",") if value}
        monitors_only = not args.all_locations
        if args.batch == "catalog":
            run_catalog(client, args.bbox)
        elif args.batch == "latest":
            run_latest(client, monitors_only)
        else:
            end = parse_datetime(args.end) if args.end else utc_now()
            start = parse_datetime(args.start) if args.start else end - timedelta(days=args.days)
            if start >= end:
                parser.error("--start must be earlier than --end")
            run_hourly(client, start, end, parameter_ids, monitors_only)
    except (ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
