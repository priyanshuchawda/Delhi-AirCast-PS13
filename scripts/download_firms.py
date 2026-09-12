#!/usr/bin/env python3
"""Download NASA FIRMS active-fire CSV windows for Delhi and its upwind region.

The MAP_KEY is intentionally accepted only through ``FIRMS_MAP_KEY`` or
``NASA_FIRMS_MAP_KEY``.  It is never written to a manifest, URL log, or
repository file.  Each successful API window is saved atomically so an
interrupted multi-day acquisition can be resumed safely.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
from datetime import date, timedelta
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "raw" / "firms"
DEFAULT_BBOX = "74,25,79.5,31.5"
DEFAULT_SOURCES = ("MODIS_SP", "VIIRS_SNPP_SP", "VIIRS_NOAA20_SP")
API_ROOT = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
MAX_DAYS_PER_REQUEST = 5
USER_AGENT = "Delhi-AirCast-PS13/0.1 (research data acquisition)"


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid ISO date: {value}") from exc


def validate_bbox(value: str) -> str:
    parts = value.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("bbox must be west,south,east,north")
    try:
        west, south, east, north = (float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("bbox coordinates must be numeric") from exc
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise argparse.ArgumentTypeError("bbox coordinates are out of order or range")
    return ",".join(f"{part:g}" for part in (west, south, east, north))


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def windows(start: date, end: date, max_days: int = MAX_DAYS_PER_REQUEST):
    if end < start:
        raise ValueError("end date must not precede start date")
    current = start
    while current <= end:
        window_end = min(current + timedelta(days=max_days - 1), end)
        yield current, window_end
        current = window_end + timedelta(days=1)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _key_from_environment() -> str:
    key = os.environ.get("FIRMS_MAP_KEY") or os.environ.get("NASA_FIRMS_MAP_KEY")
    if not key:
        raise RuntimeError(
            "FIRMS_MAP_KEY (or NASA_FIRMS_MAP_KEY) is required; keep it outside the repository"
        )
    key = key.strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{16,128}", key):
        raise RuntimeError("FIRMS MAP_KEY has an unexpected format")
    return key


def _api_url(key: str, source: str, bbox: str, start: date, end: date) -> str:
    days = (end - start).days + 1
    return (
        f"{API_ROOT}/{quote(key, safe='')}/{quote(source, safe='')}/"
        f"{quote(bbox, safe=',')}/{days}/{start.isoformat()}"
    )


def _download(url: str, destination: Path, timeout: int, retries: int) -> int:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/csv"})
            with urlopen(request, timeout=timeout) as response:
                if response.status != 200:
                    raise RuntimeError(f"FIRMS returned HTTP {response.status}")
                with NamedTemporaryFile("wb", dir=destination.parent, delete=False) as temp:
                    temporary = Path(temp.name)
                    while block := response.read(1024 * 1024):
                        temp.write(block)
            temporary.replace(destination)
            return sum(1 for _ in csv.DictReader(destination.open(newline="", encoding="utf-8")))
        except (HTTPError, URLError, TimeoutError, OSError, RuntimeError) as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(min(60.0, 2.0**attempt))
    if last_error is None:
        raise RuntimeError("FIRMS download failed without an error")
    if isinstance(last_error, HTTPError):
        raise RuntimeError(f"FIRMS returned HTTP {last_error.code}") from last_error
    raise RuntimeError(f"FIRMS request failed: {type(last_error).__name__}") from last_error


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def download(
    start: date,
    end: date,
    bbox: str = DEFAULT_BBOX,
    sources: tuple[str, ...] = DEFAULT_SOURCES,
    output_dir: Path = DEFAULT_OUTPUT,
    sleep_seconds: float = 0.25,
    timeout: int = 120,
    retries: int = 4,
    force: bool = False,
    max_windows: int | None = None,
) -> dict[str, object]:
    key = _key_from_environment()
    bbox = validate_bbox(bbox)
    output_root = output_dir / f"bbox_{slug(bbox)}"
    manifest_path = output_root / "manifest.json"
    existing = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    records = {
        (item["source"], item["start_date"], item["end_date"]): item
        for item in existing.get("records", [])
    }
    errors: list[dict[str, str]] = []
    requested_windows = list(windows(start, end))
    if max_windows is not None:
        requested_windows = requested_windows[:max_windows]

    for source in sources:
        source_dir = output_root / slug(source)
        source_dir.mkdir(parents=True, exist_ok=True)
        for window_start, window_end in requested_windows:
            key_tuple = (source, window_start.isoformat(), window_end.isoformat())
            destination = source_dir / f"{window_start}_{window_end}.csv"
            previous = records.get(key_tuple)
            if (
                not force
                and previous
                and previous.get("status") == "success"
                and destination.exists()
                and previous.get("sha256") == sha256_file(destination)
            ):
                continue
            try:
                row_count = _download(
                    _api_url(key, source, bbox, window_start, window_end),
                    destination,
                    timeout,
                    retries,
                )
                records[key_tuple] = {
                    "source": source,
                    "start_date": window_start.isoformat(),
                    "end_date": window_end.isoformat(),
                    "status": "success",
                    "rows": row_count,
                    "bytes": destination.stat().st_size,
                    "sha256": sha256_file(destination),
                    "path": str(destination.relative_to(ROOT)),
                }
            except RuntimeError as exc:
                errors.append({"source": source, "start_date": window_start.isoformat(), "end_date": window_end.isoformat(), "error": str(exc)})
                records[key_tuple] = {
                    "source": source,
                    "start_date": window_start.isoformat(),
                    "end_date": window_end.isoformat(),
                    "status": "error",
                    "error": str(exc),
                }
            _write_json(
                manifest_path,
                {
                    "source": "NASA FIRMS area CSV API",
                    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "bbox": bbox,
                    "sources": list(sources),
                    "max_days_per_request": MAX_DAYS_PER_REQUEST,
                    "records": sorted(records.values(), key=lambda item: (item["source"], item["start_date"])),
                },
            )
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    result = {
        "manifest": str(manifest_path.relative_to(ROOT)),
        "requested_windows": len(requested_windows) * len(sources),
        "successful_windows": sum(item.get("status") == "success" for item in records.values()),
        "errors": errors,
    }
    if errors:
        raise RuntimeError(json.dumps(result))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", required=True, type=parse_date)
    parser.add_argument("--end-date", required=True, type=parse_date, help="inclusive ISO date")
    parser.add_argument("--bbox", default=DEFAULT_BBOX, type=validate_bbox)
    parser.add_argument("--source", action="append", dest="sources", help="repeatable FIRMS source")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sleep-seconds", type=float, default=0.25)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-windows", type=int, help="limit windows for a smoke test")
    args = parser.parse_args()
    sources = tuple(args.sources or DEFAULT_SOURCES)
    result = download(
        args.start_date,
        args.end_date,
        bbox=args.bbox,
        sources=sources,
        output_dir=args.output_dir,
        sleep_seconds=args.sleep_seconds,
        timeout=args.timeout,
        retries=args.retries,
        force=args.force,
        max_windows=args.max_windows,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
