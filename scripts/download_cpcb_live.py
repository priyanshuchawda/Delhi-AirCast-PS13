#!/usr/bin/env python3
"""Download the official CPCB live AQI resource from data.gov.in.

The data.gov.in API key is read only from ``DATAGOV_API_KEY`` or
``DATA_GOV_API_KEY``. It is never written to a URL log, manifest, or Git.
Pages are stored as raw JSON and a long-form Parquet snapshot is rebuilt from
all successful pages so interrupted downloads can be resumed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESOURCE_ID = "3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"
API_ROOT = f"https://api.data.gov.in/resource/{RESOURCE_ID}"
DEFAULT_OUTPUT = ROOT / "data" / "raw" / "cpcb_live"
USER_AGENT = "Delhi-AirCast-PS13/0.1 (research data acquisition)"


def _api_key() -> str:
    key = os.environ.get("DATAGOV_API_KEY") or os.environ.get("DATA_GOV_API_KEY")
    if not key:
        raise RuntimeError(
            "DATAGOV_API_KEY (or DATA_GOV_API_KEY) is required; keep it outside the repository"
        )
    key = key.strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{16,128}", key):
        raise RuntimeError("data.gov.in API key has an unexpected format")
    return key


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    temporary.replace(path)


def _fetch_page(
    key: str,
    limit: int,
    offset: int,
    state: str,
    city: str | None,
    timeout: int,
) -> dict[str, object]:
    params = {
        "api-key": key,
        "format": "json",
        "limit": str(limit),
        "offset": str(offset),
        "filters[state]": state,
    }
    if city:
        params["filters[city]"] = city
    request = Request(
        f"{API_ROOT}?{urlencode(params)}",
        headers={"User-Agent": USER_AGENT},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise RuntimeError(f"data.gov.in returned HTTP {response.status}")
            payload = json.load(response)
    except HTTPError as exc:
        raise RuntimeError(f"data.gov.in returned HTTP {exc.code}") from exc
    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"data.gov.in request failed: {type(exc).__name__}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise RuntimeError("data.gov.in response did not contain a records list")
    return payload


def _normalise_records(records: list[dict[str, object]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    frame = pd.json_normalize(records)
    frame.columns = [str(column).strip().lower().replace(" ", "_") for column in frame.columns]
    for column in (
        "pollutant_min",
        "pollutant_max",
        "pollutant_avg",
        "min_value",
        "max_value",
        "avg_value",
        "aqi",
        "latitude",
        "longitude",
    ):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in ("last_update", "last_updated"):
        if column in frame:
            parsed = pd.to_datetime(frame[column], errors="coerce", dayfirst=True)
            if parsed.dt.tz is None:
                parsed = parsed.dt.tz_localize("Asia/Kolkata")
            frame["timestamp_utc"] = parsed.dt.tz_convert("UTC")
            break
    return frame


def download(
    output_dir: Path = DEFAULT_OUTPUT,
    state: str = "Delhi",
    city: str | None = None,
    limit: int = 1000,
    max_pages: int | None = None,
    timeout: int = 60,
    refresh: bool = False,
) -> dict[str, object]:
    if not 1 <= limit <= 10000:
        raise ValueError("limit must be between 1 and 10000")
    key = _api_key()
    pages_dir = output_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    page_summaries = []
    offset = 0
    page_number = 0
    while True:
        if max_pages is not None and page_number >= max_pages:
            break
        page_number += 1
        page_path = pages_dir / f"page_{offset:09d}.json"
        if page_path.exists() and not refresh:
            payload = json.loads(page_path.read_text(encoding="utf-8"))
        else:
            payload = _fetch_page(key, limit, offset, state, city, timeout)
            _write_json(page_path, payload)
        page_records = payload.get("records", [])
        records.extend(page_records)
        page_summaries.append(
            {
                "page": page_number,
                "offset": offset,
                "count": len(page_records),
                "path": str(page_path.relative_to(ROOT)),
                "sha256": _sha256(page_path),
            }
        )
        if len(page_records) < limit:
            break
        offset += limit

    frame = _normalise_records(records)
    snapshot = output_dir / "cpcb_live_delhi.parquet"
    if not frame.empty:
        sort_columns = [
            column for column in ("timestamp_utc", "station", "pollutant_id") if column in frame
        ]
        frame = frame.drop_duplicates().sort_values(sort_columns or list(frame.columns))
        frame.to_parquet(snapshot, index=False)
    manifest = {
        "source": "CPCB real-time AQI via data.gov.in",
        "resource_id": RESOURCE_ID,
        "state": state,
        "city": city,
        "limit": limit,
        "pages": page_summaries,
        "record_count": len(frame),
        "columns": list(frame.columns),
        "snapshot": str(snapshot.relative_to(ROOT)) if snapshot.exists() else None,
        "snapshot_sha256": _sha256(snapshot) if snapshot.exists() else None,
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--state", default="Delhi")
    parser.add_argument("--city")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--max-pages", type=int)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    print(json.dumps(download(**vars(args)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
