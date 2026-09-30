#!/usr/bin/env python3
"""Poll the local AirCast API to build a deduplicated WAQI history archive.

Run beside the API for a persistent two-station time series. The API writes only
new source timestamps, so polling more frequently than WAQI updates is safe.
"""

from __future__ import annotations

import argparse
import json
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def collect(api_url: str) -> dict[str, object]:
    url = f"{api_url.rstrip('/')}/live/stations/forecast?horizon=1&refresh=true"
    request = Request(url, headers={"Accept": "application/json", "User-Agent": "Delhi-AirCast-Collector/1.0"})
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--interval-minutes", type=float, default=60)
    parser.add_argument("--once", action="store_true", help="collect one sample and exit")
    args = parser.parse_args()
    if args.interval_minutes <= 0:
        parser.error("--interval-minutes must be positive")
    while True:
        try:
            result = collect(args.api_url)
            statuses = ", ".join(
                f"{row.get('station_name', row.get('station_id'))}: {row.get('quality_status')}"
                for row in result.get("forecasts", [])
            )
            print(f"Collected {result.get('issued_at_utc')}: {statuses}", flush=True)
        except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
            print(f"Collection failed ({type(error).__name__}); will retry next interval.", flush=True)
        if args.once:
            break
        time.sleep(args.interval_minutes * 60)


if __name__ == "__main__":
    main()
