import importlib.util
from pathlib import Path

import pandas as pd


_MODULE_PATH = Path(__file__).parents[1] / "scripts" / "analyze_cpcb_panel.py"
_SPEC = importlib.util.spec_from_file_location("analyze_cpcb_panel", _MODULE_PATH)
assert _SPEC and _SPEC.loader
analyze_cpcb_panel = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(analyze_cpcb_panel)


def test_station_key_matches_historical_and_live_agency_suffixes():
    assert analyze_cpcb_panel._station_key("CRRI Mathura Road, Delhi - IMD") == analyze_cpcb_panel._station_key(
        "CRRI Mathura Road, Delhi - IITM"
    )


def test_station_quality_reports_missing_hours_and_missingness():
    frame = pd.DataFrame(
        {
            "station_id": ["site_test"] * 3,
            "station_name": ["Example, Delhi - DPCC"] * 3,
            "timestamp_utc": pd.to_datetime(
                ["2025-01-01 00:00", "2025-01-01 01:00", "2025-01-01 03:00"], utc=True
            ),
            "pm25": [10.0, None, 30.0],
        }
    )
    summary = analyze_cpcb_panel.station_quality(frame)
    assert summary["expected_hourly_rows"] == 4
    assert summary["missing_hour_rows"] == 1
    assert summary["coverage_fraction"] == 0.75
    assert summary["pm25_non_null_rows"] == 2
    assert summary["missing_fraction"]["pm10"] == 1.0
