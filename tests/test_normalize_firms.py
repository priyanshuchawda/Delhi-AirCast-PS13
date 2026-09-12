import importlib.util
from pathlib import Path

import pandas as pd


_MODULE_PATH = Path(__file__).parents[1] / "scripts" / "normalize_firms.py"
_SPEC = importlib.util.spec_from_file_location("normalize_firms", _MODULE_PATH)
assert _SPEC and _SPEC.loader
normalize_firms = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(normalize_firms)


def test_daily_features_separate_delhi_and_upwind_regions():
    events = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(["2024-10-01 01:00", "2024-10-01 02:00"], utc=True),
            "latitude": [28.6, 30.5],
            "longitude": [77.2, 75.5],
            "frp": [10.0, 20.0],
            "confidence_score": [90.0, 50.0],
            "satellite": ["A", "A"],
            "instrument": ["VIIRS", "VIIRS"],
            "daynight": ["D", "D"],
            "source": ["VIIRS_SNPP_SP", "VIIRS_SNPP_SP"],
            "source_file": ["a.csv", "a.csv"],
        }
    )
    result = normalize_firms._daily_features(events)
    assert result.loc[0, "fire_count"] == 2
    assert result.loc[0, "fire_count_delhi"] == 1
    assert result.loc[0, "fire_count_northwest"] == 1
    assert result.loc[0, "fire_high_confidence_count"] == 1
