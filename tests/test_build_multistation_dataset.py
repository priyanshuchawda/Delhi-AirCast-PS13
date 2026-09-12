import importlib.util
from pathlib import Path

import pandas as pd


_MODULE_PATH = Path(__file__).parents[1] / "scripts" / "build_multistation_dataset.py"
_SPEC = importlib.util.spec_from_file_location("build_multistation_dataset", _MODULE_PATH)
assert _SPEC and _SPEC.loader
build_multistation_dataset = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(build_multistation_dataset)


def test_add_temporal_features_uses_past_for_lags_and_future_for_targets():
    frame = pd.DataFrame(
        {
            "station_id": ["a"] * 4,
            "station_name": ["A"] * 4,
            "timestamp_utc": pd.date_range("2025-01-01", periods=4, freq="h", tz="UTC"),
            "pm25": [10.0, 20.0, 30.0, 40.0],
            "pm10": [20.0] * 4,
            "no2": [1.0] * 4,
            "so2": [1.0] * 4,
            "co": [1.0] * 4,
            "o3": [1.0] * 4,
            "nh3": [1.0] * 4,
            "temperature_c": [10.0] * 4,
            "relative_humidity_pct": [50.0] * 4,
            "wind_speed_mps": [1.0] * 4,
            "wind_direction_deg": [90.0] * 4,
            "rain_mm": [0.0] * 4,
        }
    )
    result = build_multistation_dataset._add_temporal_features(frame)
    assert pd.isna(result.loc[0, "pm25_lag_1h"])
    assert result.loc[1, "pm25_lag_1h"] == 10.0
    assert result.loc[0, "target_pm25_1h"] == 20.0
    assert pd.isna(result.loc[3, "target_pm25_1h"])
