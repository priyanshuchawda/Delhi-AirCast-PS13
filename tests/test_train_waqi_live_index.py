import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_MODULE_PATH = Path(__file__).parents[1] / "scripts" / "train_waqi_live_index.py"
_SPEC = importlib.util.spec_from_file_location("train_waqi_live_index", _MODULE_PATH)
assert _SPEC and _SPEC.loader
train_waqi_live_index = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = train_waqi_live_index
_SPEC.loader.exec_module(train_waqi_live_index)

_features = train_waqi_live_index._features
_history_features = train_waqi_live_index._history_features
_asof_weather_features = train_waqi_live_index._asof_weather_features
_weather_candidate_promoted = train_waqi_live_index._weather_candidate_promoted
pm25_to_us_aqi_proxy = train_waqi_live_index.pm25_to_us_aqi_proxy


def test_pm25_to_waqi_scale_proxy_interpolates_and_clips():
    assert pm25_to_us_aqi_proxy(0) == 0
    assert pm25_to_us_aqi_proxy(12) == 50
    assert pm25_to_us_aqi_proxy(35.4) == 100
    assert pm25_to_us_aqi_proxy(1000) == 500
    assert np.isnan(pm25_to_us_aqi_proxy(None))


def test_features_use_local_clock_and_station_identity():
    frame = pd.DataFrame(
        {
            "station_id": ["site_107", "site_124"],
            "timestamp_utc": pd.to_datetime(["2025-01-01T00:00:00Z", "2025-01-01T00:00:00Z"], utc=True),
            "pm25": [12.0, 35.4],
        }
    )
    result = _features(frame)
    assert result.loc[0, "current_pm25_index"] == 50
    assert result.loc[1, "current_pm25_index"] == 100
    assert result.loc[0, "station_site_107"] == 1
    assert result.loc[1, "station_site_124"] == 1
    assert result.loc[0, "hour_sin"] == result.loc[1, "hour_sin"]


def test_history_features_use_only_past_values_and_same_time_neighbor():
    timestamps = pd.date_range("2025-01-01", periods=80, freq="h", tz="UTC")
    frame = pd.DataFrame(
        [
            {"station_id": station, "timestamp_utc": timestamp, "pm25": float(i + offset)}
            for station, offset in (("site_107", 0), ("site_124", 10))
            for i, timestamp in enumerate(timestamps)
        ]
    )
    features = _history_features(frame)
    pusa = features[(frame.station_id == "site_107")].reset_index(drop=True)
    # At hour 72, lag_1 is the previous hour (not current/future), and the
    # same-time neighbor feature belongs to R.K. Puram.
    assert pusa.loc[72, "lag_1h"] == pusa.loc[71, "current_pm25_index"]
    assert pusa.loc[72, "neighbor_current_index"] == features.loc[72 + len(timestamps), "current_pm25_index"]


def test_weather_join_never_uses_a_future_weather_timestamp():
    frame = pd.DataFrame({
        "timestamp_utc": pd.to_datetime(["2025-01-01T00:30:00Z", "2025-01-01T01:30:00Z"], utc=True),
    })
    weather = pd.DataFrame({
        "timestamp_utc": pd.to_datetime(["2025-01-01T00:00:00Z", "2025-01-01T01:00:00Z", "2025-01-01T02:00:00Z"], utc=True),
        "temperature_2m": [10.0, 11.0, 99.0],
    })
    result = _asof_weather_features(frame, weather)
    assert result["wx_temperature_2m"].tolist() == [10.0, 11.0]


def test_weather_promotion_requires_validation_and_material_test_gain():
    baseline = {"validation": {"mae_index_points": 20.0}, "test": {"mae_index_points": 20.0}}
    assert _weather_candidate_promoted(baseline, {
        "validation": {"mae_index_points": 19.9}, "test": {"mae_index_points": 19.7},
    })
    assert not _weather_candidate_promoted(baseline, {
        "validation": {"mae_index_points": 20.1}, "test": {"mae_index_points": 18.0},
    })
    assert not _weather_candidate_promoted(baseline, {
        "validation": {"mae_index_points": 19.9}, "test": {"mae_index_points": 19.95},
    })
