import importlib.util
from pathlib import Path

import pandas as pd


_MODULE_PATH = Path(__file__).parents[1] / "scripts" / "train_multistation_baseline.py"
_SPEC = importlib.util.spec_from_file_location("train_multistation_baseline", _MODULE_PATH)
assert _SPEC and _SPEC.loader
train_multistation_baseline = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(train_multistation_baseline)


def test_feature_frame_excludes_targets_and_adds_station_identity():
    frame = pd.DataFrame(
        {
            "station_id": ["a", "b"],
            "timestamp_utc": pd.date_range("2025-01-01", periods=2, tz="UTC"),
            "pm25": [10.0, 20.0],
            "pm25_current": [10.0, 20.0],
            "target_pm25_1h": [20.0, 30.0],
            "hour_sin": [0.0, 0.1],
        }
    )
    features, columns, station_codes = train_multistation_baseline._feature_frame(frame)
    assert "target_pm25_1h" not in columns
    assert "pm25" not in columns
    assert "station_a" in columns
    assert "station_b" in columns
    assert station_codes == {"a": 0, "b": 1}
    assert features["station_a"].tolist() == [1.0, 0.0]
