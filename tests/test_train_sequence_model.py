import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


_MODULE_PATH = Path(__file__).parents[1] / "scripts" / "train_sequence_model.py"
_SPEC = importlib.util.spec_from_file_location("train_sequence_model", _MODULE_PATH)
assert _SPEC and _SPEC.loader
train_sequence_model = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = train_sequence_model
_SPEC.loader.exec_module(train_sequence_model)


def _frame() -> pd.DataFrame:
    frame = {
        "station_id": ["a"] * 8,
        "timestamp_utc": pd.date_range("2025-01-01", periods=8, freq="h", tz="UTC"),
        "target_pm25_1h": np.arange(1, 9, dtype=float),
    }
    for index, column in enumerate(train_sequence_model.SEQUENCE_FEATURES):
        frame[column] = np.arange(8, dtype=float) + index
    return pd.DataFrame(frame)


def test_make_sequences_preserves_past_window_and_direct_target():
    bundle = train_sequence_model._make_sequences(_frame(), horizon=1, sequence_length=3, stride=1)
    assert bundle.values.shape[0] == 5
    assert bundle.values.shape[1] == 3
    assert bundle.targets.tolist() == [3.0, 4.0, 5.0, 6.0, 7.0]
    assert bundle.issue_times[0] == np.datetime64("2025-01-01T02:00:00")
