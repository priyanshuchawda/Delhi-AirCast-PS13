import importlib.util
import sys
from pathlib import Path

import pandas as pd


_MODULE_PATH = Path(__file__).parents[1] / "scripts" / "train_multipollutant_aqi.py"
_SPEC = importlib.util.spec_from_file_location("train_multipollutant_aqi", _MODULE_PATH)
assert _SPEC and _SPEC.loader
train_multipollutant_aqi = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = train_multipollutant_aqi
_SPEC.loader.exec_module(train_multipollutant_aqi)


def test_training_split_purges_labels_that_cross_test_boundary():
    times = pd.Series(pd.to_datetime(["2025-09-30T17:00:00Z", "2025-09-30T18:00:00Z", "2025-10-01T00:00:00Z"], utc=True))
    assert train_multipollutant_aqi._training_mask(times, 6).tolist() == [True, False, False]


def test_aqi_metrics_requires_three_valid_pollutants_including_pm():
    valid = train_multipollutant_aqi._aqi_metrics(
        [{"pm25": 25.0, "pm10": 40.0, "no2": 20.0}],
        [{"pm25": 25.0, "pm10": 40.0, "no2": 20.0}],
    )
    insufficient = train_multipollutant_aqi._aqi_metrics([{"pm25": 25.0, "no2": 20.0}], [{"pm25": 25.0, "no2": 20.0}])
    assert valid["rows"] == 1
    assert valid["aqi_mae"] == 0.0
    assert insufficient["rows"] == 0
