import importlib.util
import sys
from pathlib import Path


_MODULE_PATH = Path(__file__).parents[1] / "scripts" / "compare_model_strategies.py"
_SPEC = importlib.util.spec_from_file_location("compare_model_strategies", _MODULE_PATH)
assert _SPEC and _SPEC.loader
compare_model_strategies = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = compare_model_strategies
_SPEC.loader.exec_module(compare_model_strategies)


def test_feature_families_are_nested_and_exclude_targets():
    columns = ["pm25_current", "neighbor_pm25_mean_current", "wx_temperature", "cams_pm25", "firms_count", "target_pm25_6h"]
    families = compare_model_strategies._feature_sets(columns)
    assert "target_pm25_6h" not in families["CPCB_sensor_spatiotemporal"]
    assert "wx_temperature" not in families["CPCB_sensor_spatiotemporal"]
    assert "wx_temperature" in families["core_plus_weather"]
    assert "cams_pm25" not in families["core_plus_weather"]
    assert "cams_pm25" in families["core_plus_weather_cams"]
    assert "firms_count" in families["all_sources"]
    assert len(families["core_plus_weather_cams"]) <= len(families["all_sources"])
