import importlib.util
from pathlib import Path


_MODULE_PATH = Path(__file__).parents[1] / "scripts" / "normalize_cpcb_legacy_aqi.py"
_SPEC = importlib.util.spec_from_file_location("normalize_cpcb_legacy_aqi", _MODULE_PATH)
assert _SPEC and _SPEC.loader
legacy = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(legacy)


def test_parse_rows_expands_month_day_hour_grid_to_ist_and_utc():
    rows = [
        ["Year", "2020"],
        ["January-2020", "00:00:00", "01:00:00"],
        ["1", "101", "102"],
    ]
    result = legacy.parse_rows(rows, "legacy_test", "Test")
    assert len(result) == 2
    assert result[0]["aqi"] == 101
    assert str(result[0]["timestamp_utc"]) == "2019-12-31 18:30:00+00:00"
