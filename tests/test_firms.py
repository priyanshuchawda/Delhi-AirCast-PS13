from datetime import date
import importlib.util
from pathlib import Path

import pytest


_MODULE_PATH = Path(__file__).parents[1] / "scripts" / "download_firms.py"
_SPEC = importlib.util.spec_from_file_location("download_firms", _MODULE_PATH)
assert _SPEC and _SPEC.loader
download_firms = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(download_firms)

DEFAULT_BBOX = download_firms.DEFAULT_BBOX
parse_date = download_firms.parse_date
validate_bbox = download_firms.validate_bbox
windows = download_firms.windows


def test_default_bbox_is_north_india_upwind_region():
    assert validate_bbox(DEFAULT_BBOX) == "74,25,79.5,31.5"


def test_windows_are_inclusive_and_max_five_days():
    result = list(windows(date(2024, 1, 1), date(2024, 1, 12)))
    assert result == [
        (date(2024, 1, 1), date(2024, 1, 5)),
        (date(2024, 1, 6), date(2024, 1, 10)),
        (date(2024, 1, 11), date(2024, 1, 12)),
    ]


def test_invalid_bbox_and_dates_are_rejected():
    with pytest.raises(Exception):
        validate_bbox("77,29,76,28")
    with pytest.raises(Exception):
        parse_date("2024/01/01")
