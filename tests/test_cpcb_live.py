import importlib.util
from pathlib import Path


_MODULE_PATH = Path(__file__).parents[1] / "scripts" / "download_cpcb_live.py"
_SPEC = importlib.util.spec_from_file_location("download_cpcb_live", _MODULE_PATH)
assert _SPEC and _SPEC.loader
download_cpcb_live = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(download_cpcb_live)


def test_normalise_records_keeps_long_pollutant_rows_and_numeric_values():
    frame = download_cpcb_live._normalise_records(
        [
            {
                "station": "North Campus",
                "pollutant_id": "PM2.5",
                "avg_value": "145",
                "latitude": "28.7",
                "longitude": "77.2",
                "last_update": "12-09-2026 10:00:00",
            }
        ]
    )
    assert frame.loc[0, "avg_value"] == 145
    assert frame.loc[0, "latitude"] == 28.7
    assert frame.loc[0, "station"] == "North Campus"
    assert str(frame.loc[0, "timestamp_utc"].tz) == "UTC"
    assert str(frame.loc[0, "timestamp_utc"]) == "2026-09-12 04:30:00+00:00"


def test_resource_id_is_the_official_cpcb_live_resource():
    assert download_cpcb_live.RESOURCE_ID == "3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"
