import pandas as pd

from delhi_aircast.api import AQIRequest, _multistation_feature_row, aqi, health


def test_health_exposes_engine_and_local_data_state():
    payload = health()
    assert payload["status"] == "ok"
    assert payload["aqi_engine"] == "available"


def test_aqi_endpoint_keeps_pm25_proxy_explicit():
    payload = aqi(AQIRequest(pm25=145))
    assert payload["value"] == 319
    assert payload["status"] == "pm25_derived_proxy"


def test_multistation_feature_row_adds_station_identity_without_future_fields():
    row = pd.Series({"pm25_current": 42.0, "target_pm25_1h": 999.0})
    vector = _multistation_feature_row(
        row,
        "site_105",
        ["pm25_current", "station_site_105", "station_site_113", "missing_feature"],
    )
    assert vector.loc[0, "pm25_current"] == 42.0
    assert vector.loc[0, "station_site_105"] == 1.0
    assert vector.loc[0, "station_site_113"] == 0.0
    assert pd.isna(vector.loc[0, "missing_feature"])
