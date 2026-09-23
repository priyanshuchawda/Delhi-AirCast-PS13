import pandas as pd
import pytest
from fastapi import HTTPException

from delhi_aircast.api import AQIRequest, _multistation_feature_row, aqi, health, station_history, stations_forecast


def test_health_exposes_engine_and_local_data_state():
    payload = health()
    assert payload["status"] == "ok"
    assert payload["aqi_engine"] == "available"
    assert "unified_dataset" in payload
    assert "final_xgboost_horizons" in payload
    assert isinstance(payload["final_xgboost_horizons"], list)
    assert "serving_features" in payload


def test_station_forecast_rejects_unsupported_horizons():
    with pytest.raises(HTTPException) as error:
        stations_forecast(horizon=2)
    assert error.value.status_code == 400


def test_history_rejects_unbounded_requests():
    with pytest.raises(HTTPException) as error:
        station_history("site_105", hours=500)
    assert error.value.status_code == 400


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


def test_multistation_feature_row_keeps_aux_features_when_present():
    row = pd.Series(
        {
            "pm25_current": 42.0,
            "wx_temperature_c": 18.0,
            "cams_pm25_lag6h": 90.0,
            "target_pm25_1h": 999.0,
        }
    )
    vector = _multistation_feature_row(
        row,
        "site_105",
        ["pm25_current", "wx_temperature_c", "cams_pm25_lag6h", "station_site_105"],
    )
    assert vector.loc[0, "pm25_current"] == 42.0
    assert vector.loc[0, "wx_temperature_c"] == 18.0
    assert vector.loc[0, "cams_pm25_lag6h"] == 90.0
    assert vector.loc[0, "station_site_105"] == 1.0
