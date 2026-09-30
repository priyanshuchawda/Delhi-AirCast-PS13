import pandas as pd
import pytest
from fastapi import HTTPException

from delhi_aircast.api import AQIRequest, _multistation_feature_row, aqi, health, live_stations_forecast, station_history, stations_forecast


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


def test_station_forecast_marks_old_artifact_as_backtest(monkeypatch):
    monkeypatch.setattr(
        "delhi_aircast.api._forecast_rows",
        lambda _horizon: [{"as_of_utc": "2025-12-31T00:00:00+00:00", "target_timestamp_utc": "2025-12-31T06:00:00+00:00"}],
    )
    payload = stations_forecast(horizon=6)
    assert payload["forecast_mode"] == "historical_backtest"
    assert payload["target_is_future"] is False


def test_live_forecast_uses_separate_index_model_and_freshness_gate(monkeypatch, tmp_path):
    from delhi_aircast import api

    model_dir = tmp_path / "models" / "waqi_live_index"
    model_dir.mkdir(parents=True)
    (model_dir / "model_3h.joblib").touch()
    monkeypatch.setattr(api, "LIVE_MODEL_DIR", model_dir)

    class StubModel:
        def predict(self, features):
            return [91.0]

    monkeypatch.setattr(api, "_load_bundle", lambda _path: {
        "model": StubModel(),
        "feature_columns": [
            "current_pm25_index", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
            "year_day_sin", "year_day_cos", "station_site_107", "station_site_124",
        ],
    })
    now = pd.Timestamp.now(tz="UTC")
    observations = {
        10124: {"pm25_index": 83.0, "observed_at": now - pd.Timedelta(minutes=10)},
        2556: {"pm25_index": 110.0, "observed_at": now - pd.Timedelta(hours=5)},
    }
    refresh_flags = []

    def stub_observation(uid, force_refresh=False):
        refresh_flags.append(force_refresh)
        return observations[uid]

    monkeypatch.setattr(api, "_waqi_observation", stub_observation)

    payload = live_stations_forecast(horizon=3, refresh=True)
    assert refresh_flags == [True, True]
    assert payload["source_status"] == "waqi_station_feed"
    assert payload["forecasts"][0]["quality_status"] == "live"
    assert payload["forecasts"][0]["forecast_pm25_index"] == 91.0
    assert payload["forecasts"][0]["aqi_semantics"] == "PM2.5-only index proxy; not CPCB composite AQI"
    assert payload["forecasts"][1]["quality_status"] == "stale"
    assert payload["forecasts"][1]["forecast_pm25_index"] is None


def test_live_history_recorder_deduplicates_source_observations(monkeypatch, tmp_path):
    from delhi_aircast import api

    path = tmp_path / "observations.jsonl"
    monkeypatch.setattr(api, "LIVE_HISTORY_PATH", path)
    observation = {
        "pm25_index": 88.0,
        "observed_at": pd.Timestamp("2026-10-01T04:00:00Z"),
    }
    api._record_live_observation(10124, observation)
    api._record_live_observation(10124, observation)
    api._record_live_observation(2556, observation)
    rows = [__import__("json").loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 2
    assert {row["station_id"] for row in rows} == {"site_107", "site_124"}


def test_open_meteo_live_weather_selects_latest_row_not_after_sensor_time(monkeypatch):
    from delhi_aircast import api

    rows = []
    for hour, temperature in ((0, 10.0), (1, 11.0), (2, 99.0)):
        rows.append({
            "timestamp_utc": pd.Timestamp("2026-10-01T00:00:00Z") + pd.Timedelta(hours=hour),
            "temperature_2m": temperature,
            "relative_humidity_2m": 50.0,
            "precipitation": 0.0,
            "wind_speed_10m": 3.6,
            "wind_direction_10m": 0.0,
            "surface_pressure": 1000.0,
            "cloud_cover": 20.0,
        })
    monkeypatch.setattr(api, "_WEATHER_CACHE", {"site_107": (api.time.monotonic(), rows)})
    result = api._open_meteo_weather("site_107", pd.Timestamp("2026-10-01T01:30:00Z"))
    assert result is not None
    assert result["values"]["wx_temperature_2m"] == 11.0
    assert result["observed_at"] == pd.Timestamp("2026-10-01T01:00:00Z")


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
