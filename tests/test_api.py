from delhi_aircast.api import AQIRequest, aqi, health


def test_health_exposes_engine_and_local_data_state():
    payload = health()
    assert payload["status"] == "ok"
    assert payload["aqi_engine"] == "available"


def test_aqi_endpoint_keeps_pm25_proxy_explicit():
    payload = aqi(AQIRequest(pm25=145))
    assert payload["value"] == 319
    assert payload["status"] == "pm25_derived_proxy"
