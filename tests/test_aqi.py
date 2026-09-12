from delhi_aircast.aqi import calculate_aqi, calculate_subindex, pm25_proxy


def test_pm25_breakpoint_boundaries():
    assert calculate_subindex("PM2.5", 30) == 50
    assert calculate_subindex("PM2.5", 60) == 100
    assert calculate_subindex("PM2.5", 90) == 200
    assert calculate_subindex("PM2.5", 120) == 300
    assert calculate_subindex("PM2.5", 250) == 400
    assert calculate_subindex("PM2.5", 251) == 401


def test_pm25_example_is_not_the_old_mock_aqi():
    result = pm25_proxy(145)
    assert result.status == "pm25_derived_proxy"
    assert result.value == 319
    assert result.category == "Very Poor"


def test_overall_requires_three_pollutants_and_pm():
    insufficient = calculate_aqi({"pm25": 145, "pm10": 230})
    assert insufficient.value is None
    assert insufficient.status == "insufficient_for_overall"

    sufficient = calculate_aqi({"pm25": 145, "pm10": 230, "no2": 40})
    assert sufficient.status == "ok"
    assert sufficient.value == 319
    assert sufficient.category == "Very Poor"


def test_invalid_values_are_not_silent_zeroes():
    assert calculate_subindex("PM2.5", None) is None
    assert calculate_subindex("PM2.5", -1) is None
    assert calculate_subindex("unknown", 50) is None


def test_common_cpcb_pollutants_have_breakpoints():
    assert calculate_subindex("NO2", 40) == 50
    assert calculate_subindex("SO2", 80) == 100
    assert calculate_subindex("O3", 168) == 200
    assert calculate_subindex("CO", 1) == 50
    assert calculate_subindex("NH3", 200) == 50
