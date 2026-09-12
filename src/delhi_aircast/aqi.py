"""CPCB National AQI sub-index helpers.

The official CPCB method uses short-term averaging windows, pollutant
sub-indices, and the worst available sub-index. This module intentionally
returns an explicit insufficiency state instead of inventing a full AQI when
only PM2.5 is present.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class Breakpoint:
    concentration_low: float
    concentration_high: float
    index_low: float
    index_high: float


@dataclass(frozen=True)
class AQIResult:
    value: int | None
    category: str | None
    status: str
    subindices: dict[str, int]


# The CPCB final report publishes these concentration/category breakpoints.
# Concentrations are rounded to the integer precision used by the table before
# interpolation, which keeps table boundary values deterministic.
BREAKPOINTS: dict[str, tuple[Breakpoint, ...]] = {
    "pm25": (
        Breakpoint(0, 30, 0, 50),
        Breakpoint(31, 60, 51, 100),
        Breakpoint(61, 90, 101, 200),
        Breakpoint(91, 120, 201, 300),
        Breakpoint(121, 250, 301, 400),
        Breakpoint(251, 500, 401, 500),
    ),
    "pm10": (
        Breakpoint(0, 50, 0, 50),
        Breakpoint(51, 100, 51, 100),
        Breakpoint(101, 250, 101, 200),
        Breakpoint(251, 350, 201, 300),
        Breakpoint(351, 430, 301, 400),
        Breakpoint(431, 500, 401, 500),
    ),
    "no2": (
        Breakpoint(0, 40, 0, 50),
        Breakpoint(41, 80, 51, 100),
        Breakpoint(81, 180, 101, 200),
        Breakpoint(181, 280, 201, 300),
        Breakpoint(281, 400, 301, 400),
        Breakpoint(401, 1000, 401, 500),
    ),
    "so2": (
        Breakpoint(0, 40, 0, 50),
        Breakpoint(41, 80, 51, 100),
        Breakpoint(81, 380, 101, 200),
        Breakpoint(381, 800, 201, 300),
        Breakpoint(801, 1600, 301, 400),
        Breakpoint(1601, 5000, 401, 500),
    ),
    "o3": (
        Breakpoint(0, 50, 0, 50),
        Breakpoint(51, 100, 51, 100),
        Breakpoint(101, 168, 101, 200),
        Breakpoint(169, 208, 201, 300),
        Breakpoint(209, 748, 301, 400),
        Breakpoint(749, 1000, 401, 500),
    ),
    "co": (
        Breakpoint(0, 1, 0, 50),
        Breakpoint(1.01, 2, 51, 100),
        Breakpoint(2.01, 10, 101, 200),
        Breakpoint(10.01, 17, 201, 300),
        Breakpoint(17.01, 34, 301, 400),
        Breakpoint(34.01, 50, 401, 500),
    ),
    "nh3": (
        Breakpoint(0, 200, 0, 50),
        Breakpoint(201, 400, 51, 100),
        Breakpoint(401, 800, 101, 200),
        Breakpoint(801, 1200, 201, 300),
        Breakpoint(1201, 1800, 301, 400),
        Breakpoint(1801, 5000, 401, 500),
    ),
}

CATEGORIES = (
    (0, 50, "Good"),
    (51, 100, "Satisfactory"),
    (101, 200, "Moderate"),
    (201, 300, "Poor"),
    (301, 400, "Very Poor"),
    (401, 500, "Severe"),
)


def _category(value: int | None) -> str | None:
    if value is None:
        return None
    for low, high, name in CATEGORIES:
        if low <= value <= high:
            return name
    return "Severe" if value > 500 else None


def _canonical_pollutant(pollutant: str) -> str:
    key = pollutant.lower().replace(".", "").replace(" ", "")
    aliases = {
        "pm25": "pm25",
        "pm10": "pm10",
        "no2": "no2",
        "so2": "so2",
        "o3": "o3",
        "co": "co",
        "nh3": "nh3",
    }
    return aliases.get(key, key)


def calculate_subindex(pollutant: str, concentration: float | int | None) -> int | None:
    """Return one CPCB pollutant sub-index, or None for invalid data."""

    if concentration is None:
        return None
    try:
        numeric = float(concentration)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or numeric < 0:
        return None

    key = _canonical_pollutant(pollutant)
    if key not in BREAKPOINTS:
        return None

    rounded = math.floor(numeric + 0.5)
    for breakpoint in BREAKPOINTS[key]:
        if breakpoint.concentration_low <= rounded <= breakpoint.concentration_high:
            span = breakpoint.concentration_high - breakpoint.concentration_low
            value = breakpoint.index_low + ((rounded - breakpoint.concentration_low) / span) * (
                breakpoint.index_high - breakpoint.index_low
            )
            return max(0, min(500, round(value)))
    return 500 if rounded > 0 else 0


def calculate_aqi(pollutants: Mapping[str, float | int | None]) -> AQIResult:
    """Calculate available sub-indices and an official AQI when sufficient.

    CPCB requires at least three pollutant sub-indices, including PM2.5 or
    PM10, for an overall AQI. With fewer inputs the returned status is
    ``insufficient_for_overall`` while valid sub-indices remain available.
    """

    subindices: dict[str, int] = {}
    for pollutant, concentration in pollutants.items():
        subindex = calculate_subindex(pollutant, concentration)
        if subindex is not None:
            subindices[_canonical_pollutant(pollutant)] = subindex

    if len(subindices) < 3 or not ({"pm25", "pm10"} & set(subindices)):
        return AQIResult(None, None, "insufficient_for_overall", subindices)
    value = max(subindices.values())
    return AQIResult(value, _category(value), "ok", subindices)


def pm25_proxy(concentration: float | int | None) -> AQIResult:
    """Return an explicitly labelled PM2.5-only proxy result."""

    value = calculate_subindex("pm25", concentration)
    return AQIResult(
        value,
        _category(value),
        "pm25_derived_proxy" if value is not None else "invalid",
        {"pm25": value} if value is not None else {},
    )
