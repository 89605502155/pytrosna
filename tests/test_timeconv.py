"""Conversion between user-facing times and raw time stamps."""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from pytrosna import format_time, to_datetime, to_raw
from pytrosna.errors import InvalidArgumentError
from pytrosna.timeconv import parse_text, tzinfo

MOSCOW_10H = 1_791_442_800  # 2026-10-08 10:00:00 Europe/Moscow, in seconds


@pytest.mark.parametrize(
    "value",
    [
        "2026-10-08T10:00:00+03:00",
        "2026-10-08T07:00:00Z",
        "2026-10-08 07:00:00z",
        "2026-10-08T10:00:00+0300",
        dt.datetime(2026, 10, 8, 7, tzinfo=dt.UTC),
        dt.datetime(2026, 10, 8, 10, tzinfo=ZoneInfo("Europe/Moscow")),
        pd.Timestamp("2026-10-08 07:00", tz="UTC"),
        MOSCOW_10H,
    ],
)
def test_instants(value: object) -> None:
    assert to_raw(value, "s") == MOSCOW_10H


@pytest.mark.parametrize(
    "value",
    [
        "2026-10-08 10:00",
        "2026-10-08T10:00:00",
        dt.datetime(2026, 10, 8, 10),
        pd.Timestamp("2026-10-08 10:00"),
        np.datetime64("2026-10-08T10:00:00"),
    ],
)
def test_wall_clock_times_use_the_device_zone(value: object) -> None:
    assert to_raw(value, "s", "Europe/Moscow") == MOSCOW_10H
    assert to_raw(value, "s", "+03:00") == MOSCOW_10H
    assert to_raw(value, "s") == MOSCOW_10H + 3 * 3600  # UTC when the device has none


def test_dates() -> None:
    assert to_raw(dt.date(1970, 1, 2), "ms") == 86_400_000
    assert to_raw("1970-01-02", "s") == 86_400


def test_fractions_and_rounding() -> None:
    text = "1970-01-01T00:00:01.123456789Z"
    assert to_raw(text, "ns") == 1_123_456_789
    assert to_raw(text, "ms") == 1123
    assert to_raw(text, "ms", rounding="ceil") == 1124
    assert to_raw("1970-01-01T00:00:01.5Z", "s", rounding="ceil") == 2
    assert to_raw("1970-01-01T00:00:01Z", "s", rounding="exact") == 1
    with pytest.raises(InvalidArgumentError, match="more precise"):
        to_raw(text, "ms", rounding="exact")
    with pytest.raises(InvalidArgumentError, match="rounding"):
        to_raw(text, "ms", rounding="nearest")
    assert to_raw(pd.Timestamp("1970-01-01 00:00:00.000000005"), "ns") == 5
    assert to_raw(np.datetime64(7, "ns"), "ns") == 7
    assert to_raw(np.datetime64(7, "ns"), "ns", "UTC") == 7


def test_daylight_saving_transitions() -> None:
    berlin = "Europe/Berlin"
    # 2026-03-29 02:30 does not exist in Berlin
    with pytest.raises(InvalidArgumentError, match="does not exist"):
        to_raw("2026-03-29 02:30", "s", berlin)
    # 2026-10-25 02:30 happens twice; the earlier moment (CEST, +02:00) is taken
    assert to_raw("2026-10-25 02:30", "s", berlin) == to_raw("2026-10-25T02:30:00+02:00", "s")
    later = dt.datetime(2026, 10, 25, 2, 30, fold=1)
    assert to_raw(later, "s", berlin) == to_raw("2026-10-25T02:30:00+02:00", "s")


def test_invalid_times() -> None:
    with pytest.raises(InvalidArgumentError, match="cannot read"):
        to_raw("yesterday", "s")
    with pytest.raises(InvalidArgumentError, match="cannot read"):
        to_raw("2026-13-01", "s")
    with pytest.raises(TypeError, match="bool"):
        to_raw(True, "s")
    with pytest.raises(TypeError, match="cannot use"):
        to_raw(1.5, "s")
    with pytest.raises(InvalidArgumentError, match="time zone"):
        tzinfo("Mars/Olympus")


def test_to_datetime_and_format_time() -> None:
    assert to_datetime(MOSCOW_10H, "s", "Europe/Moscow") == dt.datetime(
        2026, 10, 8, 10, tzinfo=ZoneInfo("Europe/Moscow")
    )
    assert to_datetime(1500, "ms") == dt.datetime(1970, 1, 1, 0, 0, 1, 500_000)
    assert format_time(MOSCOW_10H * 1000 + 500, "ms", "+03:00") == "2026-10-08T10:00:00.500+03:00"
    assert format_time(1, "ns", "UTC") == "1970-01-01T00:00:00.000000001Z"
    assert format_time(-1, "s") == "1969-12-31T23:59:59Z"
    assert format_time(0, "us", "Asia/Kolkata") == "1970-01-01T05:30:00.000000+05:30"


def test_tzinfo() -> None:
    assert tzinfo(None) is None
    assert tzinfo("Z") is dt.UTC
    assert tzinfo("-05:30").utcoffset(None) == -dt.timedelta(hours=5, minutes=30)  # type: ignore[union-attr]
    assert parse_text(" 1970-01-01 ", None) == 0
