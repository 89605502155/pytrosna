"""Conversion between user-facing times and raw time stamps.

A raw time stamp is an integer number of the device's time unit since
1970-01-01 UTC. Times can be given as ``datetime``/``date``,
``pandas.Timestamp``, ``numpy.datetime64``, ISO 8601 text or an integer
(already raw). A time without a UTC offset is a wall-clock time of the
device's time zone: in a repeated hour (clocks moved back) the earlier moment
is taken; a time skipped when the clocks were moved forward is an error.
"""

from __future__ import annotations

import datetime as dt
import numbers
import re
from typing import Any

from .errors import InvalidArgumentError
from .types import TimeUnit

__all__ = ["format_time", "to_datetime", "to_raw", "tzinfo"]

_EPOCH = dt.datetime(1970, 1, 1, tzinfo=dt.UTC)
_OFFSET = re.compile(r"^([+-])(\d{2})(?::?(\d{2}))?$")
_ISO = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})"
    r"(?:[T ](\d{2}):(\d{2})(?::(\d{2})(?:[.,](\d{1,9}))?)?)?"
    r"\s*(Z|z|[+-]\d{2}:?\d{2})?$"
)
ROUNDINGS = ("floor", "ceil", "exact")


def tzinfo(tz: str | None) -> dt.tzinfo | None:
    """The ``tzinfo`` of a device time zone (an IANA name, ``UTC`` or ``+HH:MM``)."""
    if tz is None:
        return None
    m = _OFFSET.match(tz)
    if m:
        sign = -1 if m.group(1) == "-" else 1
        offset = dt.timedelta(hours=int(m.group(2)), minutes=int(m.group(3) or 0))
        return dt.timezone(sign * offset)
    if tz.upper() in ("UTC", "Z"):
        return dt.UTC
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

    try:
        return ZoneInfo(tz)
    except (ValueError, OSError, ZoneInfoNotFoundError) as e:
        msg = f"unknown time zone {tz!r}"
        raise InvalidArgumentError(msg) from e


def _datetime_ns(value: dt.datetime, tz: str | None) -> int:
    """Exact nanoseconds since the epoch; naive values are in ``tz`` (UTC if None)."""
    if value.tzinfo is None:
        zone = tzinfo(tz) or dt.UTC
        naive = value.replace(fold=0)
        value = naive.replace(tzinfo=zone)
        back = value.astimezone(dt.UTC).astimezone(zone).replace(tzinfo=None, fold=0)
        if back != naive:
            msg = f"{naive} does not exist in time zone {tz} (the clocks were moved forward)"
            raise InvalidArgumentError(msg)
    delta = value - _EPOCH
    return (delta.days * 86_400 + delta.seconds) * 1_000_000_000 + delta.microseconds * 1_000


def parse_text(text: str, tz: str | None) -> int:
    """Nanoseconds since the epoch of an ISO 8601 time such as
    ``2026-10-08T10:00:00+03:00`` or ``2026-10-08 10:00:00.5``."""
    m = _ISO.match(text.strip())
    if not m:
        msg = (
            f"cannot read {text!r} as a time (expected e.g. 2026-10-08T10:00:00+03:00 "
            "or 2026-10-08 10:00:00)"
        )
        raise InvalidArgumentError(msg)
    year, month, day, hour, minute, second, fraction, offset = m.groups()
    try:
        base = dt.datetime(
            int(year), int(month), int(day), int(hour or 0), int(minute or 0), int(second or 0)
        )
    except ValueError as e:
        msg = f"cannot read {text!r} as a time: {e}"
        raise InvalidArgumentError(msg) from None
    if offset:
        offset = "+00:00" if offset in ("Z", "z") else offset
        base = base.replace(tzinfo=tzinfo(offset))
    nanos = int((fraction or "").ljust(9, "0"))
    return _datetime_ns(base, tz) + nanos


def to_nanos(value: Any, tz: str | None) -> int:
    """Nanoseconds since the epoch of a time given in any supported form except
    a raw integer."""
    if isinstance(value, str):
        return parse_text(value, tz)
    if hasattr(value, "tz_localize") and hasattr(value, "value"):  # pandas.Timestamp
        if value.tzinfo is None:
            return _datetime_ns(value.to_pydatetime(warn=False), tz) + value.nanosecond
        return int(value.value)
    if type(value).__name__ == "datetime64":  # numpy, naive by definition
        ns = int(value.astype("datetime64[ns]").astype("int64"))
        if tz is None:
            return ns
        local = _EPOCH.replace(tzinfo=None) + dt.timedelta(microseconds=ns // 1_000)
        return _datetime_ns(local, tz) + ns % 1_000
    if isinstance(value, dt.datetime):
        return _datetime_ns(value, tz)
    if isinstance(value, dt.date):
        return _datetime_ns(dt.datetime(value.year, value.month, value.day), tz)
    msg = f"cannot use a {type(value).__name__} as a time"
    raise TypeError(msg)


def to_raw(value: Any, unit: TimeUnit | str, tz: str | None = None, rounding: str = "floor") -> int:
    """Converts a time to a raw time stamp in ``unit``.

    Integers are returned unchanged (they are already raw). ``rounding``
    decides what happens to a time finer than the unit: ``"floor"``,
    ``"ceil"`` or ``"exact"`` (an error).

    >>> to_raw("2026-10-08T10:00:00+03:00", "s")
    1791442800
    >>> to_raw("2026-10-08 10:00:00", "ms", "Europe/Moscow")
    1791442800000
    """
    if isinstance(value, bool):
        msg = "a bool is not a time"
        raise TypeError(msg)
    if isinstance(value, numbers.Integral):
        return int(value)
    if rounding not in ROUNDINGS:
        msg = f"rounding must be 'floor', 'ceil' or 'exact', not {rounding!r}"
        raise InvalidArgumentError(msg)
    ns = to_nanos(value, tz)
    step = TimeUnit.parse(unit).nanos
    if ns % step == 0 or rounding == "floor":
        return ns // step
    if rounding == "ceil":
        return -(-ns // step)
    msg = f"{value!r} is more precise than the device's time unit ({TimeUnit.parse(unit)})"
    raise InvalidArgumentError(msg)


def to_datetime(raw: int, unit: TimeUnit | str, tz: str | None = None) -> dt.datetime:
    """Converts a raw time stamp to a ``datetime`` (microsecond precision):
    aware in the device's time zone, or naive UTC if the device has none."""
    ns = int(raw) * TimeUnit.parse(unit).nanos
    value = _EPOCH + dt.timedelta(microseconds=ns // 1_000)
    zone = tzinfo(tz)
    return value.astimezone(zone) if zone else value.replace(tzinfo=None)


def format_time(raw: int, unit: TimeUnit | str, tz: str | None = None) -> str:
    """ISO 8601 text of a raw time stamp with the full precision of ``unit``.

    >>> format_time(1791442800500, "ms", "+03:00")
    '2026-10-08T10:00:00.500+03:00'
    """
    unit = TimeUnit.parse(unit)
    seconds, fraction = divmod(int(raw), unit.per_second)
    value = _EPOCH + dt.timedelta(seconds=seconds)
    zone = tzinfo(tz)
    local = value.astimezone(zone) if zone else value
    text = local.strftime("%Y-%m-%dT%H:%M:%S")
    digits = 3 * int(unit)
    if digits:
        text += "." + str(fraction).rjust(digits, "0")
    if zone is None or tz in ("UTC", "Z", "utc"):
        return text + "Z"
    offset = local.strftime("%z")
    return f"{text}{offset[:3]}:{offset[3:]}"
