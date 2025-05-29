"""Tests for building active market sessions."""

from __future__ import annotations

import datetime as _dt
from zoneinfo import ZoneInfo as ZI  #noqa N817

from forecast_vol.market.build_sessions import (
    _to_epoch_ms,
    build_session_calendar,
)

TZ_NY = ZI("America/New_York")
TZ_UTC = ZI("UTC")


def _expected_ms(day: _dt.date, h: int, m: int) -> int:
    ny = _dt.datetime.combine(day, _dt.time(hour=h, minute=m), tzinfo=TZ_NY)
    return _to_epoch_ms(ny.astimezone(TZ_UTC))


def test_full_trading_day() -> None:
    day = _dt.date(2024, 6, 3)  # Monday
    cal = build_session_calendar(day, day, holidays=set(), half_days=set())

    open_ms, close_ms = cal["2024-06-03"]
    assert open_ms == _expected_ms(day, 9, 30)
    assert close_ms == _expected_ms(day, 16, 0)


def test_half_day() -> None:
    day = _dt.date(2024, 11, 29)  # post-Thanksgiving half-day
    cal = build_session_calendar(day, day, holidays=set(), half_days={day})

    open_ms, close_ms = cal["2024-11-29"]
    assert open_ms == _expected_ms(day, 9, 30)
    assert close_ms == _expected_ms(day, 13, 0)


def test_holiday_skipped() -> None:
    day = _dt.date(2024, 7, 4)  # Independence Day
    cal = build_session_calendar(day, day, holidays={day}, half_days=set())
    assert cal == {}


def test_weekend_excluded() -> None:
    start = _dt.date(2024, 6, 1)  # Saturday
    end = _dt.date(2024, 6, 2)    # Sunday
    cal = build_session_calendar(start, end, holidays=set(), half_days=set())
    assert cal == {}
