"""Tests for the minute-feature engineering helpers."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

import forecast_vol.preprocess.basic_features as bf
from forecast_vol.preprocess.build_dataset import _apply_pipe


# --------------------------------------------------------------------------- #
# Minimal NYSE calendar fixture
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _patch_calendar(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Provide a toy (two-minute) trading session so the helpers don't try to
    load the real JSON calendar.
    """
    day = "2024-06-03"
    open_ms = 0
    close_ms = 2 * 60_000  # two one-minute bars

    monkeypatch.setattr(bf, "CAL", {day: (open_ms, close_ms)}, raising=False)
    monkeypatch.setattr(
        bf,
        "ACTIVE_DAYS",
        pl.Series("active_days", [day], dtype=pl.Utf8),
        raising=False,
    )


# --------------------------------------------------------------------------- #
# Dummy two-row minute frame
# --------------------------------------------------------------------------- #
def _dummy_minutes() -> pl.DataFrame:
    """Return a minimal DataFrame with the core OHLCV columns."""
    return pl.DataFrame(
        {
            "t": [0, 60_000],
            "dt": [
                dt.datetime(2024, 6, 3, 0, 0, tzinfo=dt.UTC),
                dt.datetime(2024, 6, 3, 0, 1, tzinfo=dt.UTC),
            ],
            "day": ["2024-06-03", "2024-06-03"],
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.5, 100.5],
            "close": [100.5, 101.5],
            "volume": [1_000.0, 1_200.0],
        }
    )


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_apply_pipe_roundtrip() -> None:
    """_apply_pipe must preserve row-count and add all expected columns."""
    df = _dummy_minutes()
    out = _apply_pipe(df)

    assert out.height == df.height

    expected_cols = {
        "active_prev",
        "active_next",
        "since_open_ms",
        "until_close_ms",
        "ret",
        "log_ret",
        "park_vol",
        "rv",
        "bpv",
        "gk_vol",
        "iqv",
        "trade_intensity",
        "vov",
    }
    assert expected_cols.issubset(out.columns)


def test_feature_dependency_order() -> None:
    """Ensure helper ordering satisfies downstream dependencies."""
    names = [fn.__name__ for fn in bf.PIPE]

    # log_ret required before rv
    assert names.index("add_log_return") < names.index("add_realised_var")

    # rv must precede vov
    assert names.index("add_realised_var") < names.index("add_vov")
