# tests/test_helpers.py

import pytest
from src.utils.helpers import (
    extract_ticker,
    num_covers,
    all_tickers,
    max_lookback_window,
    convert_time,
)


def test_extract_ticker():
    filepath = "/some/path/AAPL_1m.csv"
    assert extract_ticker(filepath) == "AAPL"

    filepath_no_underscore = "/other/path/GOOG.csv"
    assert extract_ticker(filepath_no_underscore) == "GOOG"


def test_num_covers():
    # Define a coverage interval (in milliseconds)
    one_minute_msec = 60_000
    cover_interval = (
        1672842600000 + 2 * one_minute_msec,
        1673298000000 - 5 * one_minute_msec,
    )
    mock_active_sessions = {
        "2023-01-03": (1672756200000, 1672779600000),
        "2023-01-04": (1672842600000, 1672866000000),
        "2023-01-05": (1672929000000, 1672952400000),
        "2023-01-06": (1673015400000, 1673038800000),
        "2023-01-09": (1673274600000, 1673298000000),
        "2023-01-10": (1673361000000, 1673384400000),
    }

    over = (
        (1672866000000 - 1672842600000)
        + (1672952400000 - 1672929000000)
        + (1673038800000 - 1673015400000)
        + (1673298000000 - 1673274600000)
    )
    under = 7 * one_minute_msec
    res = (over - under) // one_minute_msec

    assert num_covers(cover_interval, mock_active_sessions) == res


def test_all_tickers():
    config = {
        "data": {"stock_tickers": ["AAPL", "MSFT"], "etf_tickers": ["SPY", "QQQ"]}
    }
    result = all_tickers(config)
    assert result == ["AAPL", "MSFT", "SPY", "QQQ"]


def test_max_lookback_window():
    config = {
        "feature_engineering": {
            "compute_realized_variance": True,
            "rv_window": 10,
            "compute_something_else": False,
        },
        "look_back_features": [
            {"compute_key": "compute_realized_variance", "window_key": "rv_window"},
            {
                "compute_key": "compute_something_else",
                "window_key": "some_other_window",
            },
        ],
    }
    # max_window would become max(1, 10) = 10. Then we return 10 - 1 = 9
    assert max_lookback_window(config) == 9


def test_convert_time_from_int():
    """
    Test converting an integer Unix timestamp (in ms).
    """
    input_val = 1672929000000
    result = convert_time(input_val)
    assert result["unix_ms"] == input_val
    assert result["dt"].tzinfo is not None
    assert str(result["dt"].tzinfo) == "America/New_York"
    assert result["date"] == "2023-01-05"


def test_convert_time_from_str():
    """
    Test converting a UTC datetime string => dictionary with local NYC time.
    """
    input_val = "2023-01-05 14:30:00"
    result = convert_time(input_val)
    assert result["unix_ms"] == 1672929000000
    assert result["date"] == "2023-01-05"
    # Check local datetime matches:
    assert result["dt"].strftime("%Y-%m-%d %H:%M:%S") == "2023-01-05 09:30:00"
