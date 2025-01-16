# src/utils/helpers.py
import os
import numpy as np
import pandas as pd
import datetime
import pytz

# Timezone references
NYC_TZ = pytz.timezone("America/New_York")
UTC_TZ = datetime.timezone.utc

# Reference epoch in UTC
EPOCH_UTC = datetime.datetime(1970, 1, 1, tzinfo=UTC_TZ)


def extract_ticker(filepath: str) -> str:
    """
    ../ticker_something.fileEnding -> ticker
    ../ticker.fileEnding -> ticker

    e.g. data/raw/stock/AAPL_1m.csv -> 'AAPL'
    data/raw/stock/AAPL.csv -> 'AAPL'
    """
    base = os.path.basename(filepath)
    base_minus_ext = os.path.splitext(base)[0]
    return base_minus_ext.split("_")[0]


def num_covers(cover_interval, active_sessions: dict) -> int:
    """
    Given coverage interval (start, end):
        - sum the minutes covered across all active session
        days in 'active_sessions'.
    Returns the total number of 1-minute bars in that interval.
    """
    one_minute_ms = 60000
    s, e = cover_interval
    total_ms = 0
    for day_str, (day_s, day_e) in active_sessions.items():
        start_overlap = max(s, day_s)
        end_overlap = min(e, day_e)
        if end_overlap > start_overlap:
            total_ms += end_overlap - start_overlap
    return total_ms // one_minute_ms


def all_tickers(config: dict) -> list:
    """
    Returns the list of all stock + ETF tickers from the data config.
    """
    stock_list = config["data"].get("stock_tickers", [])
    etf_list = config["data"].get("etf_tickers", [])
    return stock_list + etf_list


def max_lookback_window(config: dict) -> int:
    """
    Reads the 'feature_engineering' section of the config,
    checks each item in 'look_back_features' (which specify
    both the compute_key and its window_key),
    and returns the maximum window for all enabled features.
    """
    fe_cfg = config["feature_engineering"]
    lb_features = config["look_back_features"]

    max_window = 1
    for feat in lb_features:
        compute_key = feat["compute_key"]  # e.g. "compute_realized_variance"
        window_key = feat["window_key"]  # e.g. "rv_window"

        # Is this feature enabled?
        if fe_cfg.get(compute_key, False):
            # If enabled, read its window; default to 1 if not found
            w = fe_cfg.get(window_key, 1)
            max_window = max(max_window, w)

    return max_window - 1


def total_tickers(config):
    return config["data"]["stock_tickers"] + config["data"]["etf_tickers"]


def convert_time(input_val):
    """
    Convert either:
      - An integer (Unix time in milliseconds)
      - A string in the form "YYYY-MM-DD HH:MM:SS" representing UTC

    Returns a dict with:
      {
        'unix_ms': <int>,
        'dt': <datetime in local NYC time>,
        'date': <str in 'YYYY-MM-DD' in NYC local>
      }
    """
    if isinstance(input_val, int):
        # 1) Handle Unix msec -> NYC datetime
        unix_ms = input_val
        # Convert milliseconds to seconds
        unix_sec = unix_ms / 1000.0

        # Create a UTC datetime from the timestamp
        dt_utc = datetime.datetime.utcfromtimestamp(unix_sec).replace(tzinfo=UTC_TZ)

    elif isinstance(input_val, str):
        # 2) Handle UTC datetime string -> Unix msec
        # Parse naive datetime
        dt_naive = datetime.datetime.strptime(input_val, "%Y-%m-%d %H:%M:%S")
        # Mark it as UTC
        dt_utc = dt_naive.replace(tzinfo=UTC_TZ)

        # Compute Unix ms from dt_utc
        delta = dt_utc - EPOCH_UTC
        unix_ms = int(delta.total_seconds() * 1000)

    else:
        raise TypeError(
            "Input must be either an int (Unix ms) or a UTC datetime string."
        )

    # Convert UTC -> NYC
    dt_nyc = dt_utc.astimezone(NYC_TZ)

    # Extract YYYY-MM-DD from NYC datetime
    date_str = dt_nyc.strftime("%Y-%m-%d")

    return {"unix_ms": unix_ms, "dt": dt_nyc, "date": date_str}
