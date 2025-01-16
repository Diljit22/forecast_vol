import numpy as np
import pandas as pd
from typing import Tuple

from src.data_ingestion.market_sessions.find_active_sessions import (
    is_active_trading_day,
)
from script_generated_files.market_hours import ACTIVE_SESSIONS

ONE_MINUTE_MS = 60_000


def compute_active_nhood(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized approach:
    For each row's date, compute (date - day_delta) and (date + day_delta)
    check if those are active days in the holidays dict.
    """
    day_delta = 1
    df["yday"] = df["datetime"] - pd.to_timedelta(day_delta, unit="D")
    df["tom"] = df["datetime"] + pd.to_timedelta(day_delta, unit="D")

    df["active_yday"] = df["yday"].apply(
        lambda d: is_active_trading_day(d, ACTIVE_SESSIONS)
    )
    df["active_tom"] = df["tom"].apply(
        lambda d: is_active_trading_day(d, ACTIVE_SESSIONS)
    )

    df.drop(columns=["yday", "tom"], inplace=True)
    return df


def compute_distance_open_close(df: pd.DataFrame, active_sessions=ACTIVE_SESSIONS):
    """
    For each row: time_since_open = t - open_msec
    time_until_close = close_msec - t - ONE_MINUTE_MS
    """
    df["datetime"] = pd.to_datetime(df["datetime"])
    dt_str = df["datetime"].dt.strftime("%Y-%m-%d")

    open_times = dt_str.map(
        lambda d: active_sessions[d][0] if d in active_sessions else np.nan
    )
    close_times = dt_str.map(
        lambda d: active_sessions[d][1] if d in active_sessions else np.nan
    )

    df["time_since_open"] = df["t"] - open_times
    df["time_until_close"] = (close_times - df["t"]) - ONE_MINUTE_MS
    return df


def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Close-to-close returns."""
    df["returns"] = df["close"].pct_change().fillna(0)
    return df


def compute_log_return(df: pd.DataFrame, price_col="close") -> pd.DataFrame:
    """Log returns."""
    df["log_return"] = np.log(df[price_col]).diff().fillna(0.0)
    return df


def compute_parkinson_vol(df, window=30, eps=1.0e-8):
    """
    Parkinson formula: sqrt( (1/(4log(2))) * mean( ln((high+eps)/(low+eps))^2 ) )
    on a rolling window.
    """
    factor = 1.0 / (4.0 * np.log(2))
    df["log_ratio"] = np.log((df["high"] + eps) / (df["low"] + eps))
    df["park_vol"] = (
        df["log_ratio"]
        .rolling(window=window)
        .apply(lambda x: np.sqrt(factor * (x**2).mean()), raw=True)
    )
    df.drop(columns=["log_ratio"], inplace=True)
    return df


def compute_realized_variance(df: pd.DataFrame, window=5) -> pd.DataFrame:
    """Realized variance: sum of squared log returns over rolling window."""
    df["rv"] = (
        df["log_return"].rolling(window=window).apply(lambda x: np.sum(x**2), raw=True)
    )
    df["rv"] = df["rv"].bfill()
    return df


def compute_bipower_variation(
    df: pd.DataFrame, price_col="close", window=5
) -> pd.DataFrame:
    """Bipower Variation: mean(|r_{t-1}| * |r_t|) using % changes."""
    df["pct_return"] = df[price_col].pct_change()
    df["abs_return"] = df["pct_return"].abs()

    df["bpv"] = (
        df["abs_return"]
        .rolling(window=window)
        .apply(
            lambda arr: np.mean(arr[:-1] * arr[1:]) if len(arr) == window else np.nan,
            raw=True,
        )
    )
    df["bpv"] = df["bpv"].bfill()
    return df


def compute_garman_klass(df: pd.DataFrame, window=5) -> pd.DataFrame:
    """Garman–Klass volatility measure."""
    ln_hl = np.log(df["high"] / df["low"]) ** 2
    ln_co = np.log(df["close"] / df["open"]) ** 2
    gk = 0.5 * ln_hl - (2 * np.log(2) - 1) * ln_co
    df["gk"] = gk
    df["gk_vol"] = df["gk"].rolling(window=window).mean()
    df["gk_vol"] = df["gk_vol"].bfill()
    return df


def compute_integrated_quartic_var(df: pd.DataFrame, window=30) -> pd.DataFrame:
    """Sum of log_return^4 over a rolling window."""
    df["iqv"] = (
        df["log_return"].rolling(window=window).apply(lambda x: np.sum(x**4), raw=True)
    )
    df["iqv"] = df["iqv"].bfill()
    return df


def compute_trade_intensity(df: pd.DataFrame) -> pd.DataFrame:
    """Naive trade intensity or order imbalance measure."""
    if "volume" not in df.columns:
        df["order_imbalance"] = np.nan
    else:
        df["order_imbalance"] = df["volume"].diff().fillna(0)
    return df


def compute_vov(df: pd.DataFrame, vol_col="rv", window=5) -> pd.DataFrame:
    """Vol-of-Vol: rolling std of a chosen volatility measure."""
    df["vov"] = df[vol_col].rolling(window=window).std().bfill()
    return df
