"""
Vectorised Polars helpers used to engineer minute-level features.

Details
--------
Each `add_*` function appends one numeric column to a Polars DataFrame.
`PIPE` is the ordered list consumed by `build_dataset.apply_pipe`.

Configuration (configs/preprocess.yaml)
---------------------------------------
paths.calendar_json   : data/processed/nyse_2023-2024.json

Public API
----------
- add_active_neighbourhood : prev/next trading-day flags
- add_distance_open_close  : millis since open / until close
- add_simple_returns       : close-to-close percent return
- add_log_return           : log return of close price
- add_parkinson            : Parkinson high-low volatility
- add_realised_var         : rolling realised variance
- add_bipower              : bipower variation estimator
- add_garman               : Garman-Klass volatility
- add_iqv                  : integrated quartic variation
- add_trade_intensity      : first-difference volume proxy
- add_vov                  : volatility-of-volatility
- PIPE                     : tuple with above helpers in execution order
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import polars as pl
import yaml

# --------------------------------------------------------------------------- #
# Configuration & global constants
# --------------------------------------------------------------------------- #

_CFG_PATH = Path("configs/preprocess.yaml")
_CFG = yaml.safe_load(_CFG_PATH.read_text())
_PATHS = _CFG["paths"]
ONE_MIN_MS: int = 60_000

# NYSE calendar
_CAL_JS = Path(_PATHS["calendar_json"])
_raw_cal = json.loads(_CAL_JS.read_text())
CAL: dict[str, tuple[int, int]] = {
    d: (span[0], span[1]) for d, span in _raw_cal.items()
    }

# Pre-compute the trading-day Series
ACTIVE_DAYS = pl.Series("active_days", list(CAL.keys()), dtype=pl.Utf8)

# Logging 
log = logging.getLogger(__name__.split(".")[-1])
if not log.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
    log.addHandler(_handler)
log.setLevel(logging.INFO)

# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #

def add_active_neighbourhood(df: pl.DataFrame) -> pl.DataFrame:
    """Flag whether the previous and next calendar days are trading days."""
    return df.with_columns(
        [
            (
                pl.col("dt")
                .dt.offset_by("-1d")
                .dt.strftime("%Y-%m-%d")
                .is_in(ACTIVE_DAYS)
            ).alias("active_prev"),
            (
                pl.col("dt")
                .dt.offset_by("1d")
                .dt.strftime("%Y-%m-%d")
                .is_in(ACTIVE_DAYS)
            ).alias("active_next"),
        ]
    )


def add_distance_open_close(df: pl.DataFrame) -> pl.DataFrame:
    """Add distance in milliseconds to the day open and close."""
    cal_df = pl.DataFrame(
        {
            "day": list(CAL.keys()),
            "open_ms": [v[0] for v in CAL.values()],
            "close_ms": [v[1] for v in CAL.values()],
        }
    )

    df = df.join(cal_df, on="day", how="left")

    return (
        df.with_columns(
            [
                (pl.col("t") - pl.col("open_ms")).alias("since_open_ms"),
                (pl.col("close_ms") - pl.col("t") - ONE_MIN_MS).alias(
                    "until_close_ms"
                ),
            ]
        ).drop(["open_ms", "close_ms"])
    )


def add_simple_returns(df: pl.DataFrame) -> pl.DataFrame:
    """Add simple percent change of the close price."""
    return df.with_columns(
        pl.col("close").pct_change().fill_null(0.0).alias("ret")
        )


def add_log_return(df: pl.DataFrame) -> pl.DataFrame:
    """Add log return of the close price."""
    return df.with_columns(
        pl.col("close").log().diff().fill_null(0.0).alias("log_ret")
        )


def add_parkinson(df: pl.DataFrame, window: int = 30) -> pl.DataFrame:
    """Add Parkinson volatility estimator."""
    ln_ratio = (pl.col("high") / pl.col("low").clip(1e-8)).log()
    factor = 1.0 / (4.0 * np.log(2.0))
    return df.with_columns(
        ((ln_ratio.pow(2).rolling_mean(window) * factor).sqrt()).alias("park_vol")
    )


def add_realised_var(df: pl.DataFrame, window: int = 5) -> pl.DataFrame:
    """Add realised variance."""
    return df.with_columns(
        pl.col("log_ret").pow(2).rolling_sum(window).alias("rv")
    )


def add_bipower(df: pl.DataFrame, window: int = 5) -> pl.DataFrame:
    """Add bipower variation estimator."""
    abs_ret = pl.col("close").pct_change().abs()
    prod = abs_ret * abs_ret.shift(1)
    bv = prod.rolling_mean(window - 1)
    return df.with_columns(bv.alias("bpv"))


def add_garman(df: pl.DataFrame, window: int = 5) -> pl.DataFrame:
    """Add Garman-Klass volatility estimator."""
    ln_hl = (pl.col("high") / pl.col("low")).log().pow(2)
    ln_oc = (pl.col("close") / pl.col("open")).log().pow(2)
    gk = 0.5 * ln_hl - (2 * np.log(2) - 1) * ln_oc
    return df.with_columns(gk.rolling_mean(window).alias("gk_vol"))


def add_iqv(df: pl.DataFrame, window: int = 30) -> pl.DataFrame:
    """Add integrated quartic variation."""
    return df.with_columns(
        pl.col("log_ret").pow(4).rolling_sum(window).alias("iqv")
    )


def add_trade_intensity(df: pl.DataFrame) -> pl.DataFrame:
    """Add trade intensity feature based on volume differences."""
    return df.with_columns(
        pl.col("volume").diff().fill_null(0).alias("trade_intensity")
        )


def add_vov(df: pl.DataFrame, vol_col: str = "rv", window: int = 5) -> pl.DataFrame:
    """Add volatility of volatility."""
    return df.with_columns(pl.col(vol_col).rolling_std(window).alias("vov"))

# --------------------------------------------------------------------------- #
# Public pipeline
# --------------------------------------------------------------------------- #

PIPE: list[Callable[[pl.DataFrame], pl.DataFrame]] = [
    add_active_neighbourhood,
    add_distance_open_close,
    add_simple_returns,
    add_log_return,
    add_parkinson,
    add_realised_var,
    add_bipower,
    add_garman,
    add_iqv,
    add_trade_intensity,
    add_vov,
]
