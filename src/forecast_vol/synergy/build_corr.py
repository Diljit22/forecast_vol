"""
Build a cross-ticker correlation matrix from return residuals.

Details
--------
Computes OLS residuals against benchmark tickers and constructs
a NaN/inf-safe correlation matrix.

Public API
----------
- build_corr : return list of symbols and correlation matrix.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import polars as pl

# --------------------------------------------------------------------------- #
# Configuration & global constants
# --------------------------------------------------------------------------- #

# Logging 
log = logging.getLogger(__name__.split(".")[-1])
if not log.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
    log.addHandler(_handler)
log.setLevel(logging.INFO)

# --------------------------------------------------------------------------- #
# Core function
# --------------------------------------------------------------------------- #

def build_corr(
    dfs: dict[str, pl.DataFrame],
    benches: Sequence[str],
    window: int,
) -> tuple[list[str], np.ndarray]:
    """Build symbols list NaN/inf-safe correlation matrix.

    Parameters
    ----------
    dfs
        Mapping ticker -> DataFrame containing at `ret` column of minute returns.
    benches
        Benchmark tickers used for OLS residual design matrix.
    window
        Number of most recent rows to consider.

    Returns
    -------
    symbols
        Sorted list of tickers.
    corr
        (n, n) correlation matrix with NaN/inf replaced by 0.
    """
    symbols = sorted(dfs)

    # Design matrix of benchmark returns
    X = np.column_stack( #noqa N806
        [dfs[b]["ret"].tail(window).to_numpy() for b in benches]
    ).astype(np.float32)

    def _ols_residual(y: np.ndarray) -> np.ndarray:
        X_ = np.column_stack([np.ones_like(y), X]) #noqa N806
        beta, *_ = np.linalg.lstsq(X_, y, rcond=None)
        return y - X_ @ beta

    series: list[np.ndarray] = []
    for sym in symbols:
        y = dfs[sym]["ret"].tail(window).to_numpy().astype(np.float32)
        series.append(y if sym in benches else _ols_residual(y))
        log.debug("%s residual vector built (len=%d)", sym, y.size)

    corr = np.corrcoef(np.vstack(series))
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    
    return symbols, corr
