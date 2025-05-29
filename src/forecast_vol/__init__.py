"""
Intraday volatility-forecasting toolkit.

Sub-packages:
-------------
    market      : Market session utilities for NYSE calendar
    preprocess  : Minute-level cleaning and feature engineering
    stochastic  : Wavelet, EGARCH, and RFSV feature fits
    synergy     : Regime labelling + GNN correlation model
    attention   : Transformer model for RV forecasting
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # editable install
    __version__ = "0.0.0+dev"

__all__: list[str] = [
    "__version__"
    ]
