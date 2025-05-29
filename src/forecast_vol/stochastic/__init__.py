"""
Stochastic-feature (Wavelet, EGARCH, and RFSV feature fits) pipeline.

Compute per-ticker wavelet fractal metrics, EGARCH parameters,
and rough-volatility statistics.

Public API
----------
run_fit : Fit all stochastic models for every ticker
"""

from __future__ import annotations

from .fit_all import main as run_fit

__all__: list[str] = [
    "run_fit"
    ]
