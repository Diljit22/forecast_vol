"""
Minute-level cleaning and feature engineering.

Trim raw CSVs to the official NYSE minute grid, gap-fill restricted
bars, recompute VWAP, and build a feature-rich parquet dataset.

Public API
----------
run_restrict      : Restrict raw data to the trading calendar
run_resample      : Fill gaps and recompute VWAP
run_build_dataset : Merge parquet files and create engineered features
"""

from __future__ import annotations

from .build_dataset import main as run_build_dataset
from .resample import main as run_resample
from .restrict import main as run_restrict

__all__: list[str] = [
    "run_restrict",
    "run_resample",
    "run_build_dataset",
    ]
