"""Tests for the synergy layer."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest
import torch

from forecast_vol.synergy.build_corr import build_corr
from forecast_vol.synergy.gnn_model import gnn_data, train_once


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def _toy_returns() -> dict[str, pl.DataFrame]:
    """
    Two perfectly anti-correlated five-step return series.

    AAA := 0 .. 1 (inclusive)           (monotone inc.)
    BBB := 1 .. 0 (inclusive)           (monotone dec., rho ~ -1)

    Using equally-spaced floats avoids numerical headaches.
    """
    aaa = np.linspace(0.0, 1.0, 5, dtype=np.float32)
    bbb = np.linspace(1.0, 0.0, 5, dtype=np.float32)
    return {
        "AAA": pl.DataFrame({"ret": aaa}),
        "BBB": pl.DataFrame({"ret": bbb}),
    }


@pytest.fixture
def _toy_stats() -> pl.DataFrame:
    """
    Minimal per-ticker stats frame required by ``gnn_data``.
    Values are arbitrary; the test asserts shapes, not magnitudes.
    """
    return pl.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "vov": [0.1, 0.2],
            "iqv": [0.3, 0.4],
            "trade_intensity": [100.0, 150.0],
        }
    ).sort("ticker")


# --------------------------------------------------------------------------- #
# build_corr
# --------------------------------------------------------------------------- #
def test_build_corr_properties(_toy_returns: dict[str, pl.DataFrame]) -> None:
    """Matrix must be square, finite, and rho(AAA,AAA)=1."""
    symbols, corr = build_corr(_toy_returns, benches=["AAA"], window=5)

    # square & dtype
    n = len(symbols)
    assert corr.shape == (n, n)
    assert corr.dtype == np.float64 or corr.dtype == np.float32

    # finite and symmetric
    assert not np.isnan(corr).any()
    assert np.allclose(corr, corr.T, atol=1e-6)

    idx_aaa = symbols.index("AAA")
    assert math.isclose(corr[idx_aaa, idx_aaa], 1.0, abs_tol=1e-6)



# --------------------------------------------------------------------------- #
# gnn_data
# --------------------------------------------------------------------------- #
def test_gnn_data_shapes(
    _toy_stats: pl.DataFrame,
    _toy_returns: dict[str, pl.DataFrame],
) -> None:
    """gnn_data must produce correct feature/edge shapes."""
    symbols, corr = build_corr(_toy_returns, benches=["AAA"], window=5)
    data = gnn_data(symbols, corr, _toy_stats, thr=0.0)  # keep all edges

    # node feature matrix: [corr | 3 stats]
    n = len(symbols)
    expected_feat_dim = n + 3
    assert data.x.shape == (n, expected_feat_dim)

    # edge_index has shape (2, E) contiguous LongTensor
    assert data.edge_index.shape[0] == 2
    assert data.edge_index.dtype == torch.long

    # training target present
    assert hasattr(data, "corr")
    assert data.corr.shape == (n, n)


# --------------------------------------------------------------------------- #
# train_once
# --------------------------------------------------------------------------- #
def test_train_once_runs(
    _toy_stats: pl.DataFrame,
    _toy_returns: dict[str, pl.DataFrame],
) -> None:
    """
    A 3-epoch training run should finish quickly, return finite embeddings,
    and a non-negative loss.
    """
    symbols, corr = build_corr(_toy_returns, benches=["AAA"], window=5)
    data = gnn_data(symbols, corr, _toy_stats, thr=0.0)

    emb, loss = train_once(
        data=data,
        hidden_dim=4,
        layers=2,
        lr=1e-2,
        epochs=3,           # keep the test fast
        device="cpu",
    )

    # shapes and dtypes
    assert emb.shape == (len(symbols), 4)
    assert emb.dtype == np.float32

    # finite & reasonable loss
    assert np.isfinite(emb).all()
    assert np.isfinite(loss) and loss >= 0
