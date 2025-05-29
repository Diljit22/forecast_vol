"""Tests for the attention layer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from forecast_vol.attention.data import MinuteDataset, chrono_split
from forecast_vol.attention.model import Predictor


# --------------------------------------------------------------------------- #
# Predictor forward pass
# --------------------------------------------------------------------------- #
def test_predictor_forward() -> None:
    """Predictor must accept [B,F] input and return [B,1] output without NaNs."""
    B, F = 8, 16 # noqa N806
    x = torch.randn(B, F)
    net = Predictor(
        inp_dim=F,
        d_model=32,
        n_heads=2,
        n_layers=1,
        ff_dim=64,
        dropout=0.0,
    )
    y = net(x)

    assert y.shape == (B, 1)
    assert torch.isfinite(y).all()


# --------------------------------------------------------------------------- #
# chrono_split ordering & sizes
# --------------------------------------------------------------------------- #
def test_chrono_split_order() -> None:
    """train < val < test chronologically and sizes add up."""
    n = 100
    df = pd.DataFrame({"t": np.arange(n), "rv_fwd": np.random.rand(n)})
    tr, va, te = chrono_split(df)

    assert len(tr) + len(va) + len(te) == n
    assert tr["t"].iloc[-1] < va["t"].iloc[0] < te["t"].iloc[0]


# --------------------------------------------------------------------------- #
# MinuteDataset basic indexing
# --------------------------------------------------------------------------- #
def test_minute_dataset_getitem() -> None:
    """Dataset returns float32 arrays with expected shapes."""
    feats = ["f0", "f1"]
    df = pd.DataFrame(
        {
            "f0": np.random.rand(10),
            "f1": np.random.rand(10),
            "rv_fwd": np.random.rand(10),
        }
    )
    ds = MinuteDataset(df, feats, "rv_fwd")

    assert len(ds) == 10
    x, y = ds[0]
    assert x.dtype == np.float32 and y.dtype == np.float32
    assert x.shape == (len(feats),) and y.shape == (1,)
