"""Tests for the stochastic layer (fractal, EGARCH, RFSV)."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from forecast_vol.stochastic.egarch import fit_student_t_egarch

# Targets
from forecast_vol.stochastic.fractal import fractal_dim_wavelet
from forecast_vol.stochastic.rfsv import fit_rfsf

# Silence convergence chatter from the arch package
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# --------------------------------------------------------------------------- #
# Shared synthetic data fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def _rv() -> np.ndarray:
    """Positive realised-variance sequence (len=256)."""
    rng = np.random.default_rng(0)
    return np.exp(rng.standard_normal(256) * 0.15)


@pytest.fixture(scope="module")
def _returns() -> np.ndarray:
    """Mean-zero return series for GARCH (len=512)."""
    rng = np.random.default_rng(1)
    return rng.standard_t(df=5, size=512) * 0.01


# --------------------------------------------------------------------------- #
# fractal_dim_wavelet
# --------------------------------------------------------------------------- #
def test_fractal_dimension_range(_rv: np.ndarray) -> None:
    """fractal_dim_wavelet should return a finite positive number."""
    d = fractal_dim_wavelet(_rv, scales=[2, 4, 8, 16])
    assert math.isfinite(d) and d > 0.0


# --------------------------------------------------------------------------- #
# fit_rfsf
# --------------------------------------------------------------------------- #
def test_rfsv_basic(_rv: np.ndarray) -> None:
    """Returns finite H in (0,1) and sigma > 0."""
    fract_dim = 1.5  # dummy driver
    out = fit_rfsf(_rv, sub=4, fract_dim=fract_dim)

    assert 0.0 < out["rfsf_H"] < 1.0
    assert out["rfsf_sigma"] > 0.0
    assert isinstance(out["rfsf_ok"], bool)


# --------------------------------------------------------------------------- #
# fit_student_t_egarch
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    pytest.importorskip("arch", reason="arch package not installed") is None,
    reason="arch package missing",
)
def test_egarch_fit(_returns: np.ndarray) -> None:
    """EGARCH fit should converge (or at least return the key fields)."""
    z = (_returns - _returns.mean()) / (_returns.std(ddof=0) or 1.0)
    pars, elapsed = fit_student_t_egarch(z, max_retries=1)

    expected_keys = {
        "garch_omega",
        "garch_alpha",
        "garch_beta",
        "garch_df",
        "garch_llf",
        "garch_ok",
    }
    assert expected_keys <= pars.keys()
    assert math.isfinite(elapsed) and elapsed >= 0.0
