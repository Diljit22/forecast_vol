"""
Fit RFSV (Rough Fractional Stochastic Volatility) via MLE.

Details
--------
Takes a realized-variance vector, sub-samples it, and runs a bounded
L-BFGS-B optimization of the negative log-likelihood under the assumption
that log-RV increments are i.i.d. Normal (0, sigma**2). The initial guess for
H is (fractal dimension / 2); sigma is the empirical stdev of log-increments.

Public API
----------
- fit_rfsf : return H, sigma, and a success flag
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.optimize as opt

# --------------------------------------------------------------------------- #
# Core function
# --------------------------------------------------------------------------- #

def fit_rfsf(
    rv: np.ndarray,
    sub: int,
    fract_dim: float,
    eps: float = 1e-8,
) -> dict[str, Any]:
    """
    Perform RFSV MLE on a sub-sampled realized-variance series.

    Parameters
    ----------
    rv
        Realized-variance series.
    sub
        Sub-sampling interval for the log-RV.
    fract_dim
        Fractal dimension computed from RV series.
    eps
        Small constant to avoid log(0) and division by 0.

    Returns
    -------
    dict
        Keys are: rfsf_H, rfsf_sigma, rfsf_ok.
    """
    logv = np.log(rv[::sub] + eps)
    dv = np.diff(logv)
    if dv.size < 10:
        return {"rfsf_H": np.nan, "rfsf_sigma": np.nan, "rfsf_ok": False}

    # initialize H = fract_dim/2, sigma = std(dv)
    H0 = max(0.05, min(0.95, fract_dim / 2)) #noqa N806
    s0 = float(np.std(dv) or eps)
    n = dv.size

    def _nll(params: np.ndarray) -> float:
        H, sig = params #noqa N806
        if not (0 < H < 1 and sig > 0):
            return np.inf
        var = sig**2
        return 0.5 * (n * np.log(2 * np.pi * var) + dv @ dv / var)

    res = opt.minimize(
        _nll,
        (H0, s0),
        bounds=[(0.01, 0.99), (eps, None)],
        method="L-BFGS-B",
        )
    
    return {
        "rfsf_H": float(res.x[0]),
        "rfsf_sigma": float(res.x[1]),
        "rfsf_ok": bool(res.success),
        }