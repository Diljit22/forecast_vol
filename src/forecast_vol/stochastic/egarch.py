"""
Fit Student-t EGARCH(1,1) models per ticker.

Details
--------
Takes a z-scored return series, tightening beta if needed, and returns
the converged parameters. All fits are done with arch.arch_model.

Public API
----------
- fit_student_t_egarch : return parameter dict and elapsed time
"""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
from arch import arch_model

# --------------------------------------------------------------------------- #
# Configuration & global constants
# --------------------------------------------------------------------------- #

MAX_BETA = 0.98

# --------------------------------------------------------------------------- #
# Core function
# --------------------------------------------------------------------------- #

def fit_student_t_egarch(
    z_ret: np.ndarray,
    max_retries: int,
) -> tuple[dict[str, Any], float]:
    """
    Fit EGARCH(1,1) with Student-t innovations to a pair of return series.

    Parameters
    ----------
    z_ret
        Z-scored return series (fed directly into arch).
    max_retries
        Number of retry attempts with tightened beta upper bound.

    Returns
    -------
    pars
        Dict with keys garch_omega, garch_alpha, garch_beta,
        garch_df, garch_llf, garch_ok.
    elapsed
        Seconds taken for the successful fit or last attempt.
    """
    tic = perf_counter()
    model = arch_model(
        z_ret,
        vol="EGarch",
        p=1,
        q=1,
        dist="t",
        rescale=False,
        hold_back=0,
        )
    
    for retry in range(max_retries):
        try:
            res = model.fit(disp="off")
            p = res.params

            omega = float(p["omega"])
            alpha = max(0.0, min(float(p["alpha[1]"]), 1.0))
            beta  = min(float(p["beta[1]"]), MAX_BETA)

            return {
                "garch_omega": omega,
                "garch_alpha": alpha,
                "garch_beta": beta,
                "garch_df": float(p["nu"]),
                "garch_llf": float(res.loglikelihood),
                "garch_ok": bool(getattr(res, "converged", True)),
            }, perf_counter() - tic
            
        except Exception:
            # tighten beta bound and retry
            upper = MAX_BETA - .05 * (retry + 1)
            model = arch_model(
                z_ret,
                vol="EGarch", p=1, q=1, dist="t",
                rescale=False,
                hold_back=0,
                constraints=(([0,0,1,0],), ([0.0, upper],)),
            )
            
    nan_dict = dict.fromkeys(
        [
            "garch_omega",
            "garch_alpha",
            "garch_beta",
            "garch_df",
            "garch_llf",
            ],
        np.nan,
        )
    nan_dict["garch_ok"] = False
    
    return nan_dict, perf_counter() - tic