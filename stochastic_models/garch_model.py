"""
Implements a GARCH(1,1) fitting approach using the arch library for better realism.
"""

import logging
import numpy as np
import pandas as pd
from arch import arch_model  # pip install arch
logger = logging.getLogger(__name__)


def fit_garch_params(returns: np.ndarray):
    """
    Fit GARCH(1,1) using arch library
    """
    if len(returns) < 10:  # arbitrary minimal length
        logger.warning("Not enough data for GARCH fitting. Using defaults.")
        return {"omega": 1e-6, "alpha": 0.05, "beta": 0.9, "success": False}

    try:
        am = arch_model(returns, p=1, q=1, rescale=False)
        res = am.fit(disp="off")  # no console printing
        params = res.params
        garch_params = {
            "omega": float(params.get("omega", 1e-6)),
            "alpha": float(params.get("alpha[1]", 0.05)),
            "beta": float(params.get("beta[1]", 0.9)),
            "success": True
        }
        return garch_params
    except Exception as e:
        logger.error(f"GARCH fitting failed: {e}. Using defaults.")
        return {"omega": 1e-6, "alpha": 0.05, "beta": 0.9, "success": False}


def simulate_garch_path(garch_params: dict, n_steps=100):
    """
    GARCH(1,1) simulation from fitted params. 
    """
    if not garch_params.get("success", False):
        logger.warning("simulate_garch_path called with unsuccessful GARCH params. Returning empty.")
        return np.array([])

    np.random.seed(42)
    omega = garch_params["omega"]
    alpha = garch_params["alpha"]
    beta = garch_params["beta"]

    returns = []
    sigma2 = [omega / (1 - alpha - beta) if (alpha + beta < 1) else 1e-4]
    for _ in range(n_steps):
        stdev = np.sqrt(sigma2[-1])
        r_t = stdev * np.random.randn()
        returns.append(r_t)
        next_sigma2 = omega + alpha * (r_t**2) + beta * sigma2[-1]
        sigma2.append(next_sigma2)

    return np.array(returns)
