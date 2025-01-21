"""
Implements wavelet-based Hurst exponent estimation for Rough FSV,
plus MLE fitting for (H, mu, sigma), and simulation.
"""

import logging
import numpy as np
import pywt
from fbm import FBM
from math import log
from scipy.optimize import minimize, Bounds

logger = logging.getLogger(__name__)


def wavelet_estimate_hurst(rv_series, wavelet="haar", levels=6, eps=1e-8):
    """
    Computes an initial Hurst exponent estimate via wavelet decomposition.
    """
    x = np.log(rv_series + eps)
    coeffs = pywt.wavedec(x, wavelet, level=levels)
    detail_vars = []
    scales = []
    for i, d in enumerate(coeffs[1:], start=1):
        var_d = np.var(d) if len(d) else eps
        detail_vars.append(var_d)
        scales.append(2**i)

    log_scales = np.log(scales)
    detail_vars = np.array(detail_vars)
    detail_vars[detail_vars <= 0] = eps
    log_vars = np.log(detail_vars)

    mask = (~np.isnan(log_scales)) & (~np.isnan(log_vars))
    if np.sum(mask) < 2:
        return 0.1

    slope, _ = np.polyfit(log_scales[mask], log_vars[mask], 1)
    hurst_est = slope / 2
    return max(0.01, min(0.99, hurst_est))


def _negative_log_likelihood(params, rv_series, eps=1e-8):
    """
    Approximate negative log-likelihood for RFSV param fitting.
    """
    H, mu, sigma = params
    if H <= 0 or H >= 1 or sigma <= 0:
        return 1e10

    log_rv = np.log(rv_series + eps)
    n = len(log_rv)
    increments = np.diff(log_rv)
    residuals = increments - (mu / max(1, n))

    ll = 0.5 * np.sum((residuals / sigma) ** 2) + n * log(sigma + 1e-12)
    return ll


def fit_rfsv_params(rv_series, wavelet="haar", levels=6, random_restarts=3, eps=1e-8):
    """
    Fits (H, mu, sigma) to rv_series. Returns dict with hurst_exponent, mu, sigma, etc.
    """
    if len(rv_series) < 2:
        logger.warning("rv_series too short for RFSV, returning fallback params.")
        mean_val = float(np.mean(rv_series)) if len(rv_series) else 1e-5
        return {
            "hurst_exponent": 0.1,
            "mu": -10,
            "sigma": 0.1,
            "long_run_var": mean_val,
            "vol_of_vol": 0.1,
        }

    # Initial guesses
    H_init = wavelet_estimate_hurst(rv_series, wavelet=wavelet, levels=levels)
    mu_init = float(np.mean(np.log(rv_series + eps)))
    sigma_init = float(np.std(np.diff(np.log(rv_series + eps))) or 1e-3)

    param_bounds = Bounds([0.01, -np.inf, eps], [0.99, np.inf, np.inf])

    best_params = None
    best_obj = 1e15

    for _ in range(random_restarts):
        init_H = np.clip(H_init + 0.1 * (np.random.rand() - 0.5), 0.01, 0.99)
        init_mu = mu_init + (np.random.rand() - 0.5)
        init_sigma = max(eps, sigma_init * (1 + 0.5 * (np.random.rand() - 0.5)))
        init_guess = [init_H, init_mu, init_sigma]

        result = minimize(
            _negative_log_likelihood,
            init_guess,
            args=(rv_series,),
            method="L-BFGS-B",
            bounds=param_bounds,
        )
        if result.success and result.fun < best_obj:
            best_obj = result.fun
            best_params = result.x

    if best_params is None:
        logger.warning("RFSV MLE did not converge; fallback to wavelet inits.")
        best_params = [H_init, mu_init, sigma_init]

    H_mle, mu_mle, sigma_mle = best_params
    long_run_var = float(np.mean(rv_series))
    return {
        "hurst_exponent": float(H_mle),
        "mu": float(mu_mle),
        "sigma": float(sigma_mle),
        "long_run_var": float(long_run_var),
        "vol_of_vol": float(sigma_mle),
    }


def simulate_rough_vol_path(params, n_steps=100):
    """
    simulation
    """
    if not params or "hurst_exponent" not in params:
        logger.warning("simulate_rough_vol_path called with empty params.")
        return np.array([])

    H = params["hurst_exponent"]
    sigma = params["sigma"]
    mu = params["mu"]
    long_run_var = params["long_run_var"]

    fbm_gen = FBM(n=n_steps, hurst=H, length=1.0, method="cholesky")
    fbm_samples = fbm_gen.fbm()[:n_steps]
    increments = mu + sigma * (fbm_samples - fbm_samples[0])
    rough_vol = np.exp(increments)
    final_vol = long_run_var * rough_vol
    return final_vol
