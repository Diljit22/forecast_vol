# stochastic_models.py

import numpy as np
import pandas as pd
import logging
import pywt
from fbm import FBM
from typing import Dict, Any
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from scipy.optimize import minimize
from math import log

logger = logging.getLogger(__name__)


# Wavelet-based Hurst exponent (RFSV)


def wavelet_estimate_hurst(rv_series, wavelet="haar", levels=6, eps=1.0e-8):
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

    mask = ~np.isnan(log_scales) & ~np.isnan(log_vars)
    if np.sum(mask) < 2:
        return 0.1

    slope, _ = np.polyfit(log_scales[mask], log_vars[mask], 1)
    hurst_est = slope / 2
    return max(0.01, min(0.99, hurst_est))


def negative_log_likelihood(params, rv_series, eps=1.0e-8):
    """
    RFSV param fitting via approximate log-likelihood.
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
    if len(rv_series) < 2:
        logger.warning("rv_series too short for RFSV, returning fallback params.")
        mean_val = np.mean(rv_series) if len(rv_series) else 1e-5
        return {
            "hurst_exponent": 0.1,
            "mu": -10,
            "sigma": 0.1,
            "long_run_var": float(mean_val),
            "vol_of_vol": 0.1,
        }

    best_params = None
    best_obj = 1e15

    H_init = wavelet_estimate_hurst(rv_series, wavelet=wavelet, levels=levels)
    mu_init = np.mean(np.log(rv_series + eps))
    sigma_init = np.std(np.diff(np.log(rv_series + eps))) or 1e-3

    from scipy.optimize import Bounds

    param_bounds = Bounds([0.01, -np.inf, eps], [0.99, np.inf, np.inf])

    for _ in range(random_restarts):
        init_H = np.clip(H_init + 0.1 * (np.random.rand() - 0.5), 0.01, 0.99)
        init_mu = mu_init + (np.random.rand() - 0.5)
        init_sigma = max(eps, sigma_init * (1 + 0.5 * (np.random.rand() - 0.5)))
        init_guess = [init_H, init_mu, init_sigma]

        result = minimize(
            negative_log_likelihood,
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
        "long_run_var": long_run_var,
        "vol_of_vol": sigma_mle,
    }


def simulate_rough_vol_path(params, n_steps=100):
    if not params or "hurst_exponent" not in params:
        logger.warning("simulate_rough_vol_path called with empty params.")
        return np.array([])

    H = params["hurst_exponent"]
    sigma = params["sigma"]
    mu = params["mu"]
    long_run_var = params["long_run_var"]

    fbm_gen = FBM(n=n_steps, hurst=H, length=1.0, method="cholesky")
    fbm_samples = fbm_gen.fbm()
    fbm_samples = fbm_samples[:n_steps]

    increments = mu + sigma * (fbm_samples - fbm_samples[0])
    rough_vol = np.exp(increments)
    final_vol = long_run_var * rough_vol
    return final_vol


def rough_fsv_pipeline(rv_series, config):
    stoch_cfg = config.get("stochastic", {})
    wavelet_name = stoch_cfg.get("wavelet_name", "haar")
    levels = stoch_cfg.get("wavelet_levels", 6)
    n_steps = stoch_cfg.get("n_steps_sim", 100)
    random_restarts = stoch_cfg.get("random_restarts", 3)

    # Cache check

    params = fit_rfsv_params(
        rv_series, wavelet=wavelet_name, levels=levels, random_restarts=random_restarts
    )
    sim_path = simulate_rough_vol_path(params, n_steps=n_steps)

    return {"params": params, "simulated_path": sim_path}


# GARCH(1,1) approach


def minimal_garch_fit(returns: np.ndarray):
    """
    Placeholder for GARCH(1,1) fitting.
    In production, consider using 'arch' library for robust fitting.
    """
    return {"omega": 1e-6, "alpha": 0.05, "beta": 0.9}


def simulate_garch_path(params, n_steps=100):
    if not params or "omega" not in params:
        logger.warning("simulate_garch_path called with empty params.")
        return np.array([])

    omega = params["omega"]
    alpha = params["alpha"]
    beta = params["beta"]

    np.random.seed(42)
    returns = []
    sigma2 = [omega / (1 - alpha - beta) if (alpha + beta < 1) else 1e-4]
    for _ in range(n_steps):
        stdev = np.sqrt(sigma2[-1])
        r_t = stdev * np.random.randn()
        returns.append(r_t)
        next_sigma2 = omega + alpha * (r_t**2) + beta * sigma2[-1]
        sigma2.append(next_sigma2)

    return np.array(returns)


# Multi-scale fractal approach


def multi_scale_volatility_estimate(
    rv_series, wavelet="haar", scales=[2, 4, 8], eps=1e-8
):
    """
    Multi-scale approach using wavelet decomposition.
    fractal_dim ~ 2 - slope from log-log wavelet energy plot.
    """
    if len(rv_series) < max(scales):
        logger.warning("rv_series too short for multi-scale approach. returning empty.")
        return {}

    x = np.log(rv_series + eps)
    scale_energy = []
    for s in scales:
        coeffs = pywt.wavedec(x, wavelet, level=s)
        d = coeffs[-1]
        energy = np.sum(d**2)
        scale_energy.append((s, energy))

    s_vals = np.array([s for (s, e) in scale_energy], dtype=float)
    e_vals = np.array([e for (s, e) in scale_energy], dtype=float)
    mask = e_vals > 0
    if sum(mask) < 2:
        return {"fractal_dim": 2.0, "scale_energy": dict(scale_energy)}

    log_s = np.log(s_vals[mask])
    log_e = np.log(e_vals[mask])
    slope, intercept = np.polyfit(log_s, log_e, 1)
    fractal_dim = 2.0 - slope

    return {
        "fractal_dim": float(fractal_dim),
        "scale_energy": dict(scale_energy),
        "slope": float(slope),
        "intercept": float(intercept),
    }

def run_stochastic_models(df_or_series, config):
    """
    Main entry to run wavelet-based RFSV, GARCH, and multi-scale fractal.
    """
    stoch_cfg = config.get("stochastic", {})
    if not stoch_cfg.get("enabled", False):
        logger.info("Stochastic models not enabled. Returning.")
        return {}

    results = {}

    # 1) RFSV
    if stoch_cfg.get("wavelet_name", None):
        rfsv_out = rough_fsv_pipeline(df_or_series, config)
        results["rfsv"] = rfsv_out

    # 2) GARCH
    if stoch_cfg.get("use_garch", False):
        garch_params = minimal_garch_fit(df_or_series.values)
        garch_sim = simulate_garch_path(
            garch_params, n_steps=stoch_cfg.get("n_steps_sim", 100)
        )
        results["garch_params"] = garch_params
        results["garch_sim"] = garch_sim

    # 3) Multi-scale fractal
    if stoch_cfg.get("multi_scale", False):
        scales = stoch_cfg.get("fractal_scales", [2, 4, 8])
        fractal_res = multi_scale_volatility_estimate(
            df_or_series.values, scales=scales
        )
        results["multi_scale"] = fractal_res

    return results
