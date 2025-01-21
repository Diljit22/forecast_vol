"""
Multi-scale fractal dimension estimation using wavelet decomposition.
"""

import logging
import numpy as np
import pywt

logger = logging.getLogger(__name__)


def multi_scale_volatility_estimate(rv_series, wavelet="haar", scales=[2,4,8], eps=1e-8):
    """
    Decompose rv_series at multiple scales, measure wavelet energy, 
    estimate fractal_dim ~ 2 - slope from log-log plot.
    Returns dict with fractal_dim, slope, intercept, scale_energy, etc.
    """
    if len(rv_series) < max(scales):
        logger.warning("rv_series too short for multi-scale fractal. returning empty.")
        return {"fractal_dim": 2.0, "scale_energy": {}, "slope": 0.0, "intercept": 0.0}

    x = np.log(rv_series + eps)
    scale_energy = []
    for s in scales:
        coeffs = pywt.wavedec(x, wavelet, level=s)
        d = coeffs[-1]
        energy = np.sum(d**2)
        scale_energy.append((s, energy))

    s_vals = np.array([s for (s, e) in scale_energy], dtype=float)
    e_vals = np.array([e for (s, e) in scale_energy], dtype=float)
    mask = (e_vals > 0)
    if sum(mask) < 2:
        return {"fractal_dim": 2.0, "scale_energy": dict(scale_energy), "slope":0.0, "intercept":0.0}

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
