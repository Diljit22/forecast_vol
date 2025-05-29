"""
Compute fractal dimension via wavelet decomposition.

Details
--------
Runs a log(log()) regression of wavelet detail energies across a set of
dyadic scales and returns translated slope as the fractal dimension.

Public API
----------
- fractal_dim_wavelet : return fractal dimension from an RV series
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pywt

# --------------------------------------------------------------------------- #
# Core function
# --------------------------------------------------------------------------- #

def fractal_dim_wavelet(
    rv: np.ndarray,
    scales: Sequence[int] | np.ndarray,
    wavelet: str = "haar",
    eps: float = 1e-8,
) -> float:
    """Estimate fractal dimension from a realized-variance series.

    Parameters
    ----------
    rv
        Realized-variance series (positive floats).
    scales
        Iterable of dyadic scales.
    wavelet
        PyWavelets wavelet (default is `haar`).
    eps
        Small constant to avoid log(0).

    Returns
    -------
    float
        Estimated fractal dimension (2 - slope of log-energy vs log-scale).
    """
    wav = pywt.Wavelet(wavelet)
    max_possible = pywt.dwt_max_level(rv.size, wav.dec_len)
    max_lev = min(int(np.log2(max(scales))), max_possible)

    coeffs = pywt.wavedec(
        rv.astype(np.float32),
        wavelet,
        mode="periodization",
        level=max_lev
        )
    
    energies: list[float] = []
    for s in scales:
        lev = int(np.log2(s))
        detail = coeffs[max_lev - lev + 1]
        energies.append(float(detail @ detail + eps))
        
    slope, _ = np.polyfit(np.log2(scales), np.log2(energies), 1)
    
    return 2.0 - slope
