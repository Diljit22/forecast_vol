"""
Compute and write static stochastic features per ticker.

Details
--------
Orchestrates wavelet fractal, EGARCH, and RFSV fits in parallel,
outputs a parquet of features, a CSV report, and parameter histograms.

Configuration (configs/stoch.yaml)
----------------------------------
paths.hmm_out             : data/interim/hmm.parquet
paths.stoch_out           : data/interim/stochastic.parquet
paths.reports_dir         : reports/stochastic
params.jobs               : 8
params.egarch.max_retries : 2
params.fractal.scales     : [2,4,8]
params.fractal.wavelet    : haar
params.rfsf.sub           : 5

Outputs
-------
- `paths.stoch_out` (stochastic.parquet)
- `paths.reports_dir`/fit_report.csv
- `paths.reports_dir`/params_hist_<col>.png

Public API
----------
- main : command-line entry-point used by `make stochastic [--jobs 8 --debug]`
"""
from __future__ import annotations

import argparse
import logging
import warnings
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml
from arch.univariate.base import DataScaleWarning as _DW  # noqa: N814
from arch.utility.exceptions import ConvergenceWarning as _CW  #noqa N814

from .egarch import fit_student_t_egarch
from .fractal import fractal_dim_wavelet
from .rfsv import fit_rfsf

# --------------------------------------------------------------------------- #
# Configuration & global constants
# --------------------------------------------------------------------------- #

_CFG_PATH = Path("configs/stoch.yaml")
_CFG = yaml.safe_load(_CFG_PATH.read_text())
_PATHS = _CFG["paths"]
SCALES = _CFG["params"]["fractal"]["scales"]
WAVELET = _CFG["params"]["fractal"]["wavelet"]
MAX_RETRIES = _CFG["params"]["egarch"]["max_retries"]
JOBS = _CFG["params"]["jobs"]
RFSF_SUB = _CFG["params"]["rfsf"]["sub"]

SRC_PARQ = Path(_PATHS["hmm_out"])
DST_PARQ = Path(_PATHS["stoch_out"])
REPORT_DIR = Path(_PATHS["reports_dir"])

DST_PARQ.parent.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore", category=_CW)
warnings.filterwarnings("ignore", category=_DW)

# Logging 
log = logging.getLogger(__name__.split(".")[-1])
if not log.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
    log.addHandler(_handler)
log.setLevel(logging.INFO)

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def _worker(sym: str, pdf: pl.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute all stochastic features for one ticker DataFrame."""
    t0 = perf_counter()
    
    rv = pdf["rv"].to_numpy()
    rv = np.clip(rv, 1e-12, None)
    ret_raw = pdf["ret"].to_numpy()
    mu, sig = ret_raw.mean(), ret_raw.std(ddof=0) or 1.0
    ret_z = (ret_raw - mu) / sig

    feat: dict[str, Any] = {"ticker": sym}
    
    # fractal
    fract_dim = fractal_dim_wavelet(rv, scales=SCALES, wavelet=WAVELET)
    feat["fract_dim"] = fract_dim
    
    # egarch
    gpars, t_g = fit_student_t_egarch(ret_z, MAX_RETRIES)
    feat.update(gpars)
    
    # rfsv
    feat.update(fit_rfsf(rv, RFSF_SUB, fract_dim))

    timing = {
        "ticker": sym,
        "t_total": perf_counter() - t0,
        "t_garch": t_g,
        }
    
    return feat, timing

# --------------------------------------------------------------------------- #
# CLI entrypoint
# --------------------------------------------------------------------------- #

def main(argv: Sequence[str] | None = None) -> None:
    """Entrypoint for static stochastic feature computation."""
    parser = argparse.ArgumentParser("Stochastic features")
    parser.add_argument("--jobs", type=int, default=JOBS)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="DEBUG logs + progress"
        )
    args = parser.parse_args(argv)
    
    if args.debug:
        log.setLevel(logging.DEBUG)

    if not SRC_PARQ.exists():
        log.error("HMM parquet missing: %s", SRC_PARQ)
        return
    
    df = pl.read_parquet(SRC_PARQ)
    tasks = [
        (str(k[0]) if isinstance(k, tuple) else str(k), grp)
        for k, grp in df.group_by("ticker", maintain_order=True)
    ]
    log.info("Fitting stochastic features for %d tickers ...", len(tasks))

    results = joblib.Parallel(n_jobs=args.jobs)(
        joblib.delayed(_worker)(sym, grp) for sym, grp in tasks
    )
    feats, times = zip(*results, strict=False)

    feat_df = pl.DataFrame(feats).sort("ticker")
    time_df = pl.DataFrame(times).sort("ticker")
    wide = feat_df.join(time_df, on="ticker")

    feat_df = (
        feat_df
        .with_columns(
            pl.col("garch_alpha").clip(0.0, 1.0).alias("garch_alpha_clip")
        )
        .with_columns(
            pl.col("garch_alpha").log1p().alias("garch_alpha_log")
        )
        .drop("garch_llf")
    )

    
    o_min = feat_df["garch_omega"].min()
    o_max = feat_df["garch_omega"].max()
    denom = (o_max - o_min) or 1.0
    feat_df = feat_df.with_columns(
        ((pl.col("garch_omega") - o_min) / denom)
        .alias("garch_omega_scaled")
    )

    feat_df.write_parquet(DST_PARQ)
    
    wide.write_csv(REPORT_DIR / "fit_report.csv")
    log.info("Wrote stochastic parquet -> %s", DST_PARQ)
    log.info("Wrote fit report CSV -> %s", REPORT_DIR / "fit_report.csv")

    if args.debug:
        for r in feats:
            log.debug(
                "%s fract=%.3f beta=%.3f H=%.2f ok=%s",
                r["ticker"],
                r["fract_dim"],
                r["garch_beta"],
                r["rfsf_H"],
                r["garch_ok"],
            )

    # histograms
    for col in ["fract_dim", "garch_beta", "rfsf_H"]:
        vals = feat_df[col].drop_nulls().to_numpy()
        plt.clf()
        plt.hist(vals, bins=20, edgecolor="k")
        plt.title(col)
        plt.savefig(REPORT_DIR / f"params_hist_{col}.png", dpi=120)



if __name__ == "__main__":
    main()