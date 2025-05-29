"""
Per-ticker Gaussian HMM labelling (regime detection).

Details
--------
Reads `preprocessed.parquet` (feature rich, NaN/inf free, OHE tickers),
fits a Gaussian HMM to a volatility proxy column (set to `rv`),
writes an enriched parquet with an integer `regime` column and a summary CSV.
Caching is stored in `paths.artifacts_dir`.

Configuration (configs/hmm.yaml)
--------------------------------
paths.interim_out   : data/interim/preprocessed.parquet
paths.hmm_out       : data/interim/hmm.parquet
paths.artifacts_dir : artifacts/hmm
paths.reports_dir   : reports/synergy
hmm.n_states        : 3
hmm.covariance_type : full
hmm.random_state    : 42
hmm.max_workers     : 8
hmm.vol_col         : rv

Outputs
-------
- `paths.hmm_out` (hmm.parquet)
- `paths.reports_dir`/hmm_regime_counts.csv
- `paths.artifacts_dir`/artifacts/hmm/<CACHE_SCHEME>.npz

Public API
----------
- main : command-line entry-point used by `make hmm [--debug]`
"""

from __future__ import annotations

import argparse
import logging
import warnings
from collections.abc import Sequence
from hashlib import blake2s
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl
import yaml
from hmmlearn import hmm
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning

# --------------------------------------------------------------------------- #
# Configuration & global constants
# --------------------------------------------------------------------------- #

_CFG_PATH = Path("configs/hmm.yaml")
_CFG = yaml.safe_load(_CFG_PATH.read_text())
_PATHS = _CFG["paths"]
_HCFG = _CFG["hmm"]

N_STATES = int(_HCFG["n_states"])
COV_TYPE = str(_HCFG["covariance_type"])
RAND_SEED = int(_HCFG["random_state"])
MAX_WORKER = int(_HCFG["max_workers"])
VOL_COL = str(_HCFG["vol_col"])
CACHE_PATTERN = "{digest}_{sym}_{states}{cov}{seed}.npz"

SRC_PARQ = Path(_PATHS["interim_out"])
DST_PARQ = Path(_PATHS["hmm_out"])
ART_DIR = Path(_PATHS["artifacts_dir"])
REPORT_DIR = Path(_PATHS["reports_dir"])

ART_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Logging 
log = logging.getLogger(__name__.split(".")[-1])
if not log.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
    log.addHandler(_handler)
log.setLevel(logging.INFO)

# Suppress HMM convergence warnings
warnings.filterwarnings(
    "ignore",
    category=ConvergenceWarning,
    module="hmmlearn"
    )

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def _cache_key(sym: str, data: np.ndarray) -> Path:
    """Return artifact file path for this ticker & HMM configuration."""
    digest = blake2s(data.tobytes(), digest_size=6).hexdigest()
    fname = CACHE_PATTERN.format(
        digest=digest,
        sym=sym,
        states=N_STATES,
        cov=COV_TYPE[0],
        seed=RAND_SEED,
    )
    return ART_DIR / fname


def _worker(sym: str, grp: pl.DataFrame) -> pl.DataFrame:
    """Fit Gaussian HMM on grp[VOL_COL]; add integer `regime` column."""
    start = perf_counter()
    
    if VOL_COL not in grp.columns:
        return grp.with_columns(pl.lit(0, dtype=pl.Int8).alias("regime"))

    vals = grp[VOL_COL].to_numpy().astype(np.float64)
    mask = np.isfinite(vals)
    arr = vals[mask].reshape(-1, 1)
    if arr.shape[0] < N_STATES:
        return grp.with_columns(pl.lit(0, dtype=pl.Int8).alias("regime"))

    cache_fp = _cache_key(sym, arr)
    if cache_fp.exists():
        regimes = np.load(cache_fp)["regime"].astype(np.int8)
        
    else:
        model = hmm.GaussianHMM(
            n_components=N_STATES,
            covariance_type=COV_TYPE,
            random_state=RAND_SEED,
        )
        model.fit(arr)
        regimes_full = np.copy(vals)
        regimes_full[:] = np.nan
        regimes_full[mask] = model.predict(arr).astype(np.int8)
        regimes = np.where(
            np.isfinite(regimes_full),
            regimes_full,
            -1
            ).astype(np.int8)
        np.savez_compressed(cache_fp, regime=regimes)

    log.debug("%s fitted in %.2fs", sym, perf_counter() - start)
    return grp.with_columns(pl.Series("regime", regimes))

# --------------------------------------------------------------------------- #
# CLI entrypoint
# --------------------------------------------------------------------------- #

def main(argv: Sequence[str] | None = None) -> None:
    """Entrypoint for per-ticker HMM labelling."""
    parser = argparse.ArgumentParser("Per-ticker HMM labelling")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="DEBUG logs + per-ticker timing"
        )
    args = parser.parse_args(argv)
    
    if args.debug:
        log.setLevel(logging.DEBUG)

    if not SRC_PARQ.exists():
        log.error("Source parquet %s missing", SRC_PARQ)
        return

    df = pl.read_parquet(SRC_PARQ)
    if VOL_COL not in df.columns:
        log.error("Column '%s' missing in parquet", VOL_COL)
        return

    tasks: list[tuple[str, pl.DataFrame]] = [
        (str(k[0]) if isinstance(k, tuple) else str(k), grp)
        for k, grp in df.group_by("ticker", maintain_order=True)
    ]

    log.info("Fitting HMM - %d tickers, %d states", len(tasks), N_STATES)

    results = Parallel(n_jobs=MAX_WORKER,prefer="processes")(
        delayed(_worker)(sym, grp) for sym, grp in tasks
        )

    out_df = pl.concat(results, how="vertical")
    DST_PARQ.parent.mkdir(parents=True, exist_ok=True)
    out_df.write_parquet(DST_PARQ)
    log.info("HMM labels written -> %s", DST_PARQ)

    summary = (
        out_df.group_by(["ticker", "regime"], maintain_order=True)
        .len()
        .pivot(index="ticker", on="regime", values="len")
        .fill_null(0)
    )
    csv_fp = REPORT_DIR / "hmm_regime_counts.csv"
    summary.write_csv(csv_fp)
    log.info("Summary CSV      -> %s", csv_fp)


if __name__ == "__main__":
    main()
