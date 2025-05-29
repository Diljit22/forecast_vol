"""
Merge resampled Parquet, add minute level features and OHE tickers.

Details
--------
Loads all resampled files, pipes each frame through `basic_features.PIPE`,
drops the first `params.lookback` rows, equalises frame lengths,
one hot-encodes the ticker column, and stores a modelling-ready Parquet dataset.

Configuration (configs/preprocess.yaml)
---------------------------------------
paths.resampled_out : data/interim/resampled
paths.interim_out   : data/interim/preprocessed.parquet
params.lookback     : 35
paths.reports_dir   : reports/preprocess

Outputs
-------
- `paths.interim_out` (preprocessed.parquet)
- `paths.reports_dir`/dataset_shape.txt

Public API
----------
- main : command-line entry-point used by `make build-dataset [--debug]`
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from sklearn.preprocessing import OneHotEncoder

from .basic_features import PIPE

# --------------------------------------------------------------------------- #
# Configuration & global constants
# --------------------------------------------------------------------------- #

_CFG_PATH = Path("configs/preprocess.yaml")
_CFG = yaml.safe_load(_CFG_PATH.read_text())
_PATHS = _CFG["paths"]
LOOKBACK = int(_CFG["params"]["lookback"])

SRC_DIR = Path(_PATHS["resampled_out"])
DST_PARQ = Path(_PATHS["interim_out"])
REPORT_DIR = Path(_PATHS["reports_dir"])
REPORT_DIR.mkdir(parents=True, exist_ok=True)

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

def _apply_pipe(df: pl.DataFrame) -> pl.DataFrame:
    """Apply all feature functions in PIPE to the DataFrame."""
    for fn in PIPE:
        df = fn(df)
    return df

# --------------------------------------------------------------------------- #
# CLI entrypoint
# --------------------------------------------------------------------------- #

def main(argv: Sequence[str] | None = None) -> None:
    """CLI entrypoint for dataset construction."""
    parser = argparse.ArgumentParser(
        "Build dataset with basic features and one-hot encoding"
        )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="DEBUG logs + progress"
        )
    args = parser.parse_args(argv)
    
    if args.debug:
        log.setLevel(logging.DEBUG)

    files = sorted(SRC_DIR.glob("*.parquet"))
    if not files:
        log.error("No parquet in %s", SRC_DIR)
        return

    frames: list[pl.DataFrame] = []
    for fp in files:
        df = pl.read_parquet(fp)
        tic = time.perf_counter()
        tkr = fp.stem.upper()
        df = (
            df.with_columns(pl.lit(tkr).alias("ticker"))
            .with_columns(
                pl.col("t")
                .cast(pl.Datetime("ms"))
                .dt.replace_time_zone("UTC")
                .alias("dt")
            )
        )
        df = _apply_pipe(df)

        df = df.slice(LOOKBACK)
        df = df.fill_null("forward").fill_null("backward")

        df = df.with_columns(pl.col(pl.datatypes.Float64).cast(pl.Float32))
        frames.append(df)
        log.debug("%s feature pipe %.2fs", tkr, time.perf_counter() - tic)

    # Equalise lengths and stack
    min_len = min(f.height for f in frames)
    frames = [f.head(min_len) for f in frames]
    stacked = pl.concat(frames, how="vertical")

    # OHE
    tic_np = stacked["ticker"].to_numpy().reshape(-1, 1)
    ohe = OneHotEncoder(sparse_output=False, drop="first", dtype=np.int8)
    ohe_arr = ohe.fit_transform(tic_np)
    ohe_df = pl.from_numpy(
        ohe_arr, schema=ohe.get_feature_names_out(["ticker"]).tolist()
        )
    stacked = pl.concat([stacked, ohe_df], how="horizontal")
    stacked = stacked.drop_nulls()

    DST_PARQ.parent.mkdir(parents=True, exist_ok=True)
    stacked.write_parquet(DST_PARQ)
    log.info(
        "Dataset written -> %s (%d rows, %d cols)",
        DST_PARQ,
        stacked.height,
        stacked.width,
    )

    with (REPORT_DIR / "dataset_shape.txt").open("w") as fh:
        fh.write(f"rows={stacked.height}, cols={stacked.width}\n")


if __name__ == "__main__":
    main()
