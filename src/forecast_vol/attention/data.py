"""
Dataset builder for the attention forecaster.

Details
--------
Merges minute-bars, stochastic features, and pre-computed GNN embeddings,
derives the forward-realised variance target (`rv_fwd`), drops constant
columns, median-imputes / standard-scales numeric fields, and adds
the fitted `StandardScaler`. Returns an in memory df, writes
a list of features and a compressed joblib.

Configuration (configs/attention.yaml)
--------------------------------------
paths.basic_parquet    : data/interim/minute.parquet
paths.stoch_parquet    : data/intermediate/stochastic.parquet
paths.gnn_parquet      : data/intermediate/gnn.parquet
paths.scaler_pkl       : models/attention/std_scaler.pkl
paths.reports_dir      : reports/attention
data.lookahead         : 60          
split.val_frac         : 0.15
split.test_frac        : 0.10

Outputs
-------
- numpy-ready `(df, feature_cols, target)` returned in memory
- `paths.scaler_pkl`  (std_scaler.pkl)
- `paths.reports_dir`/feature_list.csv

Public API
----------
- main : command-line entry-point used by `make final-dataset [--debug]`
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl
import yaml
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

# --------------------------------------------------------------------------- #
# Configuration & global constants
# --------------------------------------------------------------------------- #

_CFG_PATH = Path("configs/attention.yaml")
_CFG = yaml.safe_load(_CFG_PATH.read_text())
_PATHS = _CFG["paths"]
LOOKAHEAD = int(_CFG["data"]["lookahead"])
F_VAL  = float(_CFG["split"]["val_frac"])
F_TEST = float(_CFG["split"]["test_frac"])


BSC_PARQ = Path(_PATHS["basic_parquet"])
STC_PARQ = Path(_PATHS["stoch_parquet"])
DST_PKL = Path(_PATHS["scaler_pkl"])
DST_PKL.parent.mkdir(parents=True, exist_ok=True)

REPORT_DIR = Path(_PATHS["reports_dir"])
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SNAP_DIR = Path("data/interim/synergy_daily")
snap_files = {p.stem: p for p in SNAP_DIR.glob("*.parquet")}


# Logging 
log = logging.getLogger(__name__.split(".")[-1])
if not log.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
    log.addHandler(_handler)
log.setLevel(logging.INFO)

# --------------------------------------------------------------------------- #
# Core Logic
# --------------------------------------------------------------------------- #

def build_dataset() -> tuple[pd.DataFrame, list[str], str]:
    """Return minute, stochastic, and GNN tables -> pandas DataFrame."""
    basic = (
        pl.read_parquet(BSC_PARQ)
        .with_columns(pl.col("dt").dt.strftime("%Y-%m-%d").alias("day"))
    )
    stoch = pl.read_parquet(STC_PARQ)
    gnn = pl.concat(
        [pl.read_parquet(fp) for fp in SNAP_DIR.glob("*.parquet")],
        how="vertical",
    )

    def n_numeric(tbl: pl.DataFrame) -> int:
        return sum(dtype.is_numeric() for dtype in tbl.dtypes)

    for tag, tbl in [("basic", basic), ("stoch", stoch), ("gnn", gnn)]:
        log.debug(f"{tag:6s} numeric cols: {n_numeric(tbl)}")

    df = (
        basic.join(gnn, on=["ticker","day"], how="inner")
            .join(stoch,   on="ticker", how="inner")
            .sort(["t"])
    )

    num_cols = [c for c in df.columns if df[c].dtype.is_numeric()]
    log.debug(f"JOIN   numeric cols: {len(num_cols)}")
    
    # Build forward-realised variance if req.
    if "rv_fwd" not in basic.columns:
        if "rv" in basic.columns:
            base = pl.col("rv")
        elif "log_ret" in basic.columns:
            base = pl.col("log_ret") ** 2
        else:
            log.error("Minute parquet must contain 'rv' or 'log_ret'.")
            return
        
        basic = basic.with_columns(
            (base.shift(-LOOKAHEAD).rolling_sum(LOOKAHEAD)).alias("rv_fwd")
        )
    basic = basic.drop_nulls(["rv_fwd"])

    df = (
        basic.join(gnn, on=["ticker","day"], how="inner")
            .join(stoch,   on="ticker", how="inner")
            .sort(["t"])
    )


    pdf: pd.DataFrame = df.to_pandas(use_pyarrow_extension_array=False)
    target = "rv_fwd"
    features = [c for c in pdf.select_dtypes("number").columns if c != target]

    log.debug(f"AFTER  numeric cols: {len(features)}")

    zero_std = pdf[features].std(axis=0, skipna=True) == 0.0
    drop_cols = zero_std[zero_std].index.tolist()
    if drop_cols:
        pdf.drop(columns=drop_cols, inplace=True)
        features = [c for c in features if c not in drop_cols]
        log.debug("Dropped constant columns: %s", drop_cols)

    pdf[features] = pdf[features].fillna(
        pdf[features]
        .median(axis=0, skipna=True)
        )
    scaler = StandardScaler()
    pdf[features] = scaler.fit_transform(pdf[features].values)
    joblib.dump(scaler, DST_PKL, compress=3)

    pd.Series(
        features, name="features"
        ).to_csv(
            REPORT_DIR / "feature_list.csv", index=False
            )

    return pdf, features, target

def chrono_split(
    pdf: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological (train, val, test) split."""
    n = len(pdf)
    n_train = int(n * (1 - F_VAL - F_TEST))
    n_val   = int(n * F_VAL)

    pdf_sorted = pdf.sort_values("t", kind="mergesort")
    return (
        pdf_sorted.iloc[:n_train],
        pdf_sorted.iloc[n_train : n_train + n_val],
        pdf_sorted.iloc[n_train + n_val :],
    )

class MinuteDataset(Dataset):
    """Torch dataset: returns `(x, y)` float32 arrays."""

    def __init__(
        self,
        df: pd.DataFrame,
        feats: list[str],
        tgt: str
        ) -> None:
        self.x = df[feats].to_numpy(np.float32, copy=False)
        self.y = df[[tgt]].to_numpy(np.float32, copy=False)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]

# --------------------------------------------------------------------------- #
# CLI entrypoint
# --------------------------------------------------------------------------- #

def main(argv: Sequence[str] | None = None) -> None:
    """Run end-to-end dataset assembly and print split sizes."""
    parser = argparse.ArgumentParser("Build attention dataset")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="verbose logs"
        )
    args = parser.parse_args(argv)
    
    if args.debug:
        log.setLevel(logging.DEBUG)

    df, feats, _ = build_dataset()
    tr, va, te = chrono_split(df)
    log.info(
        "Dataset ready -> train:%d  val:%d  test:%d  (features:%d)",
        len(tr),
        len(va),
        len(te),
        len(feats),
    )
    log.info("Feature list saved -> %s/feature_list.csv", REPORT_DIR)


if __name__ == "__main__":
    main()
