"""
Generate daily GNN based correlation embeddings.

Details
--------
For each valid day computes a Pearson-correlation matrix on a
time-frame window, adds per-ticker statistics and feeds the graph into a
trained GNN to obtain a dense embedding. 

Configuration (configs/gnn.yaml)
---------------------------------------
paths.hmm_out           : data/interim/hmm.parquet
paths.reports_dir       : reports/synergy
gnn.correlation_window  : 50
gnn.epochs              : 100
gnn.device              : cpu
gnn.benches             : - SPY - VXX - QQQ

Outputs
-------
- `data/interim/synergy_daily/<YYYY-MM-DD>.parquet` (one per trading day)

Public API
----------
- main : command-line entry-point used by `make gnn-snapshots [--debug]`
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import polars as pl
import torch
import yaml

from .build_corr import build_corr
from .gnn_model import gnn_data, train_once

# --------------------------------------------------------------------------- #
# Configuration & global constants
# --------------------------------------------------------------------------- #

_CFG_PATH = Path("configs/gnn.yaml")
_CFG = yaml.safe_load(_CFG_PATH.read_text())
_PATHS = _CFG["paths"]
BENCHES = [str(t) for t in _CFG["gnn"]["benches"]]
EPOCHS = int(_CFG["gnn"]["epochs"])
WINDOW = int(_CFG["gnn"]["correlation_window"])
DEVICE = str(_CFG["gnn"]["device"])

BEST_FP = Path(_PATHS["reports_dir"]) / "best_params.yaml"
BEST = yaml.safe_load(BEST_FP.read_text())

SRC_PARQ = Path(_PATHS["hmm_out"])
DST_DIR = Path("data/interim/synergy_daily")
DST_DIR.mkdir(parents=True, exist_ok=True)

# Logger
log = logging.getLogger("gnn_snap")
if not log.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
    log.addHandler(_handler)
log.setLevel(logging.INFO)

# --------------------------------------------------------------------------- #
# Helper Functions
# --------------------------------------------------------------------------- #

def _iter_days(df: pl.DataFrame):
    """Yield (YYYY-MM-DD, frame) pairs, preserving order."""
    for grp in df.partition_by("day", maintain_order=True):
        yield grp["day"][0], grp


# --------------------------------------------------------------------------- #
# CLI entrypoint
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser("daily GNN snapshots")
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args(argv)
    if args.debug:
        log.setLevel(logging.DEBUG)

    if not SRC_PARQ.exists():
        raise SystemExit(f"Source parquet missing: {SRC_PARQ}")

    df = pl.read_parquet(SRC_PARQ)
    if "day" not in df.columns:
        df = df.with_columns(
            df["dt"].dt.strftime("%Y-%m-%d").alias("day")
        )

    log.info("Building daily embeddings (window=%d)", args.window)
    for day, grp in _iter_days(df):
        dfs = {
            str(k[0]) if isinstance(k, tuple) else str(k): g
            for k, g in grp.group_by("ticker", maintain_order=True)
        }

        missing = [b for b in BENCHES if b not in dfs]
        if missing:  # skip day if any benchmark absent
            log.debug("Skip %s - missing %s", day, ",".join(missing))
            continue

        symbols, corr = build_corr(dfs, BENCHES, args.window)

        stats = (
            grp.select(["ticker", "vov", "iqv", "trade_intensity"])
            .group_by("ticker", maintain_order=True)
            .mean()
            .filter(pl.col("ticker").is_in(symbols))
            .sort("ticker")
        )

        data = gnn_data(symbols, corr, stats, BEST["thr"])
        emb, _ = train_once(
            data,
            hidden_dim=BEST["hidden_dim"],
            layers=BEST["layers"],
            lr=BEST["lr"],
            epochs=EPOCHS,
            device=DEVICE if torch.cuda.is_available() else "cpu",
        )

        cols = [f"gnn_emb_{i}" for i in range(emb.shape[1])]
        out_df = pl.DataFrame(
            {"ticker": symbols, **{c: emb[:, i] for i, c in enumerate(cols)}}
        ).with_columns(pl.lit(day).alias("day"))

        out_df.write_parquet(DST_DIR / f"{day}.parquet")
        log.debug("Snapshot %s written (%d tickers)", day, len(symbols))


if __name__ == "__main__":
    main()
