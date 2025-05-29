"""
Hyper-parameter search for the GNN correlation model.

Details
--------
Loads HMM labelled minute data, builds a centered residual correlation
matrix, and runs an Optuna study to minimize the cosine target loss of a
GCN model. The best embeddings are logged alongside trial reports.

Configuration (configs/gnn.yaml)
--------------------------------
paths.hmm_out           : data/interim/hmm.parquet
paths.synergy_out       : data/interim/synergy.parquet
paths.reports_dir       : reports/synergy
gnn.correlation_window  : 50
gnn.epochs              : 100
gnn.device              : cpu
gnn.benches             : - SPY - VXX - QQQ

Outputs
-------
- `paths.synergy_out` (synergy.parquet)
- `paths.reports_dir`/corr_matrix_centered.csv
- `paths.reports_dir`/optuna.csv
- `paths.reports_dir`/best_params.yaml

Public API
----------
- main : command-line entry-point used by `make gnn [--trials 30 --device cpu --debug]`
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import optuna
import polars as pl
import torch
import yaml
from optuna.trial import Trial

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

SRC_PARQ = Path(_PATHS["hmm_out"])
DST_PARQ = Path(_PATHS["synergy_out"])
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

def _prepare_training_data():
    if not SRC_PARQ.exists():
        log.error("HMM parquet %s missing - run hmm_fit first", SRC_PARQ)
        return

    df = pl.read_parquet(SRC_PARQ)
    dfs = {
        str(k[0]) if isinstance(k, tuple) else str(k): grp
        for k, grp in df.group_by("ticker", maintain_order=True)
        }
    
    symbols, corr = build_corr(dfs, BENCHES, WINDOW)
    corr_c = corr - corr.mean()
    stats = (
        df.select(["ticker", "vov", "iqv", "trade_intensity"])
        .group_by("ticker", maintain_order=True)
        .mean()
        .filter(pl.col("ticker").is_in(symbols))
        .sort("ticker")
    )
    return symbols, corr_c, stats

def _dump_corr_centered(
    symbols: Sequence[str],
    corr_c: np.ndarray
    ) -> None:
    pl.DataFrame(
        {"ticker": symbols, **{sym: corr_c[:, i] for i, sym in enumerate(symbols)}}
    ).write_csv(REPORT_DIR / "corr_matrix_centered.csv")
    log.info("Saved centred correlation matrix")

# --------------------------------------------------------------------------- #
# Objective logic
# --------------------------------------------------------------------------- #

def _objective_factory(
    symbols: Sequence[str],
    corr_c: np.ndarray,
    stats: pl.DataFrame,
    device: str,
    debug: bool,
    ):
    def _objective(trial: Trial) -> float:
        thr = trial.suggest_float("thr", 0.05, 0.5, log=True)
        hdim = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256])
        layers = trial.suggest_int("layers", 2, 4)
        lr = trial.suggest_float("lr", 1e-4, 2e-3, log=True)

        data = gnn_data(symbols, corr_c, stats, thr)
        emb, loss = train_once(
            data,
            hidden_dim=hdim,
            layers=layers,
            lr=lr,
            epochs=EPOCHS,
            device=device,
        )
        if debug:
            log.debug(
                "Trial %d -> thr=%.3f hdim=%d layers=%d lr=%.1e loss=%.4f",
                trial.number,
                thr,
                hdim,
                layers,
                lr,
                loss,
            )
        trial.set_user_attr("emb", emb)
        return loss
    
    return _objective

# --------------------------------------------------------------------------- #
# CLI entrypoint
# --------------------------------------------------------------------------- #

def main(argv: Sequence[str] | None = None) -> None:
    """Entrypoint for GNN hyperparameter search via Optuna."""
    parser = argparse.ArgumentParser("GNN Optuna search")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="DEBUG logs + progress + incremental loss"
        )
    args = parser.parse_args(argv)
    
    if args.debug:
        log.setLevel(logging.DEBUG)

    device = "cpu"
    if args.device.startswith("cuda") and torch.cuda.is_available():
        device = "cuda"

    if args.device.startswith("cuda") and device == "cpu":
        log.warning("CUDA unavailable - falling back to CPU")

    symbols, corr_c, stats = _prepare_training_data()
    if symbols is None:
        log.error("symbols is None")
        return
        
    _dump_corr_centered(symbols, corr_c)

    objective = _objective_factory(symbols, corr_c, stats, device, args.debug)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.trials)
    
    # Log relevant data
    best = study.best_trial
    log.info("Best params: %s  loss=%.4e", best.params, best.value)

    emb = best.user_attrs["emb"]
    cols = [f"gnn_emb_{i}" for i in range(emb.shape[1])]
    out_df = pl.DataFrame(
        {"ticker": symbols, **{col: emb[:, i] for i, col in enumerate(cols)}}
        )
    DST_PARQ.parent.mkdir(parents=True, exist_ok=True)
    out_df.write_parquet(DST_PARQ)
    log.info("Embeddings written -> %s", DST_PARQ)

    trials_df = pl.DataFrame(
        [{"trial": t.number, **t.params, "loss": t.value} for t in study.trials]
        )
    trials_df.write_csv(REPORT_DIR / "optuna.csv")
    with open(REPORT_DIR / "best_params.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(best.params, fh)
        
    log.info("Reports written -> %s", REPORT_DIR)


if __name__ == "__main__":
    main()
