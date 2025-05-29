"""
Attention forecaster; HPO and final training.

Details
--------
Loads the fully enriched minute dataset, runs an Optuna search over a
row-wise Transformer, then trains the best model on (train+val) and
evaluates on the chronologically held out test set.

Configuration (configs/attention.yaml)
--------------------------------
paths.*                   : artefact locations
hpo.n_trials / timeout    : Optuna budget
attention.*               : default model/batch/epochs

Outputs
-------
- `paths.best_params`      (best hyper-parameters, YAML)
- `paths.optuna_trials`    (Optuna study, CSV)
- `paths.checkpoint`       (trained model, *.pt*)
- `paths.reports_dir`/train_loss.png
- `paths.reports_dir`/time_series.png
- `paths.reports_dir`/scatter.png
- `paths.reports_dir`/error_hist.png

Public API
----------
- main : command-line entry-point used by `make attention [--debug]`
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import yaml
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from .data import MinuteDataset, build_dataset, chrono_split
from .model import Predictor
from .objective import DEVICE, objective

# --------------------------------------------------------------------------- #
# Configuration & global constants
# --------------------------------------------------------------------------- #

_CFG_PATH = Path("configs/attention.yaml")
_CFG = yaml.safe_load(_CFG_PATH.read_text())
_PATHS = _CFG["paths"]
HPO_TRIALS = int(_CFG["hpo"]["n_trials"])
HPO_TIMEOUT = int(_CFG["hpo"]["timeout"])
FINAL_BATCH = int(_CFG["final_train"]["batch"])
OPTUNA_TRIALS = _PATHS["optuna_trials"]
EPOCHS = int(_CFG["final_train"]["epochs"])
EARLY_STOP = int(_CFG["attention"]["early_stopping_patience"])
BEST_PARAMS = Path(_PATHS["best_params"])

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
# Core function
# --------------------------------------------------------------------------- #

def train_and_evaluate(debug: bool = False) -> None:  # noqa: D401
    if debug:
        log.setLevel(logging.DEBUG)

    df, feats, tgt = build_dataset()
    tr_df, va_df, te_df = chrono_split(df)
    log.info(
        "Split -> train:%d  val:%d  test:%d",
        len(tr_df),
        len(va_df),
        len(te_df)
        )

    tr_ds = MinuteDataset(tr_df, feats, tgt)
    va_ds = MinuteDataset(va_df, feats, tgt)
    te_ds = MinuteDataset(te_df, feats, tgt)

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
    )
    study.optimize(
        lambda t: objective(t, tr_ds, va_ds, len(feats)),
        n_trials=HPO_TRIALS,
        timeout=HPO_TIMEOUT,
    )

    best: dict[str, Any] = study.best_params
    BEST_PARAMS.write_text(yaml.safe_dump(best))
    pd.DataFrame(
        [
            {"trial": t.number, **t.params, "loss": t.value}
            for t in study.trials
            ]
    ).to_csv(OPTUNA_TRIALS, index=False)
    log.info("Best HPO params: %s", best)

    full_df = pd.concat([tr_df, va_df], ignore_index=True)
    loader = DataLoader(
        MinuteDataset(
            full_df,
            feats,
            tgt
            ),
        batch_size=FINAL_BATCH, shuffle=True)

    model = Predictor(
        len(feats),
        best["d_model"],
        best["num_heads"],
        best["num_layers"],
        best["ff_dim"],
        best["dropout"],
    ).to(DEVICE)
    opti = torch.optim.AdamW(model.parameters(), lr=best["lr"])
    loss_fn = nn.MSELoss()

    patience = EARLY_STOP
    bad, best_val, losses = 0, float("inf"), []
    va_loader = DataLoader(va_ds, batch_size=512)

    for epoch in range(EPOCHS):
        model.train()
        ep = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opti.zero_grad()
            loss_ = loss_fn(model(xb), yb)
            loss_.backward()
            opti.step()
            ep += loss_.item()
        ep /= len(loader)
        losses.append(ep)

        # val check
        model.eval()
        v = 0.0
        with torch.no_grad():
            for xb, yb in va_loader:
                v += loss_fn(model(xb.to(DEVICE)), yb.to(DEVICE)).item()
        v /= len(va_loader)
        log.info("Epoch %d  train=%.6f  val=%.6f", epoch + 1, ep, v)
        if v < best_val:
            best_val, bad = v, 0
        else:
            bad += 1  
        if bad >= patience:
            log.info("Early stop")
            break

    ckpt = Path(_PATHS["checkpoint"])
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt)
    log.info("Checkpoint -> %s", ckpt)

    # Validate
    te_loader = DataLoader(te_ds, batch_size=512)
    preds, trues = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in te_loader:
            preds.extend(model(xb.to(DEVICE)).cpu().numpy().ravel())
            trues.extend(yb.ravel())
    rmse = float(np.sqrt(((np.asarray(preds) - np.asarray(trues)) ** 2).mean()))
    log.info("Test RMSE = %.6f", rmse)

    # Plots
    plt.figure(dpi=110)
    plt.plot(losses)
    plt.title("Train loss")
    plt.savefig(REPORT_DIR / "train_loss.png")
    
    plt.figure(dpi=110)
    plt.plot(trues[:5000], label="true")
    plt.plot(preds[:5000], label="pred")
    plt.legend()
    plt.title("First 5k")
    plt.savefig(REPORT_DIR / "time_series.png")
    plt.figure(dpi=110)
    
    lims = [min(trues), max(trues)]
    plt.scatter(trues, preds, s=2, alpha=0.6)
    plt.plot(lims, lims, "r--")
    plt.title("Scatter")
    plt.savefig(REPORT_DIR / "scatter.png")
    
    plt.figure(dpi=110)
    plt.hist(np.asarray(preds) - np.asarray(trues), bins=60)
    plt.title("Error hist")
    plt.savefig(REPORT_DIR / "error_hist.png")

# --------------------------------------------------------------------------- #
# CLI entrypoint
# --------------------------------------------------------------------------- #

def main(argv: Sequence[str] | None = None) -> None:
    """Entrypoint for attention train."""
    parser = argparse.ArgumentParser("Attention training")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="DEBUG logs + progress + incremental loss"
        )
    args = parser.parse_args(argv)
    try:
        train_and_evaluate(debug=args.debug)
    except Exception:  # noqa: BLE001
        log.exception("Fatal error")
        return


if __name__ == "__main__":
    main()
