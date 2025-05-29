"""Optuna objective for the attention model using a MedianPruner.

Configuration (configs/attention.yaml)
--------------------------------
attention.device  : cpu

Public API
----------
objective : callable fed into `study.optimize`
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import optuna
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset

from .model import Predictor

# --------------------------------------------------------------------------- #
# Configuration & global constants
# --------------------------------------------------------------------------- #

_CFG_PATH = Path("configs/attention.yaml")
_CFG = yaml.safe_load(_CFG_PATH.read_text())
DEVICE = str(_CFG["attention"]["device"])

# --------------------------------------------------------------------------- #
# Core function
# --------------------------------------------------------------------------- #

def objective(
    trial: optuna.Trial,
    train_ds: Dataset,
    val_ds: Dataset,
    input_dim: int,
) -> float:
    """Single Optuna trial: returns validation MSE."""
    sp: dict[str, Any] = _CFG["hpo"]["search_space"]

    hp = {
        k: trial.suggest_categorical(k, sp[k]) for k in (
            "d_model", "ff_dim", "num_layers", "num_heads", "dropout"
            )
        }
    hp["epochs"] = trial.suggest_int("epochs", *sp["epochs"])
    hp["lr"] = trial.suggest_float("lr", *map(float, sp["lr_log"]), log=True)

    model = Predictor(
        input_dim,
        hp["d_model"],
        hp["num_heads"],
        hp["num_layers"],
        hp["ff_dim"],
        hp["dropout"],
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=hp["lr"])
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(
        train_ds, batch_size=256, shuffle=True, drop_last=True
        )
    val_loader = DataLoader(
        val_ds, batch_size=256, shuffle=False
        )

    for epoch in range(hp["epochs"]):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_loss += loss_fn(model(xb), yb).item()
        val_loss /= len(val_loader)

        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return val_loss