"""
Attention model pipeline.

Orchestrate the attention model pipeline by building the fully enriched
dataset, splitting into training/val/test sets running a transformer model,
perform a optuna hyperparam search and saving the model.

Public API
----------
run_merge   : Dataset builder for the attention forecaster.
run_train   : Train the transformer forecaster (HPO + final fit).
"""

from __future__ import annotations

from .data import main as run_merge
from .train import main as run_train

__all__: list[str] = [
    "run_merge",
    "run_train",
]
