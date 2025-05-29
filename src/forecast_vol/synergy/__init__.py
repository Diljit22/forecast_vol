"""
Regime labelling + GNN correlation model pipeline.

Label regimes with per-ticker HMMs and tune a GNN that
models cross-asset dependencies via Optuna.

Public API
----------
run_hmm_fit       : Fit a Hidden Markov Model per ticker
run_optuna_search : Hyper-parameter search for the GNN
"""

from __future__ import annotations

from .gnn_snapshots import main as run_snapshots
from .hmm_fit import main as run_hmm_fit
from .optuna_search import main as run_optuna_search

__all__: list[str] = [
    "run_hmm_fit",
    "run_optuna_search",
    "run_snapshots",
    ]
