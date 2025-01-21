
import logging
import optuna
import numpy as np
from .rfsv_model import fit_rfsv_params
from .garch_model import fit_garch_params

logger = logging.getLogger(__name__)


def stoch_objective(trial, rv_series):
    """
    measure sum of RFSV sigma + GARCH alpha as a silly metric to minimize.
    """
    wavelet_levels = trial.suggest_int("wavelet_levels", 3, 8)
    rfsv_out = fit_rfsv_params(rv_series, levels=wavelet_levels)
    sigma_rfsv = rfsv_out["sigma"]

    garch_out = fit_garch_params(rv_series)
    alpha = garch_out["alpha"]

    metric = sigma_rfsv + alpha
    logger.info(f"Trial {trial.number}: wavelet_levels={wavelet_levels}, sigma={sigma_rfsv:.4f}, alpha={alpha:.4f}, metric={metric:.4f}")
    return metric


def run_stoch_hyperparam_search(rv_series, config):
    """
    Runs a hyperparam search
    """
    stoch_cfg = config["stochastic"]
    n_trials = stoch_cfg.get("hp_search", {}).get("n_trials", 10)

    study = optuna.create_study(direction="minimize")
    def objective(trial):
        return stoch_objective(trial, rv_series)

    study.optimize(objective, n_trials=n_trials)
    return study.best_params
