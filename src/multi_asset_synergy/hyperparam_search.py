import logging
import optuna
import numpy as np
from .gnn_synergy import build_correlation_matrix, build_gnn_data, train_gnn_embeddings

logger = logging.getLogger(__name__)


def synergy_objective(trial, dfs_dict, symbols, base_cfg):
    """
    objective: Minimizes average L2 norm of final embeddings.
    """
    corr_thresh = trial.suggest_float("correlation_threshold", 0.05, 0.9)
    hidden_dim = trial.suggest_int("hidden_dim", 8, 64, log=True)

    col = base_cfg["correlation_col"]
    window = base_cfg["correlation_window"]
    _, corr_mat = build_correlation_matrix(dfs_dict, col=col, window=window)

    data = build_gnn_data(symbols, corr_mat, threshold=corr_thresh)
    emb = train_gnn_embeddings(
        data,
        hidden_dim=hidden_dim,
        num_layers=base_cfg["num_layers"],
        epochs=base_cfg["epochs"],
        lr=base_cfg["lr"],
        synergy_loss=base_cfg["synergy_loss"]
    )
    if emb.size == 0:
        return 9999.0

    # average L2 norm obj
    avg_norm = float(np.mean(np.linalg.norm(emb, axis=1)))
    logger.info(f"Trial {trial.number} => corr_thresh={corr_thresh}, hidden_dim={hidden_dim}, avg_norm={avg_norm:.4f}")
    return avg_norm


def run_synergy_hyperparam_search(dfs_dict, config):
    """
    Illustrates synergy hyperparam search with Optuna. 
    We assume dfs_dict is sub-DFs keyed by ticker,
    and config["multi_asset"]["gnn"] is the base GNN config.
    """
    gnn_cfg = config["multi_asset"]["gnn"]
    symbols = sorted(dfs_dict.keys())

    study = optuna.create_study(direction="minimize")

    def objective(trial):
        return synergy_objective(trial, dfs_dict, symbols, gnn_cfg)

    n_trials = gnn_cfg.get("hp_search_trials", 10)
    study.optimize(objective, n_trials=n_trials)

    return study.best_params
