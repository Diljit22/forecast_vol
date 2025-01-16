import logging
import optuna
import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, Any
import time
from src.core_ml.models.deep_vol_models import build_model
from src.core_ml.models.wavelets import fit_rfsv_params

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

_GLOBAL_DEEP_DATA = None
_GLOBAL_STOCH_DATA = None


def load_deep_datasets_once(csv_dir: str, sequence_length=30):
    """
    Loads train/tune CSVs *only once* and caches them globally.
    Creates (X_train, y_train), (X_tune, y_tune).
    If there's not enough data, returns empty arrays => you'll see an error.
    """
    global _GLOBAL_DEEP_DATA
    if _GLOBAL_DEEP_DATA is not None:
        return _GLOBAL_DEEP_DATA

    train_path = f"{csv_dir}/train.csv"
    tune_path = f"{csv_dir}/tune.csv"

    # If you know the exact format, specify date_format. Otherwise, remove date_format and parse as needed.
    df_train = pd.read_csv(
        train_path, parse_dates=["t"], date_format="%Y-%m-%d %H:%M:%S"
    )
    df_tune = pd.read_csv(tune_path, parse_dates=["t"], date_format="%Y-%m-%d %H:%M:%S")

    feature_cols = [
        "t",
        "volume",
        "vwap",
        "open",
        "close",
        "high",
        "low",
        "trades_count",
        "ticker_id",
        "is_etf",
        "active_yday",
        "active_tom",
        "time_since_open",
        "time_until_close",
        "log_return",
        "park_vol",
        "rv",
        "abs_return",
        "bpv",
        "gk",
        "gk_vol",
        "iqv",
        "order_imbalance",
        "vov",
    ]

    df_train = df_train.dropna(subset=feature_cols)
    df_tune = df_tune.dropna(subset=feature_cols)

    # 'rv' is target
    df_train["target"] = df_train["rv"].shift(-1)
    df_train.dropna(subset=["target"], inplace=True)

    df_tune["target"] = df_tune["rv"].shift(-1)
    df_tune.dropna(subset=["target"], inplace=True)

    X_train_list, y_train_list = [], []
    for i in range(len(df_train) - sequence_length):
        feat_block = df_train[feature_cols].iloc[i : i + sequence_length].values
        target_val = df_train["target"].iloc[i + sequence_length - 1]
        X_train_list.append(feat_block)
        y_train_list.append([target_val])

    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.float32)

    X_tune_list, y_tune_list = [], []
    for i in range(len(df_tune) - sequence_length):
        feat_block = df_tune[feature_cols].iloc[i : i + sequence_length].values
        target_val = df_tune["target"].iloc[i + sequence_length - 1]
        X_tune_list.append(feat_block)
        y_tune_list.append([target_val])

    X_tune = np.array(X_tune_list, dtype=np.float32)
    y_tune = np.array(y_tune_list, dtype=np.float32)

    logger.info(f"[DEBUG] X_train.shape={X_train.shape}, y_train.shape={y_train.shape}")
    logger.info(f"[DEBUG] X_tune.shape={X_tune.shape}, y_tune.shape={y_tune.shape}")
    if X_train.shape[0] == 0 or y_train.shape[0] == 0:
        logger.error(
            "[DEEP] No training data after rolling window. Check CSV or sequence_length."
        )
    if X_tune.shape[0] == 0 or y_tune.shape[0] == 0:
        logger.error(
            "[DEEP] No tuning data after rolling window. Check CSV or sequence_length."
        )

    _GLOBAL_DEEP_DATA = ((X_train, y_train), (X_tune, y_tune))
    return _GLOBAL_DEEP_DATA


def load_stoch_data_once(csv_dir: str):
    """
    Loads train/tune CSVs *only once* for RFSV usage.
    Returns (rv_train, rv_tune).
    """
    global _GLOBAL_STOCH_DATA
    if _GLOBAL_STOCH_DATA is not None:
        return _GLOBAL_STOCH_DATA

    train_path = f"{csv_dir}/train.csv"
    tune_path = f"{csv_dir}/tune.csv"

    df_train = pd.read_csv(
        train_path, parse_dates=["t"], date_format="%Y-%m-%d %H:%M:%S"
    )
    df_tune = pd.read_csv(tune_path, parse_dates=["t"], date_format="%Y-%m-%d %H:%M:%S")

    df_train.dropna(subset=["rv"], inplace=True)
    df_tune.dropna(subset=["rv"], inplace=True)

    rv_train = df_train["rv"].values.astype(np.float32)
    rv_tune = df_tune["rv"].values.astype(np.float32)

    logger.info(
        f"[DEBUG] rv_train length={len(rv_train)}, rv_tune length={len(rv_tune)}"
    )

    if len(rv_train) == 0:
        logger.error(
            "[STOCH] No training data for RFSV. Check train.csv or 'rv' column."
        )
    if len(rv_tune) == 0:
        logger.error("[STOCH] No tuning data for RFSV. Check tune.csv or 'rv' column.")

    _GLOBAL_STOCH_DATA = (rv_train, rv_tune)
    return _GLOBAL_STOCH_DATA


# Tuning obj.'s


def objective_deep(
    trial: optuna.trial.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_tune: np.ndarray,
    y_tune: np.ndarray,
    config: Dict[str, Any],
) -> float:

    # Check if data is non-empty
    if len(y_train.shape) < 2 or y_train.shape[0] == 0:
        raise ValueError(f"[DEEP] y_train empty or not 2D: shape={y_train.shape}.")

    hp_cfg = config["hyperparameter_search"]["deep_space"]
    d_model_candidates = hp_cfg.get("d_model", [8, 16, 32])
    d_model = trial.suggest_categorical("d_model", d_model_candidates)

    lr_range = hp_cfg.get("learning_rate_log_range", [1e-4, 1e-2])
    learning_rate = trial.suggest_float(
        "learning_rate", lr_range[0], lr_range[1], log=True
    )
    epochs = hp_cfg.get("epochs", 3)

    input_dim = X_train.shape[-1]
    n_tasks = y_train.shape[1]

    local_config = config.copy()
    local_config["deep_learning"] = local_config.get("deep_learning", {}).copy()
    local_config["deep_learning"]["d_model"] = d_model
    local_config["deep_learning"]["learning_rate"] = learning_rate
    local_config["deep_learning"]["input_dim"] = input_dim
    local_config["deep_learning"]["n_tasks"] = n_tasks
    local_config["deep_learning"]["epochs"] = epochs

    model = build_model(local_config["deep_learning"])

    batch_size = local_config["deep_learning"].get("batch_size", 32)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    device = torch.device("cpu")
    model.to(device)
    model.train()

    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    for epoch_i in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_epoch_loss = (
            epoch_loss / len(train_loader) if len(train_loader) > 0 else 9999.0
        )
        logger.info(
            f"[DEEP] Trial {trial.number}, Epoch {epoch_i+1}/{epochs}, "
            f"LR={learning_rate:.5f}, d_model={d_model}, loss={avg_epoch_loss:.4f}"
        )

    # Evaluate on tune set
    model.eval()
    tune_dataset = TensorDataset(torch.from_numpy(X_tune), torch.from_numpy(y_tune))
    tune_loader = DataLoader(
        tune_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    total_tune_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in tune_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            total_tune_loss += loss.item()

    avg_tune_loss = (
        total_tune_loss / len(tune_loader) if len(tune_loader) > 0 else 9999.0
    )
    logger.info(
        f"[DEEP] Trial {trial.number} finished => tune_loss={avg_tune_loss:.4f}"
    )
    return avg_tune_loss


def objective_stoch(
    trial: optuna.trial.Trial,
    rv_train: np.ndarray,
    rv_tune: np.ndarray,
    config: Dict[str, Any],
) -> float:

    stoch_cfg = config["hyperparameter_search"]["stoch_space"]
    wavelet_levels_choices = stoch_cfg.get("wavelet_levels_choices", [3, 4, 5])
    wavelet_levels = trial.suggest_categorical("wavelet_levels", wavelet_levels_choices)

    random_restarts_choices = stoch_cfg.get("random_restarts", [1, 2])
    random_restarts = trial.suggest_categorical(
        "random_restarts", random_restarts_choices
    )

    if len(rv_train) < 2:
        raise ValueError("[STOCH] rv_train has insufficient data.")
    if len(rv_tune) < 2:
        raise ValueError("[STOCH] rv_tune has insufficient data.")

    # Fit on rv_train
    params_train = fit_rfsv_params(
        rv_train, wavelet="haar", levels=wavelet_levels, random_restarts=random_restarts
    )
    sigma_train = params_train["sigma"]

    # Fit on rv_tune
    params_tune = fit_rfsv_params(
        rv_tune, wavelet="haar", levels=wavelet_levels, random_restarts=random_restarts
    )
    sigma_tune = params_tune["sigma"]

    metric = sigma_train + sigma_tune
    logger.info(
        f"[STOCH] Trial {trial.number} => wavelet_levels={wavelet_levels}, "
        f"random_restarts={random_restarts}, metric={metric:.4f}"
    )
    return metric


def run_hyperparameter_search(
    config: Dict[str, Any], csv_dir: str, method: str = "deep"
):
    """
    Multithreaded hyperparameter search with frequent logging.
    - n_jobs=-1 => use all CPU cores
    - timeout=600 => (optional) stop after 10 minutes total
    """
    hp_cfg = config.get("hyperparameter_search", {})
    if not hp_cfg.get("enabled", False):
        logger.info("Hyperparameter search not enabled. Skipping.")
        return None

    n_trials = hp_cfg.get("n_trials", 10)
    pruner_type = hp_cfg.get("pruner", "none")
    if pruner_type == "median":
        pruner = optuna.pruners.MedianPruner()
    elif pruner_type == "successive_halving":
        pruner = optuna.pruners.SuccessiveHalvingPruner()
    else:
        pruner = optuna.pruners.NopPruner()

    study = optuna.create_study(direction="minimize", pruner=pruner)
    time_limit_seconds = 600

    start_time = time.time()
    logger.info(
        f"Starting hyperparam search (method={method}), up to {n_trials} trials, "
        f"parallel jobs=-1, timeout={time_limit_seconds}s..."
    )

    if method == "deep":
        (X_train, y_train), (X_tune, y_tune) = load_deep_datasets_once(csv_dir)

        def objective(trial):
            return objective_deep(trial, X_train, y_train, X_tune, y_tune, config)

        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=-1,
            timeout=time_limit_seconds,
            show_progress_bar=True,
        )

    elif method == "stochastic":
        rv_train, rv_tune = load_stoch_data_once(csv_dir)

        def objective(trial):
            return objective_stoch(trial, rv_train, rv_tune, config)

        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=-1,
            timeout=time_limit_seconds,
            show_progress_bar=True,
        )

    elif method == "both":
        # 1) Deep
        (X_train, y_train), (X_tune, y_tune) = load_deep_datasets_once(csv_dir)

        def objective_1(trial):
            return objective_deep(trial, X_train, y_train, X_tune, y_tune, config)

        study.optimize(
            objective_1,
            n_trials=n_trials,
            n_jobs=-1,
            timeout=time_limit_seconds,
            show_progress_bar=True,
        )

        # 2) Stochastic
        rv_train, rv_tune = load_stoch_data_once(csv_dir)
        study_stoch = optuna.create_study(direction="minimize", pruner=pruner)

        def objective_2(trial):
            return objective_stoch(trial, rv_train, rv_tune, config)

        study_stoch.optimize(
            objective_2,
            n_trials=n_trials,
            n_jobs=-1,
            timeout=time_limit_seconds,
            show_progress_bar=True,
        )

    elapsed = time.time() - start_time
    logger.info(f"Hyperparam search ended. Elapsed={elapsed:.1f}s")

    # If no successful trials, .best_trial may fail
    if len(study.trials) == 0 or all(
        t.state.is_finished() is False for t in study.trials
    ):
        logger.error(
            "[HYPERPARAM] No completed trials. Possibly empty dataset or error."
        )
        return None

    best_params = study.best_trial.params
    logger.info(f"[HYPERPARAM] Best trial => {best_params}")
    return best_params
