
import logging
import time
import optuna
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from typing import Dict, Any

logger = logging.getLogger(__name__)


def objective_attention_in_memory(
    trial: optuna.Trial,
    train_ds,
    val_ds,
    base_config: Dict[str, Any],
    device: str = "cpu"
) -> float:
    """
    Single objective for attention-based hyperparam search:
      - We pick hyperparams from 'trial'.
      - We train on train_ds, evaluate on val_ds
      - Return val_loss
    """
    # Hyperparam search space
    d_model_candidates = [16, 32, 64]
    d_model = trial.suggest_categorical("d_model", d_model_candidates)

    ff_dim_candidates = [64, 128, 256]
    ff_dim = trial.suggest_categorical("ff_dim", ff_dim_candidates)

    lr_min, lr_max = 1e-4, 1e-2
    learning_rate = trial.suggest_float("learning_rate", lr_min, lr_max, log=True)

    dropout = trial.suggest_float("dropout", 0.0, 0.3)
    num_layers = trial.suggest_int("num_layers", 1, 4)
    epochs = trial.suggest_int("epochs", 3, 10)

    # Build a local config with these hyperparams
    local_cfg = base_config.copy()
    local_cfg["deep_learning"] = local_cfg.get("deep_learning", {})
    local_cfg["deep_learning"]["attention"] = {
        "enabled": True,
        "d_model": d_model,
        "num_heads": 4,  # could also be tuned
        "num_layers": num_layers,
        "ff_dim": ff_dim,
        "dropout": dropout,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "n_tasks": 1,
        "input_dim": None,  # We'll infer in code
        "checkpoint_path": None
    }

    # Build DataLoaders for train/val
    dl_cfg = local_cfg.get("dataloader", {})
    batch_size = dl_cfg.get("batch_size", 32)
    num_workers = dl_cfg.get("num_workers", 4)
    pin_mem = (device == "cuda")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_mem)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_mem)

    sample_x, _ = train_ds[0]
    input_dim = sample_x.shape[-1]
    local_cfg["deep_learning"]["attention"]["input_dim"] = input_dim

    from .train_deep_model import train_attention_model

    model = train_attention_model(local_cfg, train_loader, val_loader=val_loader, device=device)

    # Evaluate final val_loss
    criterion = nn.MSELoss()
    model.eval()

    total_val_loss = 0.0
    with torch.no_grad():
        for bx, by in val_loader:
            bx = bx.to(device)
            by = by.to(device)
            preds = model(bx)
            loss_val = criterion(preds, by)
            total_val_loss += loss_val.item()

    avg_val_loss = total_val_loss / len(val_loader)
    logger.info(f"[Trial {trial.number}] val_loss={avg_val_loss:.5f}, d_model={d_model}, ff_dim={ff_dim}, lr={learning_rate:.4f}, drop={dropout:.2f}, layers={num_layers}, epochs={epochs}")
    return avg_val_loss


def run_in_memory_hpo(
    base_config: Dict[str, Any],
    train_ds,
    val_ds,
    device: str = "cpu",
    n_trials: int = 20,
    timeout: int = 600
):
    """
    Optuna search with objective_attention_in_memory.
    """
    import optuna

    logger.info("[HPO] Starting in-memory hyperparam search ...")

    def _objective(trial):
        return objective_attention_in_memory(trial, train_ds, val_ds, base_config, device=device)

    study = optuna.create_study(direction="minimize")
    start_time = time.time()
    study.optimize(_objective, n_trials=n_trials, n_jobs=1, timeout=timeout, show_progress_bar=True)
    elapsed = time.time() - start_time

    if len(study.trials) == 0:
        logger.warning("[HPO] No trials completed.")
        return None

    best_params = study.best_params
    logger.info(f"[HPO] Best params => {best_params}, time={elapsed:.1f}s")
    return best_params
