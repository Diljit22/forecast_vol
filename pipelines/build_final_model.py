import logging
import torch

from src.deep_learning_models.hyperparam_search import run_in_memory_hpo
from src.deep_learning_models.train_deep_model import train_attention_model
from src.deep_learning_models.split import pipeline_train_val_test_on_the_fly, create_data_loaders
from src.utils.cfg import init_config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def main_setup():
    # Prepare config
    config = init_config()
    seq_len = 30
    target_cols = ["rv"]

    # Load & split => train_ds, val_ds, test_ds
    logger.info("[MAIN] Loading & splitting final data ...")
    train_ds, val_ds, test_ds = pipeline_train_val_test_on_the_fly(
        config,
        seq_len=seq_len,
        target_cols=target_cols,
        time_col="t"
    )
    return config, train_ds, val_ds, test_ds

def main_launch(config, train_ds, val_ds, test_ds):
    # Run hyperparam search in memory
    device = "cpu"

    best_params = run_in_memory_hpo(
        base_config=config,
        train_ds=train_ds,
        val_ds=val_ds,
        device=device,
        n_trials=config["hyperparameter_search"].get("n_trials", 10),
        timeout=config["hyperparameter_search"].get("timeout", 600)
    )
    if best_params is None:
        logger.error("[MAIN] No best_params found. Exiting.")
        return
    logger.info(f"[MAIN] best_params from HPO => {best_params}")

    # Merge them back into config
    attn_cfg = config["deep_learning"]["attention"]
    for k, v in best_params.items():
        attn_cfg[k] = v

    attn_cfg["epochs"] = 10
    # set checkpoint path
    attn_cfg["checkpoint_path"] = r"vol\models\final_attn.pth"

    # Re-train final model on (train+val)
    logger.info("[MAIN] Building loaders for final training ...")
    dl_cfg = config["dataloader"]
    train_loader, val_loader, test_loader = create_data_loaders(train_ds, val_ds, test_ds, dl_cfg, device)

    logger.info("[MAIN] Final training with best hyperparams ...")
    final_model = train_attention_model(config, train_loader, val_loader=val_loader, device=device)

    logger.info("[MAIN] Evaluating final model on test_ds ...")
    criterion = torch.nn.MSELoss()
    final_model.eval()
    total_test_loss = 0.0
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            by = by.to(device)
            preds = final_model(bx)
            loss_val = criterion(preds, by)
            total_test_loss += loss_val.item()

    avg_test_loss = total_test_loss / len(test_loader)
    logger.info(f"[MAIN] Test MSE = {avg_test_loss:.6f}")

    logger.info("[MAIN] Done. Model saved if checkpoint_path was specified.")
