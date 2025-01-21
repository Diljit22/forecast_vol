# evaluate_model.py

import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def evaluate_attention_model(model: nn.Module, test_loader: DataLoader, device="cpu"):
    """
    Evaluate the trained model on test_loader, 
    computing MSE over the entire test set.
    """
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            total_loss += loss.item()

    mse = total_loss / len(test_loader)
    logger.info(f"[EvaluateAttention] Test MSE = {mse:.6f}")
    return mse
