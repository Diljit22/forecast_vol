import logging
import time
import torch
import torch.nn as nn
from typing import Dict, Any
from torch.utils.data import DataLoader

from .build_model import build_attention_model

logger = logging.getLogger(__name__)



def train_attention_model(
    config: Dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader = None,
    device: str = "cpu"
) -> nn.Module:
    """
    Train attention model for 'epochs' and return it.
    """
    attn_cfg = config["deep_learning"]["attention"]
    if not attn_cfg.get("enabled", True):
        logger.info("[train_attention_model] 'attention' not enabled. Returning None.")
        return None

    model = build_attention_model(attn_cfg).to(device)
    epochs = attn_cfg.get("epochs", 5)
    lr = attn_cfg.get("learning_rate", 1e-3)
    checkpoint_path = attn_cfg.get("checkpoint_path", None)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    logger.info("[train_attention_model] Starting training loop ...")
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for bx, by in train_loader:
            bx = bx.to(device)
            by = by.to(device)

            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        logger.info(f"[Epoch {epoch+1}/{epochs}] train_loss={avg_loss:.5f}")

        if val_loader is not None:
            val_loss = 0.0
            model.eval()
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx = vx.to(device)
                    vy = vy.to(device)
                    preds_val = model(vx)
                    loss_val = criterion(preds_val, vy)
                    val_loss += loss_val.item()
            val_loss /= len(val_loader)
            logger.info(f"[Epoch {epoch+1}/{epochs}] val_loss={val_loss:.5f}")

    if checkpoint_path:
        torch.save(model.state_dict(), checkpoint_path)
        logger.info(f"[train_attention_model] Saved model to {checkpoint_path}")

    logger.info(f"[train_attention_model] Finished training in {time.time() - t0:.1f}s")
    return model