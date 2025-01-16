# deep_vol_models.py

import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any

logger = logging.getLogger(__name__)


class AttentionBlock(nn.Module):
    def __init__(self, d_model=32, num_heads=4, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, batch_first=True, dropout=dropout
        )
        self.linear = nn.Linear(d_model, d_model)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x shape: [batch_size, seq_len, d_model]
        """
        attn_output, _ = self.attention(x, x, x)
        out = self.linear(attn_output)
        out = self.layernorm(x + out)
        return out


class VolatilityPredictor_Attention(nn.Module):
    """
    Multi-task attention-based model:
        - Project input_dim -> d_model (if needed)
        - N layers of AttentionBlock
        - Final FC => n_tasks
    """

    def __init__(self, input_dim=8, d_model=8, n_tasks=2, num_layers=1, num_heads=4):
        super().__init__()
        self.d_model = d_model

        if d_model != input_dim:
            self.proj = nn.Linear(input_dim, d_model)
        else:
            self.proj = nn.Identity()

        # Stack attention blocks
        self.layers = nn.ModuleList(
            [
                AttentionBlock(d_model=d_model, num_heads=num_heads)
                for _ in range(num_layers)
            ]
        )

        self.fc = nn.Linear(d_model, n_tasks)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        x = self.proj(x)  # => [batch_size, seq_len, d_model]
        for layer in self.layers:
            x = layer(x)  # => [batch_size, seq_len, d_model]

        # Take the last time step
        last_step = x[:, -1, :]  # => [batch_size, d_model]
        out = self.fc(last_step)  # => [batch_size, n_tasks]
        return out


class VolatilityPredictor_LSTM(nn.Module):
    """
    LSTM-based approach for multi-task forecasting.
    """

    def __init__(self, input_dim=8, hidden_dim=16, n_layers=1, n_tasks=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=n_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, n_tasks)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        out, (h, c) = self.lstm(x)  # out: [batch_size, seq_len, hidden_dim]
        last_step = out[:, -1, :]
        return self.fc(last_step)


class VolatilityPredictor_CNN(nn.Module):
    """
    1D CNN approach:
        - Conv over time dimension
        - Global pool, then FC.
    """

    def __init__(self, input_dim=8, n_tasks=2, num_filters=16, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=input_dim, out_channels=num_filters, kernel_size=kernel_size
        )
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)  # global pool => [batch, num_filters, 1]
        self.fc = nn.Linear(num_filters, n_tasks)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        # Rearrange to [batch_size, input_dim, seq_len] for conv1d
        x = x.permute(0, 2, 1)  # => [batch, input_dim, seq_len]
        out = self.conv1(x)  # => [batch, num_filters, new_len]
        out = self.relu(out)
        out = self.pool(out)  # => [batch, num_filters, 1]
        out = out.squeeze(-1)  # => [batch, num_filters]
        out = self.fc(out)  # => [batch, n_tasks]
        return out


def build_model(config: Dict[str, Any]) -> nn.Module:
    dl_cfg = config.get("deep_learning", {})
    model_arch = dl_cfg.get("model_arch", "attention")

    input_dim = dl_cfg.get("input_dim", 8)
    d_model = dl_cfg.get("d_model", 8)
    n_tasks = dl_cfg.get("n_tasks", 2)
    num_layers = dl_cfg.get("num_layers", 1)
    num_heads = dl_cfg.get("num_heads", 4)
    # For CNN
    num_filters = dl_cfg.get("num_filters", 16)
    kernel_size = dl_cfg.get("kernel_size", 3)

    if model_arch == "attention":
        model = VolatilityPredictor_Attention(
            input_dim=input_dim,
            d_model=d_model,
            n_tasks=n_tasks,
            num_layers=num_layers,
            num_heads=num_heads,
        )
    elif model_arch == "lstm":
        hidden_dim = dl_cfg.get("hidden_dim", 16)
        model = VolatilityPredictor_LSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_layers=num_layers,
            n_tasks=n_tasks,
        )
    elif model_arch == "cnn":
        model = VolatilityPredictor_CNN(
            input_dim=input_dim,
            n_tasks=n_tasks,
            num_filters=num_filters,
            kernel_size=kernel_size,
        )
    else:
        raise ValueError(f"Unknown model_arch: {model_arch}")
    return model


def train_deep_model(config: Dict[str, Any], train_dataset, device="cpu"):
    """
    High-level function to:
      1) Possibly skip if cache is valid
      2) Build model
      3) Train for 'epochs' on train_dataset
      4) Save checkpoint if needed
    """
    dl_cfg = config.get("deep_learning", {})
    if not dl_cfg.get("enabled", False):
        logger.info("Deep learning not enabled. Skipping training.")
        return None

    model = build_model(config)
    epochs = dl_cfg.get("epochs", 5)
    batch_size = dl_cfg.get("batch_size", 32)
    lr = dl_cfg.get("learning_rate", 1e-3)
    checkpoint_path = dl_cfg.get("checkpoint_path", None)

    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model = model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        logger.info(f"[DeepLearning] Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.6f}")

    if checkpoint_path:
        torch.save(model.state_dict(), checkpoint_path)
        logger.info(f"Checkpoint saved at {checkpoint_path}")

    return model
