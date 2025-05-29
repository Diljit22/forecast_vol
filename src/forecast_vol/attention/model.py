"""Transformer style regression head used by the attention forecaster.

Details
--------
Implements a row-wise Transformer (no positional encoding)
with N repeated `AttentionBlock`s followed by a single linear head.
The network consumes a 2D feature vector or a [B, 1, F] tensor; in the
former case a singleton sequence dimension is added automatically.

Public API
----------
Predictor : `nn.Module` ready for training or inference.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# --------------------------------------------------------------------------- #
# Core functions
# --------------------------------------------------------------------------- #

class _FeedForward(nn.Module):
    """Position-wise feed-forward with residual + LayerNorm."""

    def __init__(self, d_model: int, ff_dim: int, dropout: float) -> None:  # noqa: D401
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.norm(x + self.net(x))


class _AttentionBlock(nn.Module):
    """Single Transformer encoder block (no positional encoding)."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_dim: int,
        dropout: float
        ) -> None:  # noqa: D401
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.ff = _FeedForward(d_model, ff_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm(x + attn_out)
        return self.ff(x)


class Predictor(nn.Module):
    """Row-wise Transformer regressor."""

    def __init__(
        self,
        inp_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        ff_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(inp_dim, d_model)
        self.blocks = nn.Sequential(
            *[_AttentionBlock(
                d_model,
                n_heads,
                ff_dim,
                dropout
                ) for _ in range(n_layers)
              ]
        )
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:  # [B, F] -> [B, 1, F]
            x = x.unsqueeze(1)
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.head(x[:, -1])