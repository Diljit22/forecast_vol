"""
Implements:
  - FeedForward: a 2-layer MLP sub-layer used in each AttentionBlock
  - AttentionBlock: multi-head self-attention + residual + layernorm + feed-forward
  - AttentionPredictor: stacks multiple AttentionBlocks, plus final linear output.
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class FeedForward(nn.Module):
    """
    A 2-layer feed-forward sub-layer used in each AttentionBlock.
    """
    def __init__(self, d_model: int, ff_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, ff_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ff_dim, d_model)
        self.layernorm = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, d_model]
        hidden = self.linear1(x)
        hidden = self.relu(hidden)
        hidden = self.dropout(hidden)
        hidden = self.linear2(hidden)
        # residual + layernorm
        out = self.layernorm(x + self.dropout2(hidden))
        return out


class AttentionBlock(nn.Module):
    """
    One transformer-style block:
      - MultiHeadAttention
      - Residual + LayerNorm
      - FeedForward sub-layer
    """
    def __init__(
        self,
        d_model: int = 32,
        num_heads: int = 4,
        ff_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )
        self.linear = nn.Linear(d_model, d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.ffn = FeedForward(d_model, ff_dim=ff_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x => [B, T, d_model]
        attn_output, _ = self.attn(x, x, x)
        attn_output = self.linear(attn_output)
        out = self.ln1(x + self.dropout(attn_output))
        # feed-forward
        out = self.ffn(out)
        return out


class AttentionPredictor(nn.Module):
    """
    Full multi-block attention model:
      repeated AttentionBlock
      final linear => n_tasks
      output from last time step
    """
    def __init__(
        self,
        input_dim: int = 8,
        d_model: int = 32,
        num_heads: int = 4,
        num_layers: int = 1,
        ff_dim: int = 128,
        dropout: float = 0.1,
        n_tasks: int = 1
    ):
        super().__init__()

        if d_model != input_dim:
            self.input_proj = nn.Linear(input_dim, d_model)
        else:
            self.input_proj = nn.Identity()

        self.blocks = nn.ModuleList([
            AttentionBlock(d_model=d_model, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.out_linear = nn.Linear(d_model, n_tasks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x => [B, T, input_dim]
        x = self.input_proj(x)  # => [B, T, d_model]
        for block in self.blocks:
            x = block(x)  # => [B, T, d_model]

        # final step => last time
        last_step = x[:, -1, :]  # => [B, d_model]
        out = self.out_linear(last_step)  # => [B, n_tasks]
        return out

