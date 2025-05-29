"""
GNN model and helper functions for embedding training.

Details
--------
Data preparation for torch_geometric and a GCN-based network architecture
with a training routine.

Public API
----------
- gnn_data   : Convert correlation + stats to a torch_geometric Data object
- GNNNet     : GCN with configurable hidden dimension and depth
- train_once : Train GNN for a fixed number of epochs on a cosine-target loss
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F  #noqa N812
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def gnn_data(
    symbols: Sequence[str],
    corr: np.ndarray,
    stats_df: pl.DataFrame,
    thr: float,
) -> Data:
    """
    Construct a torch_geometric Data object from correlation and stats.

    Parameters
    ----------
    symbols
        Ordered list of tickers.
    corr
        (n, n) correlation matrix.
    stats_df
        DataFrame with mean stats per ticker (must include vov, iqv, trade_intensity).
    thr
        Threshold for edge creation on absolute correlation.

    Returns
    -------
    Data
        Graph data with x = [corr|stats], edge_index, and corr target.
    """
    n = len(symbols)
    edges = [
        [i, j]
        for i in range(n)
        for j in range(n)
        if i != j and abs(corr[i, j]) >= thr
    ]
    if not edges:
        edges = [[i, i] for i in range(n)]

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    x_corr = torch.from_numpy(corr).float()
    x_stats = torch.from_numpy(
        stats_df.select(["vov", "iqv", "trade_intensity"])
        .to_numpy()
        .astype(np.float32, copy=False)
    )
    x = torch.cat([x_corr, x_stats], dim=1)
    
    return Data(x=x, edge_index=edge_index, corr=torch.from_numpy(corr).float())

# --------------------------------------------------------------------------- #
# Model definition
# --------------------------------------------------------------------------- #

class GNNNet(torch.nn.Module):
    """
    GCN with configurable hidden dimension and depth.

    Public API
    ----------
    - forward : compute node embeddings
    """

    def __init__(self, input_dim: int, hidden_dim: int, layers: int) -> None:
        super().__init__()
        self.convs = torch.nn.ModuleList(
            [GCNConv(input_dim, hidden_dim)]
            + [GCNConv(hidden_dim, hidden_dim) for _ in range(layers - 1)]
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Compute node embeddings."""
        for conv in self.convs:
            x = torch.relu(conv(x, edge_index))
        return x  # (n, hidden_dim)

# --------------------------------------------------------------------------- #
# Training routine
# --------------------------------------------------------------------------- #

def train_once(
    data: Data,
    hidden_dim: int,
    layers: int,
    lr: float,
    epochs: int,
    device: str,
) -> tuple[np.ndarray, float]:
    """
    Train GNN for a fixed number of epochs on a cosine-target loss.

    Parameters
    ----------
    data
        torch_geometric Data object.
    hidden_dim
        Embedding dimension.
    layers
        Number of GCN layers.
    lr
        Learning rate.
    epochs
        Training epochs.
    device
        'cpu' or 'cuda'.

    Returns
    -------
    emb
        (n, hidden_dim) learned embeddings.
    loss
        Final training loss.
    """
    net = GNNNet(data.num_node_features, hidden_dim, layers).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    data = data.to(device)

    for _ in range(epochs):
        net.train()
        opt.zero_grad()
        z = net(data.x, data.edge_index)
        z_u = F.normalize(z, p=2, dim=1)
        loss = F.mse_loss(z_u @ z_u.T, data.corr)
        loss.backward() # type: ignore[no-untyped-call]
        opt.step()

    net.eval()
    with torch.no_grad():
        emb = net(data.x, data.edge_index).cpu().numpy()

    return emb.astype(np.float32), float(loss.cpu())
