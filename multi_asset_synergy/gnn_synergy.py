
import logging
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import NormalizeFeatures
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter
from typing import Tuple
logger = logging.getLogger(__name__)


def _corr_worker(args):
    """
    Computes correlation of two numeric arrays in parallel.
    """
    i, j, base, other = args
    if base.std() == 0 or other.std() == 0:
        return (i, j, 0.0)
    corr_val = np.corrcoef(base, other)[0, 1]
    return (i, j, corr_val)


def build_correlation_matrix(
    dfs_dict: dict,
    col: str,
    window: int
) -> Tuple[list, np.ndarray]:
    """
    Builds NxN correlation matrix for final 'window' rows in dfs_dict[ticker][col].
    Parallelizes pairwise correlation computations for speed.
    """
    symbols = sorted(dfs_dict.keys())
    n = len(symbols)
    if n < 2:
        logger.warning("Less than 2 tickers; correlation matrix empty.")
        return symbols, np.array([[]], dtype=np.float32)

    # Extract last 'window' array for each ticker
    arrays = []
    for sym in symbols:
        df = dfs_dict[sym].tail(window)
        arr = df[col].values.astype(np.float32) if col in df.columns else np.zeros(window)
        arrays.append(arr)

    corr_mat = np.zeros((n, n), dtype=np.float32)
    tasks = []
    for i in range(n):
        for j in range(i, n):
            tasks.append((i, j, arrays[i], arrays[j]))

    start_t = perf_counter()
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(_corr_worker, tasks))
    for i, j, val in results:
        corr_mat[i, j] = val
        corr_mat[j, i] = val
    duration = perf_counter() - start_t
    logger.info(f"[GNN] Built correlation matrix for {n} tickers in {duration:.2f}s")

    return symbols, corr_mat


class CorrelationGNN(torch.nn.Module):
    """
    Simple GCN-based model. Each node is a ticker, edges from correlation >= threshold.
    """
    def __init__(self, num_features=1, hidden_dim=16, num_layers=2):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(num_features, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.final_layer = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x)
        x = self.final_layer(x)
        return x


def build_gnn_data(symbols, corr_mat, threshold=0.2):
    """
    Creates a PyTorch Geometric `Data` object from correlation matrix above threshold.
    """
    n = len(symbols)
    if n == 0 or corr_mat.size == 0:
        logger.warning("[GNN] Empty correlation matrix or no symbols.")
        return None

    x = torch.zeros((n, 1), dtype=torch.float)
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if abs(corr_mat[i, j]) >= threshold:
                edges.append([i, j])
                edges.append([j, i])

    if not edges:
        # Fallback: self-loop edges if no correlations pass threshold
        for i in range(n):
            edges.append([i, i])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    data = Data(x=x, edge_index=edge_index)
    data = NormalizeFeatures()(data)
    return data


def train_gnn_embeddings(data, hidden_dim=16, num_layers=2, epochs=50, lr=1e-3, synergy_loss="mse"):
    """
    Trains the GNN (self-supervised), returns node embeddings as np.ndarray.
    """
    import time
    if data is None:
        logger.warning("[GNN] Data is None. Returning empty embeddings.")
        return np.array([])

    model = CorrelationGNN(num_features=data.x.size(1), hidden_dim=hidden_dim, num_layers=num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    start_t = perf_counter()
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        if synergy_loss == "mse":
            loss = torch.mean(out**2)
        else:
            loss = torch.tensor(0.0, requires_grad=True)
        loss.backward()
        optimizer.step()

    duration = perf_counter() - start_t
    logger.info(f"[GNN] GNN training completed in {duration:.2f}s for {epochs} epochs.")

    model.eval()
    with torch.no_grad():
        emb = model(data.x, data.edge_index).cpu().numpy()
    return emb
