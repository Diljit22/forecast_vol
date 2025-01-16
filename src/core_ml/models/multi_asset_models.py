# hnn.py

import pandas as pd
import numpy as np
import logging

from hmmlearn import hmm
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import NormalizeFeatures

logger = logging.getLogger(__name__)


#  Asset Alignment & Correlation


def align_multiple_assets(dfs_dict):
    """
    Aligns multiple asset DataFrames on a shared datetime index,
    forward-filling missing values. Returns a dict of aligned DataFrames.

    :param dfs_dict: dict[str, pd.DataFrame], e.g. {"AAPL": df_aapl, "MSFT": df_msft, ...}
                    Each df is expected to have a 'timestamp' column.
    :return: dict[str, pd.DataFrame] with aligned indices
    """
    if not dfs_dict:
        logger.warning("No dataframes provided to align_multiple_assets.")
        return dfs_dict

    # 1) Build a union of all timestamps across assets
    all_idx = None
    for symbol, df in dfs_dict.items():
        if "timestamp" not in df.columns:
            logger.warning(f"{symbol} df missing 'timestamp' column.")
            continue
        ts_index = pd.Index(df["timestamp"])
        if all_idx is None:
            all_idx = ts_index
        else:
            all_idx = all_idx.union(ts_index)

    if all_idx is None:
        logger.warning("No valid timestamps found across all assets.")
        return dfs_dict

    # 2) Reindex each DataFrame on the union index, forward-fill
    aligned_dict = {}
    for symbol, df in dfs_dict.items():
        if "timestamp" not in df.columns:
            aligned_dict[symbol] = df
            continue
        df = df.set_index("timestamp")
        df = df.reindex(all_idx, method="ffill")
        df = df.reset_index().rename(columns={"index": "timestamp"})
        aligned_dict[symbol] = df

    return aligned_dict


def compute_rolling_correlations(dfs_dict, col="log_return", window=30):
    """
    Computes rolling correlation of each asset with the first 'base' asset.
    Alternatively, you could compute a matrix of pairwise correlations.

    :param dfs_dict: dict[str, pd.DataFrame], aligned
    :param col: column to use for correlation (e.g. 'log_return')
    :param window: rolling window (in rows)
    :return: pd.DataFrame with columns: ['timestamp'] + correlation columns
    """
    symbols = list(dfs_dict.keys())
    if len(symbols) < 2:
        logger.warning(
            "Only one or zero assets present; skipping cross-asset correlation."
        )
        return pd.DataFrame()

    # Ensure alignment
    dfs_aligned = align_multiple_assets(dfs_dict)
    base_sym = symbols[0]
    base_df = dfs_aligned[base_sym].copy()

    if "timestamp" not in base_df.columns:
        base_df.reset_index(drop=False, inplace=True)

    for sym in symbols[1:]:
        other_df = dfs_aligned[sym]
        corr_vals = []
        for i in range(len(base_df)):
            start_idx = max(0, i - window + 1)
            series_base = base_df[col].iloc[start_idx : i + 1]
            series_other = other_df[col].iloc[start_idx : i + 1]

            std_b = series_base.std()
            std_o = series_other.std()
            if std_b == 0 or std_o == 0 or np.isnan(std_b) or np.isnan(std_o):
                corr_vals.append(np.nan)
            else:
                c = series_base.corr(series_other)
                corr_vals.append(c)
        base_df[f"corr_{base_sym}_{sym}"] = corr_vals

    corr_cols = ["timestamp"] + [c for c in base_df.columns if c.startswith("corr_")]
    return base_df[corr_cols]


def compute_pairwise_correlation_matrix(dfs_dict, col="log_return", window=30):
    """
    Computes a single correlation matrix (NxN) for the final row (or a chosen snapshot),
    across all assets. This is for GNN adjacency building.

    :param dfs_dict: dict[str, pd.DataFrame], aligned
    :param col: str, column name for correlation
    :param window: int, rolling window size
    :return: (symbols, corr_matrix) where corr_matrix is NxN
    """
    symbols = sorted(dfs_dict.keys())
    if len(symbols) < 2:
        logger.warning("Not enough assets to build correlation matrix.")
        return symbols, np.array([[]])

    dfs_aligned = align_multiple_assets(dfs_dict)
    final_series = {}
    for sym in symbols:
        df = dfs_aligned[sym]
        if len(df) < window:
            series = df[col]
        else:
            series = df[col].iloc[-window:]
        final_series[sym] = series.reset_index(drop=True)

    n = len(symbols)
    corr_mat = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if j < i:
                corr_mat[i, j] = corr_mat[j, i]
            elif i == j:
                corr_mat[i, j] = 1.0
            else:
                base = final_series[symbols[i]]
                other = final_series[symbols[j]]
                if base.std() == 0 or other.std() == 0:
                    c = 0.0
                else:
                    c = base.corr(other)
                corr_mat[i, j] = c
    return symbols, corr_mat


# HMM for Regime-Switching


def add_regime_label(
    df, rv_col="rv", n_states=3, covariance_type="full", random_state=42
):
    """
    Fit a GaussianHMM on 'rv_col' to discover hidden regimes.
    """
    if rv_col not in df.columns:
        logger.warning(f"{rv_col} not found in DataFrame. Skipping regime labeling.")
        df["regime"] = 0
        return df

    df[rv_col] = df[rv_col].ffill().bfill()
    df[rv_col].fillna(0.0)

    if len(df) < n_states:
        logger.warning("Not enough rows for HMM fit; defaulting all regimes to 0.")
        df["regime"] = 0
        return df

    data = df[rv_col].values.reshape(-1, 1)
    if np.isnan(data).any() or np.isinf(data).any():
        logger.warning("Data contains NaN or inf, skipping regime labeling.")
        df["regime"] = 0
        return df

    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        random_state=random_state,
    )
    model.fit(data)
    regimes = model.predict(data)
    df["regime"] = regimes
    return df


# GNN for Cross-Asset Synergy


class CorrelationGNN(torch.nn.Module):
    """
    Example GNN that processes correlation-based adjacency among N assets.
    'x' can hold per-asset features, 'edge_index' defines connectivity.
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


def build_gnn_data(symbols, corr_matrix, threshold=0.2):
    """
    Build a PyTorch Geometric Data object from a correlation matrix among N assets.
    """
    N = len(symbols)
    if N == 0 or corr_matrix.size == 0:
        return None

    x = torch.zeros((N, 1), dtype=torch.float)

    edges = []
    for i in range(N):
        for j in range(N):
            if j > i:
                if abs(corr_matrix[i, j]) >= threshold:
                    edges.append([i, j])
                    edges.append([j, i])

    if not edges:
        for i in range(N):
            edges.append([i, i])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    data = Data(x=x, edge_index=edge_index)
    data = NormalizeFeatures()(data)
    return data


def train_correlation_gnn(
    data, hidden_dim=16, num_layers=2, epochs=50, lr=1e-3, synergy_loss="mse"
):
    """
    Train a GNN to embed N assets in a synergy space. Self-supervised toy approach.
    """
    if data is None:
        logger.warning("No GNN data provided. Returning None.")
        return None

    model = CorrelationGNN(
        num_features=data.x.size(1), hidden_dim=hidden_dim, num_layers=num_layers
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)

        if synergy_loss == "mse":
            loss = torch.mean(out**2)  # push embeddings near 0
        else:
            loss = torch.tensor(0.0, requires_grad=True)

        loss.backward()
        optimizer.step()

    final_emb = model(data.x, data.edge_index).detach().cpu().numpy()
    return final_emb


# Orchestrate


def run_hierarchical_nn(dfs_dict, config):
    """
    Main entry to run multi-asset synergy steps:
      1) Align data
      2) If hmm.enabled, add HMM-based regimes
      3) If gnn.enabled, build correlation adjacency & produce embeddings
    """
    ma_cfg = config.get("multi_asset", {})
    if not ma_cfg.get("enabled", False):
        logger.info("Multi-asset synergy not enabled. Returning original dfs_dict.")
        return dfs_dict, None

    # 1) Align
    aligned_dict = align_multiple_assets(dfs_dict)

    # 2) Optionally run HMM on each symbol
    final_dfs_dict = {}
    hmm_cfg = ma_cfg.get("hmm", {})
    if hmm_cfg.get("enabled", False):
        n_states = hmm_cfg.get("n_states", 3)
        cov_type = hmm_cfg.get("covariance_type", "full")
        random_st = hmm_cfg.get("random_state", 42)

        for sym, df in aligned_dict.items():
            df = add_regime_label(
                df,
                rv_col="rv",
                n_states=n_states,
                covariance_type=cov_type,
                random_state=random_st,
            )
            final_dfs_dict[sym] = df
    else:
        final_dfs_dict = aligned_dict

    # 3) GNN synergy
    gnn_cfg = ma_cfg.get("gnn", {})
    if not gnn_cfg.get("enabled", False):
        logger.info("GNN synergy not enabled.")
        return final_dfs_dict, None

    threshold = ma_cfg.get("correlation_threshold", 0.2)
    corr_col = ma_cfg.get("correlation_col", "log_return")
    window = ma_cfg.get("correlation_window", 30)

    symbols, corr_mat = compute_pairwise_correlation_matrix(
        final_dfs_dict, col=corr_col, window=window
    )
    data = build_gnn_data(symbols, corr_mat, threshold=threshold)
    if data is None:
        logger.warning("No valid GNN data. Returning None for embeddings.")
        return final_dfs_dict, None

    hidden_dim = gnn_cfg.get("hidden_dim", 16)
    num_layers = gnn_cfg.get("num_layers", 2)
    epochs = gnn_cfg.get("epochs", 50)
    lr = gnn_cfg.get("lr", 1e-3)
    synergy_loss = gnn_cfg.get("synergy_loss", "mse")

    final_emb = train_correlation_gnn(
        data, hidden_dim, num_layers, epochs, lr, synergy_loss
    )
    logger.info(
        f"GNN synergy embeddings shape: {final_emb.shape} for symbols {symbols}"
    )
    return final_dfs_dict, final_emb
