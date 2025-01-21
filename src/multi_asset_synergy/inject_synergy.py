"""
merge synergy outputs (HMM regime, GNN embeddings) into the master DataFrame.
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def assign_dropped_ticker(df: pd.DataFrame, all_ticker_cols: list, dropped_ticker: str = "AAPL"):
    if "ticker" not in df.columns:
        df["ticker"] = None

    for idx, row in df.iterrows():
        sum_tickers = sum(row[col] for col in all_ticker_cols)
        if sum_tickers == 0:
            df.at[idx, "ticker"] = dropped_ticker
        else:
            # find which col is 1
            for col in all_ticker_cols:
                if row[col] == 1:
                    sym = col.replace("ticker_", "")
                    df.at[idx, "ticker"] = sym
                    break
    return df


def merge_hmm_regime(dfs_dict: dict) -> pd.DataFrame:
    """
    Merges sub-DataFrames, each containing 'regime', into one DataFrame.
    """
    combined = pd.concat(dfs_dict.values(), ignore_index=True)
    combined.sort_values(by=["timestamp"], inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return combined


def inject_gnn_embeddings(merged_df: pd.DataFrame, symbols: list, embeddings: np.ndarray) -> pd.DataFrame:
    """
    Adds columns gnn_emb_0..X for each row based on merged_df["ticker"] matching symbols[i].
    """
    if "ticker" not in merged_df.columns:
        logger.warning("No 'ticker' col found in merged_df. Can't inject GNN embeddings.")
        return merged_df

    hidden_dim = embeddings.shape[1] if embeddings.size else 0
    if hidden_dim == 0:
        logger.warning("Empty embeddings array. No gnn_emb_* will be added.")
        return merged_df

    # Build map from sym => embedding
    embed_map = {}
    for i, sym in enumerate(symbols):
        embed_map[sym] = embeddings[i]

    # Add columns
    for d in range(hidden_dim):
        merged_df[f"gnn_emb_{d}"] = 0.0

    # Fill per row
    for idx, row in merged_df.iterrows():
        sym = row["ticker"]
        if sym in embed_map:
            for d in range(hidden_dim):
                merged_df.at[idx, f"gnn_emb_{d}"] = embed_map[sym][d]

    return merged_df
