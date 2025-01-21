
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any
from time import perf_counter

from .hmm_regime import parallel_hmm_labeling
from .gnn_synergy import build_correlation_matrix, build_gnn_data, train_gnn_embeddings
from .inject_synergy import (
    assign_dropped_ticker,
    inject_gnn_embeddings
)

logger = logging.getLogger(__name__)


def run_synergy_pipeline(partial_df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Returns synergy-enriched DataFrame (with 'regime' and 'gnn_emb_*').
    """
    syn_cfg = config["multi_asset"]  # e.g. from YAML
    logger.info("[Synergy] Starting synergy pipeline...")

    start_all = perf_counter()
    # 1) Identify dropped ticker => AAPL
    ticker_cols = [c for c in partial_df.columns if c.startswith("ticker_")]
    if len(ticker_cols) == 0:
        logger.error("No ticker columns found.")
        return partial_df

    dropped_ticker = "AAPL"
    partial_df = assign_dropped_ticker(partial_df, ticker_cols, dropped_ticker=dropped_ticker)
    logger.info(f"[Synergy] Assigned dropped ticker '{dropped_ticker}' to rows with all zero ticker_ columns.")

    # 2) Split DataFrame by ticker
    dfs_dict = {}
    for sym in partial_df["ticker"].unique():
        sub = partial_df[partial_df["ticker"] == sym].copy()
        if "t" in sub.columns:
            sub.rename(columns={"t": "timestamp"}, inplace=True)
        dfs_dict[sym] = sub

    # 3) HMM labeling
    hmm_cfg = syn_cfg["hmm"]
    rv_col = "rv"  # Hard-coded or from config
    dfs_dict = parallel_hmm_labeling(
        dfs_dict,
        rv_col=rv_col,
        n_states=hmm_cfg["n_states"],
        covariance_type=hmm_cfg["covariance_type"],
        random_state=hmm_cfg["random_state"]
    )

    # 4) GNN synergy
    gnn_cfg = syn_cfg["gnn"]
    symbols, corr_mat = build_correlation_matrix(
        dfs_dict,
        col=gnn_cfg["correlation_col"],
        window=gnn_cfg["correlation_window"]
    )
    data = build_gnn_data(symbols, corr_mat, threshold=gnn_cfg["correlation_threshold"])
    embeddings = train_gnn_embeddings(
        data,
        hidden_dim=gnn_cfg["hidden_dim"],
        num_layers=gnn_cfg["num_layers"],
        epochs=gnn_cfg["epochs"],
        lr=gnn_cfg["lr"],
        synergy_loss=gnn_cfg["synergy_loss"]
    )

    # 5) Recombine => single DataFrame
    final_df = pd.concat(dfs_dict.values(), ignore_index=True)
    final_df.sort_values(by=["timestamp", "ticker"], inplace=True)
    final_df.reset_index(drop=True, inplace=True)

    if embeddings.size > 0:
        final_df = inject_gnn_embeddings(final_df, symbols, embeddings)
    else:
        logger.warning("GNN embeddings are empty. Skipping injection of gnn_emb_*.")

    # Possibly rename 'timestamp' => 't'
    if "timestamp" in final_df.columns and "t" not in final_df.columns:
        final_df.rename(columns={"timestamp": "t"}, inplace=True)

    # 6) Save synergy intermediate CSV
    data_cfg = config["data"]
    final_df.to_csv(data_cfg["path_synergy"], index=False)

    logger.info(f"[Synergy] Wrote synergy-enriched data to {data_cfg["path_preprocessed"]}")

    total_duration = perf_counter() - start_all
    logger.info(f"[Synergy] Pipeline finished in {total_duration:.2f}s")

    return final_df


if __name__ == "__main__":

    from src.utils.cfg import init_config
    import pandas as pd
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    import logging
    cfg = init_config()
    logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    df_partial = pd.read_csv("data/intermediate/preprocessed.csv")
    logging.info("Starting preprocessing...")
    run_synergy_pipeline(df_partial, cfg)
    logging.info("Preprocessing completed.")
    
