import logging
import time
from typing import Any, Dict

import pandas as pd

from src.data_ingestion.ticker_aggs.merge import combine_by_ticker
from src.subpipelines.ingest_process import raw_to_enriched
from src.core_ml.train_test_split import split_train_test 
from src.core_ml.models.deep_vol_models import build_model, train_deep_model
from src.core_ml.models.multi_asset_models import run_hierarchical_nn 
from src.core_ml.models.wavelets import run_stochastic_models
from src.core_ml.hyperparameter_search import run_hyperparameter_search

logger = logging.getLogger(__name__)


def main_pipeline(config: Dict[str, Any], data_csv_dir: str) -> None:
    """
    Orchestrates the entire pipeline from raw data ingestion/preprocessing
    to final model training (deep/stochastic) and optional hyperparameter search.

    Parameters
    ----------
    config : dict
        The full pipeline config. Must contain relevant sub-configs for
        data ingestion, data processing, deep learning, multi_asset, stochastic, etc.
    data_csv_dir : str
        Folder or path containing input CSVs (and possibly where to write outputs).
    """

    start_time = time.time()
    logger.info("=== Starting main pipeline ===")
    # -------------------------------------------------
    # 1) Data Ingestion & Preprocessing
    # -------------------------------------------------
    df_preprocessed = raw_to_enriched(config["data"])  
    logger.info("Preprocessing completed. Shape=%s", df_preprocessed.shape)

    # -------------------------------------------------
    # 2) Multi-asset synergy (GNN, HMM, etc.)
    # -------------------------------------------------
    synergy_cfg = config.get("multi_asset", {})
    if synergy_cfg.get("enabled", False):
        # Convert single big DF to dict by ticker
        ticker_groups = df_preprocessed.groupby("ticker", group_keys=False)
        dfs_dict = {t: g.copy() for t, g in ticker_groups}
        final_dfs_dict, gnn_emb = run_hierarchical_nn(dfs_dict, config)
        # Recombine
        df_preprocessed = pd.concat(final_dfs_dict.values(), ignore_index=True)
        logger.info("Multi-asset synergy step completed. Combined shape=%s", df_preprocessed.shape)
    else:
        logger.info("Multi-asset synergy disabled. Skipping.")

    # -------------------------------------------------
    # 3) Stochastic Models
    # -------------------------------------------------

    stoch_cfg = config.get("stochastic", {})
    if stoch_cfg.get("enabled", False):

        rv_series = df_preprocessed["rv"] if "rv" in df_preprocessed.columns else None
        if rv_series is not None:
            stoch_results = run_stochastic_models(rv_series, config)
            logger.info("Stochastic modeling done: keys=%s", list(stoch_results.keys()))
        else:
            logger.warning("No 'rv' column found; skipping stochastic modeling.")
    else:
        logger.info("Stochastic models disabled. Skipping.")

    # -------------------------------------------------
    # 4) Train/Test Split
    # -------------------------------------------------
    target_col = config["data"].get("target_col", "rv")  # Or whatever the target is

    X_train, X_test, y_train, y_test = split_train_test(
        df_preprocessed, target_col=target_col, test_ratio=0.2, random_state=42
    )
    logger.info("Train/Test split done. Train shape=%s, Test shape=%s", X_train.shape, X_test.shape)

    # -------------------------------------------------
    # 5) Hyperparameter Search
    # -------------------------------------------------
    hp_search_cfg = config.get("hyperparameter_search", {})
    if hp_search_cfg.get("enabled", False):
        # Let's assume your hyperparam search logic loads some CSV or uses the data
        best_params = run_hyperparameter_search(config, data_csv_dir, method="deep")
        logger.info("Best hyperparameters found: %s", best_params)
    else:
        logger.info("Hyperparameter search disabled. Skipping.")

    # -------------------------------------------------
    # 6) Final Deep Model Training
    # -------------------------------------------------
    from torch.utils.data import TensorDataset
    import torch

    X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    model = train_deep_model(config, train_dataset, device="cpu")
    if model is not None:
        logger.info("Deep model trained successfully.")
    else:
        logger.info("No deep model trained (deep learning disabled or skipping).")


    elapsed = time.time() - start_time
    logger.info("=== Main pipeline completed in %.2f seconds ===", elapsed)


if __name__ == "__main__":
    import json
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # 1) Load config (example from a JSON file or some dictionary)
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "config.json"  # default path

    with open(config_path, "r") as f:
        config = json.load(f)

    # 2) Set the data directory (e.g. CSV location)
    data_csv_dir = config["data"]["csv_dir"]

    # 3) Run the pipeline
    main_pipeline(config, data_csv_dir)
