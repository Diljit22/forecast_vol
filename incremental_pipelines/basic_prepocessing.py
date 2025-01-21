import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from src.preprocessing.feature_engineering.inject_basic_features import (
    compute_all_features_simple,
)
from src.preprocessing.data_ingestion.merge import (
    group_paths_by_ticker,
    load_data_with_ticker_type,
)
from src.preprocessing.data_ingestion.resample import resample_tickers_1min
from src.preprocessing.data_ingestion.restrict import (
    restrict_all_tickers_to_active_sessions,
    restrict_to_joint_coverage,
)
from src.utils.file_io import save_csv

logger = logging.getLogger(__name__)


def merge_by_ticker(in_path: str) -> list[pd.DataFrame]:
    """
    For each ticker, collect all CSVs in `in_path`,
    concatenate and sort them, return a list of DFs (one per ticker).
    """
    logger.info("Beginning merge_by_ticker")
    ticker_path_dict = group_paths_by_ticker(in_path)
    all_ticker_dfs = []
    for ticker_name, paths_list in ticker_path_dict.items():
        dfs_for_this_ticker = load_data_with_ticker_type(paths_list)
        merged_df = pd.concat(dfs_for_this_ticker, ignore_index=True)
        all_ticker_dfs.append(merged_df)

    logger.info("merge_by_ticker: complete.")
    return all_ticker_dfs


def apply_restrictions(all_dfs):
    """
    Wrapper function to handle the entire 'restrict' step.
    """
    logger.info("Beginning apply_restrictions")
    logger.info("=== Step 1: Restricting to active sessions ===")
    dfs_active = restrict_all_tickers_to_active_sessions(all_dfs)
    logger.info("=== Step 2: Restrict to joint coverage ===")
    dfs_final = restrict_to_joint_coverage(dfs_active)
    logger.info("apply_restrictions: complete")
    return dfs_final


def ingestion(data_cfg):
    logger.info("Beginning Ingestion")
    ticker_merged_dfs = merge_by_ticker(data_cfg["path_raw"])
    dfs_restricted = apply_restrictions(ticker_merged_dfs)
    logger.info("Beginning resample_tickers_1min")
    dfs_sampled = resample_tickers_1min(dfs_restricted)
    logger.info("resample_tickers_1min: complete")
    df_stacked = pd.concat(dfs_sampled, ignore_index=True)
    logger.info("Ingestion: complete")
    return df_stacked

def drop_n_lowest_times_per_ticker(df: pd.DataFrame, n) -> pd.DataFrame:
    """
    For each ticker, sort by t_col and drop the N lowest times.
    Concatenates the result into one DataFrame and returns it.
    """
    df_list = []
    # Group by ticker, handle each separately
    for _, group in df.groupby("ticker", group_keys=False):
        group_sorted = group.sort_values(by="t")
        group_trimmed = group_sorted.iloc[n:]
        df_list.append(group_trimmed)
    # Re-combine into a single DataFrame
    df_result = pd.concat(df_list, ignore_index=True)
    df_result.dropna()
    return df_result

def preprocess(config):
    # 1. ingestion
    data_cfg = config["data"]
    df = ingestion(data_cfg)
    # 2. Compute additional features
    logger.info("Beginning feature_injection: dim(df) = {df.shape}")
    df = compute_all_features_simple(df)
    logger.info("feature_injection: complete: dim(df) = {df.shape}")

    # 3. Drop unnecessary columns
    logger.info(f"Drop cols and rows: dim(df) = {df.shape}")
    columns_to_drop = ["day", "datetime", "returns"]
    df.drop(columns=columns_to_drop, inplace=True, errors="ignore")

    # 4. Drop rows to align lookback windows
    max_lookback_window_ = 29
    df = drop_n_lowest_times_per_ticker(df, max_lookback_window_)
    logger.info(f"Done dropping - New dim(df) = {df.shape}")

    # 5. Mark is_etf as 1/0
    logger.info(f"Str -> Numeric; One Hot Encoding: dim(df) = {df.shape}")
    df["is_etf"] = df["ticker"].isin(["SPY", "VXX", "QQQ"]).astype(int)

    # 6. One-hot encode 'ticker' with sklearn
    ohe = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore", dtype='int64')
    ticker_ohe = ohe.fit_transform(df[["ticker"]])  # returns a numpy array

    # Get the column names from the encoder
    encoded_cols = ohe.get_feature_names_out(["ticker"])

    # Turn this array into a DataFrame
    df_ohe = pd.DataFrame(ticker_ohe, columns=encoded_cols, index=df.index)

    # 7. Merge the new columns back
    df = pd.concat([df.drop(columns=["ticker"]), df_ohe], axis=1)
    logger.info(f"Done change to numeric: dim(df) = {df.shape}")

    # 8. Save CSV
    save_csv(df, data_cfg["path_preprocessed"])

    return df

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
    from incremental_pipelines.basic_prepocessing import preprocess

    logging.info("Starting preprocessing...")
    preprocess(cfg)
    logging.info("Preprocessing completed.")