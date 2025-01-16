from src.data_ingestion.ticker_aggs.merge import (
    group_paths_by_ticker,
    load_data_with_ticker_type,
)
from src.data_ingestion.ticker_aggs.resample import resample_tickers_1min
from src.data_ingestion.ticker_aggs.restrict import (
    restrict_all_tickers_to_active_sessions,
    restrict_to_joint_coverage,
)
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def merge_by_ticker(in_path: str) -> list[pd.DataFrame]:
    """
    For each ticker, collect all CSVs in `in_path`,
    concatenate and sort them, return a list of DFs (one per ticker).
    """
    ticker_path_dict = group_paths_by_ticker(in_path)
    all_ticker_dfs = []
    for ticker_name, paths_list in ticker_path_dict.items():
        dfs_for_this_ticker = load_data_with_ticker_type(paths_list)
        # Merge all CSVs for this ticker into a single DF
        merged_df = pd.concat(dfs_for_this_ticker, ignore_index=True)
        # merged_df = concat_and_sort(dfs_for_this_ticker)
        all_ticker_dfs.append(merged_df)

    logger.info("merge_by_ticker: Done.")
    return all_ticker_dfs


def apply_restrictions(all_dfs):
    """
    Wrapper function to handle the entire 'restrict' step.
    """
    logger.info("=== Step 1: Restricting to active sessions ===")
    dfs_active = restrict_all_tickers_to_active_sessions(all_dfs)
    logger.info("=== Step 2: Restrict to joint coverage ===")
    dfs_final = restrict_to_joint_coverage(dfs_active)
    logger.info("Restrictions complete!")
    return dfs_final


def preprocess(config):
    ticker_merged_dfs = merge_by_ticker(config["path_raw"])
    dfs_restricted = apply_restrictions(ticker_merged_dfs)
    dfs_sampled = resample_tickers_1min(dfs_restricted)
    df_stacked = pd.concat(dfs_sampled, ignore_index=True)
    # df_stacked.drop('days', axis=1, inplace=True)
    return df_stacked
