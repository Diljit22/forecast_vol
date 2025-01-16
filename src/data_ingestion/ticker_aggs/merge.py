# src/pipelines/data_ingestion.py
import os
from collections import defaultdict
import pandas as pd
from typing import Tuple
from src.utils.file_io import (
    ensure_dir_exists,
    get_file_paths,
    load_csv,
    save_csv,
)
from src.utils.helpers import extract_ticker
import logging

logger = logging.getLogger(__name__)


def group_paths_by_ticker(in_path: str) -> dict[str, list[str]]:
    """
    - Finds all CSVs in 'in_path'
    - Groups them by ticker name (via `extract_ticker`)
    - Creates 'out_path' directory (just in case)
    - Returns a dict: { 'AAPL': ['/path/to/AAPL_1m.csv', ...],...}
    """
    logger.info(f"Combining CSV files by ticker from {in_path}")
    csv_paths = get_file_paths(in_path, file_format="csv")

    ticker_path_dict = defaultdict(list)
    for path_ in csv_paths:
        ticker = extract_ticker(path_)
        ticker_path_dict[ticker].append(path_)

    return dict(ticker_path_dict)


def load_data_with_ticker_type(list_paths) -> list:
    """
    Extracts the ticker from the given filepath and checks if it is an ETF.
    """
    dfs = []
    etf_list = ["SPY", "VXX", "QQQ"]
    for path_ in list_paths:
        df = load_csv(path_)
        ticker_ = extract_ticker(path_)
        df["ticker"] = ticker_
        df["is_etf"] = 1.0 if ticker_ in etf_list else 0.0
        dfs.append(df)

    return dfs


def concat_and_sort(dfs, save=False, out_path: str = None):
    """
    Stacks a list of CSV files by appending rows
    sorts by the 't' column,
    and saves the result to out_path.
    """
    df_concat = pd.concat(dfs, ignore_index=True)
    df_concat = df_concat.sort_values(["t", "ticker"])
    if save:
        folder = os.path.dirname(out_path)
        ensure_dir_exists(folder)
        save_csv(df_concat, out_path)
    return df_concat
