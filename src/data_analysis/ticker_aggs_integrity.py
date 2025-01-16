import pandas as pd
import logging
import os
import numpy as np

from script_generated_files.market_hours import ACTIVE_SESSIONS
from src.utils.file_io import get_file_paths, load_csv
from src.utils.helpers import extract_ticker

logger = logging.getLogger(__name__)
ONE_MINUTE_MS = 60_000


def restrict_to_active(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter rows so that 't' is within the open/close ms in ACTIVE_SESSIONS.
    - df['datetime'] is expected to match date keys in ACTIVE_SESSIONS
    """
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])  # ensure datetime type
    dt_str = df["datetime"].dt.strftime("%Y-%m-%d")
    open_times = dt_str.map(
        lambda d: ACTIVE_SESSIONS[d][0] if d in ACTIVE_SESSIONS else np.nan
    )
    close_times = dt_str.map(
        lambda d: ACTIVE_SESSIONS[d][1] if d in ACTIVE_SESSIONS else np.nan
    )
    mask = (df["t"] >= open_times) & ((df["t"] + ONE_MINUTE_MS) <= close_times)
    return df[mask].copy()


def compare_dfs_by_dict(df_big, df_small):
    """
    - Compare missing keys
    - Compare existing keys, note differences
    """

    df_big = df_big.drop("day", axis=1)
    cols_order = ["volume", "vwap", "open", "close", "high", "low", "trades_count"]

    dict_big = {}
    dict_small = {}

    for _, row in df_big.iterrows():
        t_val = row["t"]
        tup = tuple(row[c] for c in cols_order)
        dict_big[t_val] = tup

    for _, row in df_small.iterrows():
        t_val = row["t"]
        tup = tuple(row[c] for c in cols_order)
        dict_small[t_val] = tup

    # Missing keys
    keys_big = set(dict_big.keys())
    keys_small = set(dict_small.keys())

    # Keys in df_big that are not in df_small
    missing_in_small = keys_big - keys_small
    # Keys in df_small that are not in df_big
    missing_in_big = keys_small - keys_big

    print(f"Missing keys in df_small (present only in df_big): {len(missing_in_small)}")
    print(f"Missing keys in df_big (present only in df_small): {len(missing_in_big)}")

    common_keys = keys_big.intersection(keys_small)

    differ_count = 0
    for k in common_keys:
        big_tuple = dict_big[k]
        sml_tuple = dict_small[k]
        if big_tuple != sml_tuple:
            differ_count += 1

    print(f"Common keys that differ in at least one column: {differ_count}")

    summary = {
        "missing_in_small_count": len(missing_in_small),
        "missing_in_big_count": len(missing_in_big),
        "common_keys_differ_count": differ_count,
    }
    return summary


def print_df_info(df: pd.DataFrame, ticker: str):
    """
    Print basic info about a DataFrame: shape, columns, NaN counts, and head.
    """
    logger.info(f"--- {ticker} DataFrame Info ---")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"NaN counts:\n{df.isna().sum()}")
    logger.debug(f"Head:\n{df.head()}")


def check_missing_timestamps(df: pd.DataFrame, ticker: str):
    """
    For each date in df, check how many 1-minute timestamps are expected
    vs. how many are present. Print if there are missing timestamps.

    Assumes:
    - df['t'] is a Unix timestamp in milliseconds.
    - df['datetime'] is a pandas datetime.
    - ACTIVE_SESSIONS has open/close times for each date as (open_msec, close_msec).
    """
    if df.empty:
        logger.info(f"No active rows for {ticker}. Skipping missing-timestamps check.")
        return

    df["date_str"] = df["datetime"].dt.strftime("%Y-%m-%d")
    groups = df.groupby("date_str")

    print(f"------{ticker}")
    for date_val, group in groups:
        if date_val not in ACTIVE_SESSIONS:
            continue

        open_msec, close_msec = ACTIVE_SESSIONS[date_val]
        total_minutes = int((close_msec - open_msec) // ONE_MINUTE_MS)

        actual_count = len(group)
        missing_count = total_minutes - actual_count

        if missing_count > 0:
            print(f"{date_val}: Missing {missing_count} / {total_minutes} timestamps.")
            print(f"{date_val}: Missing {missing_count} / {total_minutes} timestamps.")
            logger.warning(
                f"{ticker} on {date_val}: Missing {missing_count} / {total_minutes} timestamps."
            )


def analyze_all_csvs(in_folder_path: str):
    """
    - Load all .csv files in `in_folder_path`.
    - For each file:
       - Extract ticker
       - Restrict to active times
       - Print basic info
       - Check for missing timestamps
    """
    csv_files = get_file_paths(in_folder_path, file_format="csv")
    for fp in csv_files:
        ticker = extract_ticker(fp)
        df = load_csv(fp)
        print(f"INFO ---- {ticker}")
        df_active = restrict_to_active(df)
        print(f"SHAPE: {df.shape} -> {df_active.shape}")
        check_missing_timestamps(df_active, ticker)

        print(f"Finished analysis for {ticker} ({os.path.basename(fp)}).")
        print("-----------------------------------------------------")


def data_peak(train=True):
    logging.basicConfig(level=logging.DEBUG)
    if train == True:
        in_path = r"artifacts\interim\merged_by_year"
    else:
        in_path = r"data\inference\raw"
    analyze_all_csvs(in_path)
