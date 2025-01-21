import pandas as pd
import logging
from script_generated_files.market_hours import ACTIVE_SESSIONS
import numpy as np

logger = logging.getLogger(__name__)

ONE_MINUTE_MS = 60_000

def restrict_to_active(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter rows so that 't' is within the open/close ms in ACTIVE_SESSIONS.
    - df['datetime'] is expected to match the date keys in
        ACTIVE_SESSIONS (YYYY-MM-DD).
    """
    df = df.copy()
    # Ensure 'datetime' is a proper datetime type
    if "datetime" not in df.columns:
        # If not present, assume 't' is the timestamp
        df["datetime"] = pd.to_datetime(df["t"], unit="ms")
    else:
        df["datetime"] = pd.to_datetime(df["datetime"])

    dt_str = df["datetime"].dt.strftime("%Y-%m-%d")

    open_times = dt_str.map(
        lambda d: ACTIVE_SESSIONS[d][0] if d in ACTIVE_SESSIONS else np.nan
    )
    close_times = dt_str.map(
        lambda d: ACTIVE_SESSIONS[d][1] if d in ACTIVE_SESSIONS else np.nan
    )

    mask = (df["t"] >= open_times) & ((df["t"] + ONE_MINUTE_MS) <= close_times)
    return df[mask].copy()


def restrict_all_tickers_to_active_sessions(all_dfs: list) -> list:
    logger.info(f"Restricting all trading days")
    dfs = []
    for df in all_dfs:
        df["t"] = pd.to_numeric(df["t"], errors="coerce")
        df.dropna(subset=["t"], inplace=True)

        df_restricted = restrict_to_active(df)
        logger.info(
            f"{df['ticker'].iloc[0]}: restricted from {df.shape} -> {df_restricted.shape}"
        )

        dfs.append(df_restricted)
    return dfs


def compute_global_coverage(restricted_to_trading_hours_dfs: list):
    """
    Reads each restricted ticker, finds min(t) and max(t).
    Returns (coverage_start, coverage_end).
    """
    min_ts_list = [df["t"].min() for df in restricted_to_trading_hours_dfs]
    max_ts_list = [df["t"].max() for df in restricted_to_trading_hours_dfs]
    coverage_start = max(min_ts_list) if min_ts_list else None
    coverage_end = min(max_ts_list) + 2 * ONE_MINUTE_MS if max_ts_list else None
    return coverage_start, coverage_end


def build_active_minutes(coverage_start: int, coverage_end: int) -> np.ndarray:
    """
    Return an array of all 1-minute timestamps within ACTIVE_SESSIONS
    intersected with [coverage_start, coverage_end).
    """
    all_timestamps = []

    for date_str, (day_start, day_end) in ACTIVE_SESSIONS.items():
        # Intersect day range with coverage
        start = max(day_start, coverage_start)
        end = min(day_end, coverage_end)
        if end > start:
            minutes = np.arange(start, end, ONE_MINUTE_MS, dtype=np.int64)
            all_timestamps.append(minutes)

    if not all_timestamps:
        return np.array([], dtype=np.int64)

    return np.concatenate(all_timestamps)


def restrict_to_joint_coverage(restricted_to_trading_hours_dfs):
    coverage_start, coverage_end = compute_global_coverage(
        restricted_to_trading_hours_dfs
    )
    dfs = []
    for df in restricted_to_trading_hours_dfs:
        df = df[(df["t"] >= coverage_start) & (df["t"] < coverage_end)].copy()
        dfs.append(df)
    return dfs
