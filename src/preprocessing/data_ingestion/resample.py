# src/pipelines/data_processing.py
import logging
import numpy as np
import pandas as pd
from script_generated_files.market_hours import ACTIVE_SESSIONS
from src.utils.helpers import num_covers
from src.preprocessing.data_ingestion.restrict import (
    build_active_minutes,
    compute_global_coverage,
)

logger = logging.getLogger(__name__)

ONE_MINUTE_MS = 60_000


def resample_1min(
    df: pd.DataFrame,
    coverage_start: int,
    coverage_end: int,
    epsilon_percent: float = 0.002,
) -> pd.DataFrame:
    """
    Fill missing 1-min bars for intervals.

    Build only those 1-minute timestamps that fall inside ACTIVE_SESSIONS
        intersected with [coverage_start, coverage_end).
    Merge with df on 't', creating NaN rows for missing bars.
    For each day:
        - Calculate average volume for that day (from existing bars).
        - Identify consecutive known bars that bound missing ranges.
        - Interpolate open/close between these known bars if multiple minutes are missing.
        - For high/low, randomly apply a small +/- percentage around (open, close).
        - Volume: fill with day-average or a smoothed interpolation.
        - vwap = (open + close)/2

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: ["t", "open", "close", "high", "low", "volume", "vwap", "trades_count"].
    coverage_start : int
        The global coverage start in Unix ms.
    coverage_end : int
        The global coverage end in Unix ms.
    epsilon_percent : float
        A fraction of the price range used to create high/low variation, e.g. 0.002 => 0.2%.

    Returns
    -------
    pd.DataFrame
        A DataFrame with missing bars filled, only during active sessions.

    """
    if df.empty:
        logger.warning("DataFrame is empty. Returning empty resampled DF.")
        return df.copy()

    df = df.copy()
    df.sort_values("t", inplace=True)

    # Build the full set of 1-minute timestamps within the coverage.
    active_ts = build_active_minutes(coverage_start, coverage_end)
    full_df = pd.DataFrame({"t": active_ts})
    merged = pd.merge(full_df, df, on="t", how="left", suffixes=("", "_orig"))

    # Rebuild 'datetime' from 't'
    merged["datetime"] = pd.to_datetime(merged["t"], unit="ms")
    merged["day"] = merged["datetime"].dt.strftime("%Y-%m-%d")

    day_groups = merged.groupby("day", group_keys=False)
    filled_list = []
    for day_val, group in day_groups:
        if group.empty:
            filled_list.append(group)
            continue

        group = group.sort_values("t").copy()
        existing_rows = group.dropna(subset=["open", "close", "volume"])

        if existing_rows.empty:
            day_avg_volume = 0.0
        else:
            day_avg_volume = existing_rows["volume"].mean()

        # Identify indices where open/close is known
        known_mask = group["open"].notna() & group["close"].notna()
        known_idxs = group.index[known_mask].tolist()

        def fill_gap(i_idx, j_idx):
            """
            Interpolate missing bars for the time range between two known bars.
            """
            if j_idx <= i_idx + 1:
                return

            open_i = group.at[i_idx, "open"]
            close_i = group.at[i_idx, "close"]
            vol_i = group.at[i_idx, "volume"]

            open_j = group.at[j_idx, "open"]
            close_j = group.at[j_idx, "close"]
            vol_j = group.at[j_idx, "volume"]

            gap_size = j_idx - i_idx - 1
            for step in range(1, gap_size + 1):
                mid_idx = i_idx + step
                frac = step / (gap_size + 1)

                # Linear interpolation for open/close
                open_m = (1 - frac) * open_i + frac * open_j
                close_m = (1 - frac) * close_i + frac * open_j

                # Volume interpolation
                if (
                    not np.isnan(vol_i)
                    and not np.isnan(vol_j)
                    and vol_i > 0
                    and vol_j > 0
                ):
                    vol_m = (1 - frac) * vol_i + frac * vol_j
                else:
                    vol_m = day_avg_volume

                # Interpolate trades_count
                trades_i = group.at[i_idx, "trades_count"]
                if pd.isna(trades_i):
                    trades_i = 0
                trades_j = group.at[j_idx, "trades_count"]
                if pd.isna(trades_j):
                    trades_j = 0
                trades_m = (1 - frac) * trades_i + frac * trades_j

                # high/low => add random ± epsilon around (open_m, close_m)
                mn_price = min(open_m, close_m)
                mx_price = max(open_m, close_m)
                price_range = mx_price - mn_price
                eps_val = price_range * epsilon_percent

                hi = mx_price + np.random.uniform(0, 2 * eps_val)
                lo = mn_price - np.random.uniform(0, 2 * eps_val)

                # vwap = avg open and close
                vwap_m = (open_m + close_m) / 2

                # fill the row
                group.at[mid_idx, "open"] = open_m
                group.at[mid_idx, "close"] = close_m
                group.at[mid_idx, "volume"] = vol_m
                group.at[mid_idx, "trades_count"] = trades_m
                group.at[mid_idx, "high"] = max(hi, lo)
                group.at[mid_idx, "low"] = min(hi, lo)
                group.at[mid_idx, "vwap"] = vwap_m

        # Fill the gaps between known bars
        for k in range(len(known_idxs) - 1):
            i_idx = known_idxs[k]
            j_idx = known_idxs[k + 1]
            fill_gap(i_idx, j_idx)

        numeric_cols = [
            "open",
            "close",
            "high",
            "low",
            "volume",
            "trades_count",
            "vwap",
        ]
        # Forward-fill, then backward-fill missing numeric columns
        group[numeric_cols] = group[numeric_cols].ffill().bfill()

        for col in numeric_cols:
            if col == "volume":
                group[col].fillna(day_avg_volume, inplace=True)
            elif col in ["open", "close"]:
                # fallback: forward or backward fill if any known values exist
                group[col].fillna(method="ffill", inplace=True)
                group[col].fillna(method="bfill", inplace=True)
            else:
                group[col].fillna(0, inplace=True)

        filled_list.append(group)

    # Combine day-level results
    final = pd.concat(filled_list, axis=0).sort_values("t")
    final.reset_index(drop=True, inplace=True)

    return final


def resample_tickers_1min(joint_coverage_restricted_dfs: list):
    """
    For each restricted CSV:
    - read & filter to [coverage_start, coverage_end)
    - resample 1-min bars to fill missing
    - save to out_folder_resampled/{ticker}.csv
    - compare_dfs_func is for debugging
    """
    cov_range = compute_global_coverage(joint_coverage_restricted_dfs)
    total_bars = num_covers(cov_range, ACTIVE_SESSIONS)
    dfs = []
    for df in joint_coverage_restricted_dfs:
        missing_count = total_bars - len(df)
        logger.info(f"{df['ticker'].iloc[0]}: Missing {missing_count} bars.")
        df_resampled = resample_1min(df, cov_range[0], cov_range[1])
        df_resampled["ticker"] = df["ticker"].iloc[0]
        dfs.append(df_resampled)
        if missing_count != 0:
            logger.info(
                f"{df['ticker'].iloc[0]}: Resampled {df.shape} -> {df_resampled.shape}"
            )
        else:
            logger.info(
                f"{df['ticker'].iloc[0]}: Not Resampled {df.shape} -> {df_resampled.shape}"
            )

    return dfs
