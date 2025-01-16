import numpy as np
import pandas as pd
from src.core_ml.feature_engineering import *
from src.subpipelines.preprocess_ticker_aggs import preprocess
from src.utils.file_io import save_csv

from sklearn.preprocessing import LabelEncoder


def compute_all_features_simple(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a preprocessed DataFrame containing multiple tickers, apply a fixed sequence
    of feature transformations to each ticker. Returns a single DataFrame with all the
    new columns added.

    Steps performed for each ticker (always, by default):
      1. Sort by 't'
      2. compute_active_nhood
      3. compute_distance_open_close
      4. compute_returns
      5. compute_log_return
      6. compute_parkinson_vol
      7. compute_realized_variance
      8. compute_bipower_variation
      9. compute_garman_klass
      10. compute_integrated_quartic_var
      11. compute_trade_intensity
      12. compute_vov

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: ['ticker', 'datetime', 't', 'open', 'high', 'low', 'close', 'volume']
        (and any others needed by your compute_* functions).

    Returns
    -------
    pd.DataFrame
        The input data plus all new feature columns.
    """
    dfs = []

    # Group by 'ticker' and apply transformations separately
    for ticker, grp in df.groupby("ticker", group_keys=False):
        # 1) Sort by 't'
        grp = grp.sort_values("t").copy()

        # 2) Active neighborhood
        grp = compute_active_nhood(grp)

        # 3) Distance to open/close times
        grp = compute_distance_open_close(grp)

        # 4) Close-to-close returns
        grp = compute_returns(grp)

        # 5) Log return
        grp = compute_log_return(grp, price_col="close")

        # 6) Parkinson volatility (default window=30)
        grp = compute_parkinson_vol(grp, window=30)

        # 7) Realized variance (default window=5)
        grp = compute_realized_variance(grp, window=5)

        # 8) Bipower variation (default window=5)
        grp = compute_bipower_variation(grp, price_col="close", window=5)

        # 9) Garman-Klass volatility measure (default window=5)
        grp = compute_garman_klass(grp, window=5)

        # 10) Integrated quartic var (default window=30)
        grp = compute_integrated_quartic_var(grp, window=30)

        # 11) Trade intensity
        grp = compute_trade_intensity(grp)

        # 12) Vol-of-Vol (default vol_col='rv', window=5)
        grp = compute_vov(grp, vol_col="rv", window=5)

        # Collect the transformed DataFrame for this ticker
        dfs.append(grp)

    # Finally, concatenate all per-ticker DataFrames
    df_final = pd.concat(dfs, ignore_index=True)
    # (Optionally) Sort by ticker and t one last time
    df_final.sort_values(["ticker", "t"], inplace=True)
    df_final.reset_index(drop=True, inplace=True)

    return df_final


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


le = LabelEncoder()


def raw_to_enriched(data_cfg):

    df = preprocess(data_cfg)
    df = compute_all_features_simple(df)
    columns_to_drop = ["day", "datetime", "returns"]
    df.drop(columns=columns_to_drop, inplace=True, errors="ignore")
    print("Data types of each column after cleanup:")
    df = drop_n_lowest_times_per_ticker(df, 5)
    df["ticker_id"] = le.fit_transform(df["ticker"])
    print(df.shape)
    save_csv(df, data_cfg["path_processed"])
