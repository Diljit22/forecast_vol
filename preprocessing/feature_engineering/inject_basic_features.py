from src.preprocessing.feature_engineering.basic_features import *

def compute_all_features_simple(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply a fixed sequence of feature transformations to each ticker. 
    Returns a single DataFrame with all the new columns added.
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

    # Concatenate all per-ticker DataFrames
    df_final = pd.concat(dfs, ignore_index=True)
    df_final.sort_values(["ticker", "t"], inplace=True)
    df_final.reset_index(drop=True, inplace=True)

    return df_final