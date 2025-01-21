"""
Implements GaussianHMM-based regime detection for each ticker's time-series data.
Uses multiprocessing.
"""

import logging
import numpy as np
import pandas as pd
from hmmlearn import hmm
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter

logger = logging.getLogger(__name__)


def _label_single_ticker(args):
    """
    Worker function for parallelizing HMM labeling.
    """
    sym, df, rv_col, n_states, covariance_type, random_state = args

    start_t = perf_counter()
    if rv_col not in df.columns:
        logger.warning(f"{sym}: {rv_col} column not found. Setting regime=0.")
        df["regime"] = 0
        return sym, df

    if len(df) < n_states:
        logger.warning(f"{sym}: Not enough rows for HMM fit; regime=0.")
        df["regime"] = 0
        return sym, df

    data = df[rv_col].values.reshape(-1, 1)
    mask = ~np.isnan(data) & ~np.isinf(data)
    valid_count = np.sum(mask)
    if valid_count < n_states:
        logger.warning(f"{sym}: After filtering NaN/inf, insufficient data; regime=0.")
        df["regime"] = 0
        return sym, df

    try:
        model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type=covariance_type,
            random_state=random_state
        )
        model.fit(data[mask.ravel()])
        # Predict across the entire series
        regimes = model.predict(data)
        df["regime"] = regimes
    except Exception as e:
        logger.error(f"{sym}: HMM fitting failed with error: {e}. regime=0.")
        df["regime"] = 0

    duration = perf_counter() - start_t
    logger.info(f"[HMM] {sym} => {len(df)} rows, took {duration:.2f}s")

    return sym, df


def parallel_hmm_labeling(
    dfs_dict: dict,
    rv_col: str,
    n_states: int = 3,
    covariance_type: str = "full",
    random_state: int = 42,
    max_workers: int = None
) -> dict:
    """
    Labels each ticker's DF with a 'regime' column using GaussianHMM, in parallel.
    """
    tasks = []
    for sym, df in dfs_dict.items():
        tasks.append((sym, df, rv_col, n_states, covariance_type, random_state))

    final_dict = {}
    start_all = perf_counter()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for sym, df_labeled in executor.map(_label_single_ticker, tasks):
            final_dict[sym] = df_labeled
    total_duration = perf_counter() - start_all
    logger.info(f"[HMM] Completed labeling for {len(dfs_dict)} tickers in {total_duration:.2f}s")

    return final_dict
