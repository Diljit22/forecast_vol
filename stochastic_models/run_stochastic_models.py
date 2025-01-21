
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any
from time import perf_counter
from concurrent.futures import ProcessPoolExecutor
from arch import arch_model

from .rfsv_model import fit_rfsv_params, simulate_rough_vol_path
from .garch_model import fit_garch_params, simulate_garch_path
from .fractal_analysis import multi_scale_volatility_estimate
from .inject_stochastic import inject_ticker_stoch_params, inject_all_stoch

logger = logging.getLogger(__name__)


def _process_one_ticker(args):
    """
    fits RFSV, GARCH, fractal for a single ticker's DF
    and injects columns. Returns (sym, df_updated).
    """
    sym, df, stoch_cfg = args
    start_t = perf_counter()

    rv_col = "rv"
    if rv_col not in df.columns:
        logger.warning(f"{sym}: No {rv_col} column found, skipping stoch. Returning df unchanged.")
        return sym, df

    rv_array = df[rv_col].values
    wavelet_name = stoch_cfg.get("wavelet_name", "haar")
    levels = stoch_cfg.get("wavelet_levels", 6)
    random_restarts = stoch_cfg.get("random_restarts", 3)
    rfsv_params = fit_rfsv_params(rv_array, wavelet=wavelet_name, levels=levels, random_restarts=random_restarts)
    rfsv_sim = simulate_rough_vol_path(rfsv_params, n_steps=stoch_cfg.get("n_steps_sim",100))  # we do it anyway

    returns_col = "log_return"
    returns_array = df[returns_col].values if returns_col in df.columns else np.zeros(len(df))
    garch_params = fit_garch_params(returns_array)
    garch_sim = simulate_garch_path(garch_params, n_steps=stoch_cfg.get("n_steps_sim", 100))

    scales = stoch_cfg.get("fractal_scales", [2,4,8])
    fractal_res = multi_scale_volatility_estimate(rv_array, wavelet=wavelet_name, scales=scales)

    stoch_out = {
        "rfsv": {"params": rfsv_params, "simulated_path": rfsv_sim},
        "garch_params": garch_params,
        "garch_sim": garch_sim,
        "multi_scale": fractal_res
    }
    df_updated = inject_ticker_stoch_params(df, stoch_out)

    duration = perf_counter() - start_t
    logger.info(f"[Stoch] {sym} => rows={len(df)} took {duration:.2f}s")
    return sym, df_updated


def run_stochastic_pipeline(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Returns final enriched DataFrame.
    """
    stoch_cfg = config["stochastic"]
    logger.info("[Stochastic] Starting pipeline with no skip steps...")

    start_all = perf_counter()
    # Identify ticker columns for splitting
    ticker_cols = [c for c in df.columns if c.startswith("ticker_")]

    dropped_ticker = "AAPL"  # or from config
    # Assign the dropped ticker if not assigned
    if "ticker" not in df.columns:
        df["ticker"] = None
        for idx, row in df.iterrows():
            s = sum(row[col] for col in ticker_cols)
            if s == 0:
                df.at[idx, "ticker"] = dropped_ticker
            else:
                for col in ticker_cols:
                    if row[col] == 1:
                        sym = col.replace("ticker_", "")
                        df.at[idx, "ticker"] = sym
                        break

    # Now split by ticker
    dfs_dict = {}
    for sym in df["ticker"].unique():
        sub_df = df[df["ticker"] == sym].copy()
        dfs_dict[sym] = sub_df

    # Parallel process
    tasks = []
    for sym, sub_df in dfs_dict.items():
        tasks.append((sym, sub_df, stoch_cfg))

    final_dict = {}
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor() as executor:
        results = executor.map(_process_one_ticker, tasks)
        for sym, df_stoch in results:
            final_dict[sym] = df_stoch

    # Merge
    final_df = pd.concat(final_dict.values(), ignore_index=True)
    if "t" in final_df.columns:
        final_df.sort_values(["t","ticker"], inplace=True)
    elif "timestamp" in final_df.columns:
        final_df.sort_values(["timestamp","ticker"], inplace=True)
    final_df.reset_index(drop=True, inplace=True)

    # Save final
    out_path = stoch_cfg.get("output_path", "data/intermediate/synergy_stoch_enriched.csv")
    final_df.to_csv(out_path, index=False)
    logger.info(f"[Stochastic] Wrote final data to {out_path}")

    total_duration = perf_counter() - start_all
    logger.info(f"[Stochastic] Pipeline completed in {total_duration:.2f}s")
    return final_df

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
    from src.stochastic_models.run_stochastic_models import run_stochastic_pipeline

    logging.info("Starting preprocessing...")
    df_synergy = pd.read_csv("data/intermediate/synergy_enriched.csv")  # after synergy
    df_stoch = run_stochastic_pipeline(df_synergy, cfg)
    logging.info("Preprocessing completed.")