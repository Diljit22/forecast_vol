
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def inject_ticker_stoch_params(df: pd.DataFrame, stoch_params: dict):
    """
    stoch_params contain:
      {
        'rfsv': {
          'params': {...},
          'simulated_path': np.array([...])
        },
        'garch_params': {...},
        'garch_sim': np.array([...]),
        'multi_scale': {...}
      }
    """
    if not stoch_params:
        logger.warning("No stoch_params for this ticker. Skipping injection.")
        return df

    # 1) RFSV
    if "rfsv" in stoch_params:
        rfsv = stoch_params["rfsv"]
        if "params" in rfsv:
            p = rfsv["params"]
            df["rfsv_hurst"] = p["hurst_exponent"]
            df["rfsv_mu"] = p["mu"]
            df["rfsv_sigma"] = p["sigma"]
            df["rfsv_long_run_var"] = p["long_run_var"]
            df["rfsv_vol_of_vol"] = p["vol_of_vol"]

    # 2) GARCH
    if "garch_params" in stoch_params:
        gp = stoch_params["garch_params"]
        df["garch_omega"] = gp["omega"]
        df["garch_alpha"] = gp["alpha"]
        df["garch_beta"] = gp["beta"]

    # 3) multi_scale
    if "multi_scale" in stoch_params:
        ms = stoch_params["multi_scale"]
        df["fractal_dim"] = ms.get("fractal_dim", 2.0)
        df["fractal_slope"] = ms.get("slope", 0.0)
        df["fractal_intercept"] = ms.get("intercept", 0.0)
    return df


def inject_all_stoch(dfs_stoch: dict) -> pd.DataFrame:
    """
    Merges sub-DataFrames into one final DataFrame, sorted by t.
    """
    final_df = pd.concat(dfs_stoch.values(), ignore_index=True)
    if "t" in final_df.columns:
        final_df.sort_values(["t","ticker"], inplace=True)
    elif "timestamp" in final_df.columns:
        final_df.sort_values(["timestamp","ticker"], inplace=True)
    else:
        logger.warning("No time column found for final sorting.")
    final_df.reset_index(drop=True, inplace=True)
    return final_df
