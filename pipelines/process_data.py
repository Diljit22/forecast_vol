
from incremental_pipelines.populate_market_minutes import generate_active_sessions
from incremental_pipelines.synergy_stoch_preprocessing import inject_syn_stock
from incremental_pipelines.basic_prepocessing import preprocess

import logging
logger = logging.getLogger(__name__)

from incremental_pipelines.basic_prepocessing import preprocess

def process_pipeline(config):
    generate_active_sessions(config)
    
    logging.info("Starting preprocessing...")
    preprocess(config)
    logging.info("Preprocessing completed.")

    df = inject_syn_stock(config)
    df = df.select_dtypes(include=["number", "bool"])  # or include=[np.number, "bool"]
    df = df.drop(columns=["ticker"], errors="ignore")  # drop object/strings
    # Then convert bool->float
    for col in df.select_dtypes(["bool"]).columns:
        df[col] = df[col].astype(float)

    out_path = config["data"].get("path_final", "data/final/enriched.csv")
    df.to_csv(out_path, index=False)
    logger.info(f"[Full Process] Wrote final data to {out_path}")

if __name__ == "__main__":
    
    from src.utils.cfg import init_config
    import pandas as pd
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)
    
    config = init_config()
    process_pipeline(config)