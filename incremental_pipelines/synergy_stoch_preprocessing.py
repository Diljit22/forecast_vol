
from src.multi_asset_synergy.run_hierarchical_nn import run_synergy_pipeline
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import logging
logging.basicConfig(
level=logging.INFO,
format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
datefmt='%Y-%m-%d %H:%M:%S'
)
from src.stochastic_models.run_stochastic_models import run_stochastic_pipeline

def inject_syn_stock(config):
    
    df_partial = pd.read_csv("data/intermediate/preprocessed.csv")
    logging.info("Starting synergy_pipeline...")
    run_synergy_pipeline(df_partial, config)
    logging.info("synergy_pipeline completed.")


    logging.info("Starting stoch...")
    df_synergy = pd.read_csv("data/intermediate/synergy_enriched.csv")
    df_stoch = run_stochastic_pipeline(df_synergy, config)
    logging.info("stoch completed.")

    return df_stoch
