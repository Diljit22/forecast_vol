from pipelines.build_final_model import main_launch, main_setup
from src.utils.cfg import init_config
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
config = init_config()
import multiprocessing

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    config, train_ds, val_ds, test_ds = main_setup()
    main_launch(config, train_ds, val_ds, test_ds)