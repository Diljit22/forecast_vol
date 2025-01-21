import logging
import logging.config
import yaml
import os
from typing import List
from src.utils.file_io import get_file_paths, load_yaml

logger = logging.getLogger(__name__)


def init_logging(config_path: str = "logging.yaml"):
    """
    Initialize logging from a YAML configuration file.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    logging.config.dictConfig(config)
    print(f"[DEBUG] Logging initialized using '{config_path}'")


def shallow_merge(base_dict: dict, override_dict: dict) -> dict:
    """
    Shallow merge override_dict into base_dict in-place.
    - If a key doesn’t exist in base_dict, it’s added.
    - If both values are dicts, update the sub-dict.
    - Otherwise, override the base_dict value entirely.
    """
    for key, val in override_dict.items():
        if key not in base_dict:
            base_dict[key] = val
        elif isinstance(val, dict) and isinstance(base_dict[key], dict):
            base_dict[key].update(val)
        else:
            base_dict[key] = val
    return base_dict


def load_configs(filepaths: List[str]) -> dict:
    """
    Given a list of YAML file paths:
      - load them in order
      - shallow-merge them into one dict
    (later configs override or merge into earlier ones)
    """
    final_config = {}
    for fp in filepaths:
        logger.info(f"Loading config: {fp}")
        if not os.path.isfile(fp):
            logger.warning(f"Config file not found, skipping: {fp}")
            continue
        cfg = load_yaml(fp)
        shallow_merge(final_config, cfg)
    return final_config


def init_config(config_dir: str = "configs") -> dict:
    """
    Loads and merges all YAML config files in `config_dir`.
    Returns a single dictionary with merged settings.

    :param config_dir: Directory containing YAML config files.
    :return: Merged config dictionary.
    """
    if not os.path.isdir(config_dir):
        logger.warning(f"Config directory '{config_dir}' does not exist.")
        return {}

    # Grab all .yaml files in config_dir (recursively)
    yaml_files = get_file_paths(config_dir, file_format="yaml")

    # Merge them into one config
    final_config = load_configs(yaml_files)

    logger.info(f"Merged config contains {len(final_config)} top-level keys.")
    return final_config
