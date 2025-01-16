import os
import glob
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def get_config_hash(config: dict) -> str:
    """
    Convert relevant config sections to string and hash them.
    """
    relevant_sections = {}
    for key in ["data", "feature_engineering", "multi_asset", "stability", "cache"]:
        if key in config:
            relevant_sections[key] = config[key]

    config_str = json.dumps(relevant_sections, sort_keys=True)
    return hashlib.md5(config_str.encode("utf-8")).hexdigest()


def get_data_modtime_hash(raw_data_path: str) -> str:
    """
    Generate a combined hash from last-modified times of all CSV files in 'raw_data_path'.
    """
    file_modtimes = []
    all_csv_files = glob.glob(
        os.path.join(raw_data_path, "**", "*.csv"), recursive=True
    )
    for filepath in sorted(all_csv_files):
        mod_time = os.path.getmtime(filepath)  # last mod time
        file_modtimes.append(f"{filepath}:{mod_time}")

    # Combine modtime strings and hash them
    modtime_str = "|".join(file_modtimes)
    return hashlib.md5(modtime_str.encode("utf-8")).hexdigest() if file_modtimes else ""


def load_cache_metadata(metadata_path: str) -> dict:
    """
    Loads the JSON cache metadata if exists, else returns empty dict.
    """
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not read cache metadata: {e}")
            return {}
    return {}


def save_cache_metadata(metadata_path: str, config_hash: str, data_hash: str) -> None:
    """
    Writes a JSON with config_hash, data_hash, timestamp. Overwrites any existing file.
    """
    meta = {
        "config_hash": config_hash,
        "data_hash": data_hash,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }
    Path(os.path.dirname(metadata_path)).mkdir(parents=True, exist_ok=True)
    try:
        with open(metadata_path, "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Cache metadata saved to {metadata_path}")
    except Exception as e:
        logger.warning(f"Could not save cache metadata: {e}")


def is_cache_valid(config: dict) -> bool:
    """
    Main entry point to check if the cache is valid:
    """
    if not config.get("cache", {}).get("enabled", False):
        return False

    metadata_dir = config["cache"].get("metadata_dir", "cache_metadata")
    metadata_file = os.path.join(metadata_dir, "cache_meta.json")

    old_meta = load_cache_metadata(metadata_file)
    if not old_meta:
        logger.info("No existing cache metadata found.")
        return False

    # Compute current config hash
    curr_conf_hash = get_config_hash(config)

    if config["cache"].get("check_data_modtime", True):
        raw_data_path = config["data"].get("path_raw", "data/raw")
        curr_data_hash = get_data_modtime_hash(raw_data_path)
    else:
        curr_data_hash = ""

    # Compare with old_meta
    if curr_conf_hash == old_meta.get(
        "config_hash", ""
    ) and curr_data_hash == old_meta.get("data_hash", ""):
        logger.info("Cache is valid. Config/data hash matches.")
        return True
    logger.info("Cache is invalid. Config/data changed.")
    return False


def update_cache_record(config: dict) -> None:
    """
    After successful pipeline run, update the metadata to
    reflect new config/data signatures.
    """
    if not config.get("cache", {}).get("enabled", False):
        return

    metadata_dir = config["cache"].get("metadata_dir", "cache_metadata")
    metadata_file = os.path.join(metadata_dir, "cache_meta.json")

    new_conf_hash = get_config_hash(config)
    if config["cache"].get("check_data_modtime", True):
        raw_data_path = config["data"].get("path_raw", "data/raw")
        new_data_hash = get_data_modtime_hash(raw_data_path)
    else:
        new_data_hash = ""

    save_cache_metadata(metadata_file, new_conf_hash, new_data_hash)
