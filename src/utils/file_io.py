# src/utils/file_io.py
import os
import glob
import logging
import yaml
import pandas as pd
import pyarrow.dataset as ds
from pprint import pformat
from collections import defaultdict
from typing import List, Dict, Optional, Union

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
#  Directory Handling
# --------------------------------------------------------------------------


def ensure_dir_exists(dir_path: str) -> None:
    """
    If the directory doesn't exist, create it and log/print a message.
    """
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")
    else:
        logger.debug(f"Directory already exists: {dir_path}")


def build_file_path(base_dir: str, filename: str, extension: str = None) -> str:
    """
    - Build a file path from a base directory + filename + extension.
    - Example: build_file_path("data/out", "AAPL_1m", "csv") => "data/out/AAPL_1m.csv"
    - Example: build_file_path("data/out", "AAPL_1m.csv") => "data/out/AAPL_1m.csv"
    """
    ensure_dir_exists(base_dir)

    if extension:
        ext = extension if extension.startswith(".") else f".{extension}"
        full_filename = f"{filename}{ext}"
    else:
        full_filename = filename

    full_path = os.path.join(base_dir, full_filename)
    logger.debug(f"Built file path: {full_path}")
    return full_path


# --------------------------------------------------------------------------
#  LOADING FUNCTIONS
# --------------------------------------------------------------------------


def load_yaml(filepath: str) -> dict:
    logger.debug(f"Loading YAML: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_csv(filepath: str, **kwargs) -> pd.DataFrame:
    logger.debug(f"Loading CSV: {filepath}")
    return pd.read_csv(filepath, **kwargs)


def load_parquet(filepath: str, **kwargs) -> pd.DataFrame:
    logger.debug(f"Loading Parquet: {filepath}")
    return pd.read_parquet(filepath, **kwargs)


def load_file_auto(filepath: str, **kwargs) -> Union[Dict, pd.DataFrame]:
    """
    - Detect file type (yaml, csv, parquet) from extension and load accordingly.
    - Returns either a dict (for YAML) or DataFrame (for CSV/Parquet).
    """
    ext = os.path.splitext(filepath)[1].lower()

    # Map file extension to loader function
    loaders = {
        ".yaml": load_yaml,
        ".yml": load_yaml,
        ".csv": load_csv,
        ".parquet": load_parquet,
    }

    if ext in loaders:
        logger.debug(f"Auto-loading {ext} file: {filepath}")
        return loaders[ext](filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


# --------------------------------------------------------------------------
#  SAVING FUNCTIONS
# --------------------------------------------------------------------------


def save_csv(df: pd.DataFrame, output_path: str, **kwargs) -> None:
    """
    - Saves a DataFrame to CSV at the given path
    - Accepts extra kwargs for df.to_csv.
    """
    ensure_dir_exists(os.path.dirname(output_path))
    logger.info(f"Saving DataFrame to CSV: {output_path}")

    df.to_csv(output_path, index=False, **kwargs)
    logger.info("CSV save complete.")


def save_parquet(
    df: pd.DataFrame,
    output_path: str,
    also_save_csv=False,
    csv_artifacts_dir="artifacts/parquet_previews",
) -> None:
    """
    - Saves a DataFrame to Parquet at the given path.
    - Optionally also saves a CSV copy
    """
    ensure_dir_exists(os.path.dirname(output_path))
    logger.info(f"Saving DataFrame to Parquet: {output_path}")

    df.to_parquet(output_path, index=False)
    logger.info("Parquet save complete.")

    # If also_save_csv is True, save a CSV copy in `csv_artifacts_dir`.
    if also_save_csv:
        base_name = os.path.splitext(os.path.basename(output_path))[0]
        csv_path = build_file_path(csv_artifacts_dir, base_name, "csv")
        save_csv(df, csv_path)


# --------------------------------------------------------------------------
#  PYTHON FILE (DICT) SAVERS
# --------------------------------------------------------------------------


def save_dict_to_python_file(
    data_dict: dict,
    file_path: str,
    var_name: str = "MY_DICT",
    import_statements: list[str] = None,
) -> None:
    """
    Saves a Python dictionary to a .py file as a variable declaration.
    """
    ensure_dir_exists(os.path.dirname(file_path))
    logger.debug(f"Saving dict to Python file: {file_path}")

    lines = []
    if import_statements:
        for imp in import_statements:
            lines.append(f"import {imp}\n")
        lines.append("\n")

    dict_str = pformat(data_dict, sort_dicts=True)
    lines.append(f"{var_name} = {dict_str}\n\n")

    with open(file_path, "w", encoding="utf-8") as py_file:
        py_file.writelines(lines)

    logger.info(f"Dictionary '{var_name}' saved to {file_path}.")


# --------------------------------------------------------------------------
#  FILE SEARCH
# --------------------------------------------------------------------------


def get_file_paths(folderpath: str, file_format: str = "csv") -> list[str]:
    """
    - Return all files (recursively) in folderpath that match the
        given extension.
    - e.g. get_file_paths("data/raw", "csv") -> ["data/raw/file1.csv", ...]
    """
    logger.debug(f"Searching for '*.{file_format}' in {folderpath}")
    pattern = os.path.join(folderpath, "**", f"*.{file_format}")
    all_files = glob.glob(pattern, recursive=True)
    logger.info(f"Found {len(all_files)} '*.{file_format}' files.")
    return all_files


def get_info(filepath_parquet: str, num_rows: int = 10, show_head: bool = True) -> None:
    """
    Displays metadata for a Parquet file using pyarrow.dataset:
      - shape
      - # of NaN in each column
      - column names
      - optional head of the data
    """
    logger.info(f"Inspecting up to {num_rows} rows from: {filepath_parquet}")
    dataset = ds.dataset(filepath_parquet, format="parquet")
    table = dataset.head(num_rows)
    df = table.to_pandas()

    logger.info(f"DataFrame shape: {df.shape}")
    logger.info(f"NaN counts: \n{df.isna().sum()}")
    logger.info(f"Columns: {list(df.columns)}")

    if show_head:
        logger.debug(f"Head:\n{df.head()}")
