import importlib.util
import os


def load_dict_from_python_file(py_file_path: str, dict_name: str):
    """
    Dynamically load a dictionary named `dict_name` from a Python file at `py_file_path`.
    """
    if not os.path.isfile(py_file_path):
        raise FileNotFoundError(f"No such file: {py_file_path}")

    spec = importlib.util.spec_from_file_location("temp_module", py_file_path)
    temp_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(temp_module)

    # Access the dictionary by name
    return getattr(temp_module, dict_name)


def generate_active_sessions(config: dict) -> None:
    from src.data_ingestion.market_sessions.find_active_sessions import (
        save_active_market_ranges,
    )
    from src.data_ingestion.market_sessions.generate_exceptions import (
        save_exceptions_to_python_file,
    )
    import datetime

    ms_cfg = config["market_sessions"]

    save_exceptions_to_python_file(
        ms_cfg["path_exceptions_yaml"], ms_cfg["path_exceptions_dict"]
    )
    HOLIDAYS = load_dict_from_python_file(ms_cfg["path_exceptions_dict"], "HOLIDAYS")
    HALF_DAYS = load_dict_from_python_file(ms_cfg["path_exceptions_dict"], "HALF_DAYS")
    save_active_market_ranges(
        ms_cfg["start_date"],
        ms_cfg["end_date"],
        HOLIDAYS,
        HALF_DAYS,
        ms_cfg["path_market_hours"],
    )
