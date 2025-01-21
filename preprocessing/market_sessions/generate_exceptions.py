import os
import datetime
import logging
from src.utils.file_io import load_yaml

logger = logging.getLogger(__name__)


def load_exceptions(yaml_path: str):
    """
    Parses a YAML file containing 'holidays' and 'half_days' sections and returns two dictionaries:

    HOLIDAYS = {
        year: {datetime.date(...), ...},
        ...
    }

    HALF_DAYS = {
        year: {datetime.date(...), ...},
        ...
    }
    """
    data = load_yaml(yaml_path)

    holidays = {}
    half_days = {}

    # Populate holidays
    if "holidays" in data:
        for year_str, dates in data["holidays"].items():
            year = int(year_str)
            holidays[year] = set()
            for date_str in dates:
                dt = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
                holidays[year].add(dt)

    # Populate half_days
    if "half_days" in data:
        for year_str, dates in data["half_days"].items():
            year = int(year_str)
            half_days[year] = set()
            for date_str in dates:
                dt = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
                half_days[year].add(dt)

    logger.info(
        f"Found {len(holidays)} holiday-year entries and {len(half_days)} half-day-year entries."
    )
    return holidays, half_days


def _dict_to_python_lines(data_dict: dict, var_name: str) -> list[str]:
    """
    Helper function to convert a {year: set_of_date_objects} dictionary
    into lines of Python code defining that dictionary.
    Example:
        2023: {
            datetime.date(2023, 1, 1),
            ...
        }
    """
    lines = [f"{var_name} = {{\n"]
    for year, date_set in sorted(data_dict.items()):
        lines.append(f"    {year}: {{\n")
        for d in sorted(date_set):
            lines.append(f"        datetime.date({d.year}, {d.month}, {d.day}),\n")
        lines.append("    },\n")
    lines.append("}\n\n")
    return lines


def save_exceptions_to_python_file(yaml_path: str, out_filepath: str) -> None:
    """
    - Loads the 'holidays' and 'half_days' sections from the YAML file at `yaml_path`.
    - Writes them into a Python file at `out_filepath`, defining:

        import datetime

        HOLIDAYS = {
            2023: {
                datetime.date(2023, 1, 2),
                ...
            },
            ...
        }

        HALF_DAYS = {
            ...
        }
    """
    logger.info(f"Converting exceptions from '{yaml_path}' -> '{out_filepath}'")
    holidays, half_days = load_exceptions(yaml_path)

    # Convert the dictionary data into lines of Python code
    lines = ["import datetime\n\n"]
    lines.extend(_dict_to_python_lines(holidays, "HOLIDAYS"))
    lines.extend(_dict_to_python_lines(half_days, "HALF_DAYS"))

    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_filepath), exist_ok=True)

    # Write the file
    with open(out_filepath, "w", encoding="utf-8") as py_file:
        py_file.writelines(lines)

    logger.info(f"Exceptions written to Python file: {out_filepath}")
