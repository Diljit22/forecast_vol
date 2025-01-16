import os
import datetime
import pytest
from pathlib import Path

from trash.process_exceptions import (
    load_exceptions_from_yaml,
    save_exceptions_to_python_file,
)

from src.data_ingestion.market_sessions.find_active_sessions import (
    load_exceptions_from_yaml,
    save_exceptions_to_python_file,
)

SAMPLE_YAML = """\
holidays:
  "2023":
    - "2023-01-02"
    - "2023-01-16"
    - "2023-02-20"
    - "2023-04-07"
    - "2023-05-29"
    - "2023-06-19"
    - "2023-07-04"
    - "2023-09-04"
    - "2023-11-23"
    - "2023-12-25"
  "2024":
    - "2024-01-01"
    - "2024-01-15"
    - "2024-02-19"
    - "2024-03-29"
    - "2024-05-27"
    - "2024-06-19"
    - "2024-07-04"
    - "2024-09-02"
    - "2024-11-28"
    - "2024-12-25"

half_days:
  "2023":
    - "2023-07-03"
    - "2023-11-24"
  "2024":
    - "2024-07-03"
    - "2024-11-29"
    - "2024-12-24"
"""


def test_load_exceptions_from_yaml(tmp_path):
    """
    Test that load_exceptions_from_yaml correctly parses the YAML
    to produce HOLIDAYS and HALF_DAYS dicts.
    """
    # 1) Write SAMPLE_YAML to a temp file
    yaml_file = tmp_path / "exceptions.yaml"
    yaml_file.write_text(SAMPLE_YAML, encoding="utf-8")

    # 2) Load the exceptions
    holidays, half_days = load_exceptions_from_yaml(str(yaml_file))

    # 3) Verify the structure for 2023
    assert 2023 in holidays
    assert datetime.date(2023, 1, 2) in holidays[2023]
    assert datetime.date(2023, 12, 25) in holidays[2023]

    assert 2023 in half_days
    assert datetime.date(2023, 7, 3) in half_days[2023]
    assert datetime.date(2023, 11, 24) in half_days[2023]

    # 4) Verify the structure for 2024
    assert 2024 in holidays
    assert datetime.date(2024, 1, 1) in holidays[2024]
    assert datetime.date(2024, 12, 25) in holidays[2024]

    assert 2024 in half_days
    assert datetime.date(2024, 7, 3) in half_days[2024]
    assert datetime.date(2024, 11, 29) in half_days[2024]


def test_save_exceptions_to_python_file(tmp_path):
    """
    Test that save_exceptions_to_python_file generates a Python file
    with the correct HOLIDAYS and HALF_DAYS definitions.
    """
    # 1) Write SAMPLE_YAML to a temp file
    yaml_file = tmp_path / "exceptions.yaml"
    yaml_file.write_text(SAMPLE_YAML, encoding="utf-8")

    # 2) Define output Python file path
    out_py = tmp_path / "output_exceptions.py"

    # 3) Convert the YAML to a Python file
    save_exceptions_to_python_file(str(yaml_file), str(out_py))
    assert out_py.exists(), "Output Python file not created."

    # 4) Read the Python file and do basic checks
    content = out_py.read_text(encoding="utf-8")

    # Check for presence of 'datetime.date(2023, 1, 2)', etc.
    assert "import datetime" in content
    # Spot checks for a couple of holiday dates
    assert "datetime.date(2023, 1, 16)" in content
    assert "datetime.date(2024, 12, 25)" in content
    # Spot checks for half-day dates
    assert "datetime.date(2023, 7, 3)" in content
    assert "datetime.date(2024, 11, 29)" in content
