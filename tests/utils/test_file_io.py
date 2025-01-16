# tests/test_file_io.py
import os
import pytest
import pandas as pd
import yaml
from pathlib import Path

from src.utils.file_io import (
    build_file_path,
    save_dict_to_python_file,
    get_file_paths,
)


@pytest.fixture
def sample_df():
    """Returns a small sample DataFrame for testing."""
    return pd.DataFrame({"col1": [1, 2, 3], "col2": ["A", "B", "C"]})


def test_build_file_path(tmp_path):
    """Check that build_file_path returns a valid path and creates the dir."""
    base_dir = tmp_path / "some_base"
    filename = "test_file"
    extension = "csv"

    full_path = build_file_path(str(base_dir), filename, extension)
    assert str(full_path).endswith("test_file.csv"), "Incorrect file path end."
    assert (
        base_dir.exists()
    ), "Base directory should have been created by ensure_dir_exists."


def test_save_dict_to_python_file(tmp_path):
    """Test saving a dict to a Python file."""
    test_dict = {"alpha": 1, "beta": [1, 2, 3]}
    out_file = tmp_path / "dict_output.py"
    var_name = "TEST_DICT"
    save_dict_to_python_file(
        test_dict, str(out_file), var_name=var_name, import_statements=["datetime"]
    )

    # Verify the file’s contents
    with open(out_file, "r", encoding="utf-8") as f:
        content = f.read()
    assert "import datetime" in content
    assert f"{var_name} = " in content
    assert "alpha" in content
    assert "beta" in content


def test_get_file_paths(tmp_path):
    """Test retrieving file paths of a certain extension."""
    csv_dir = tmp_path / "csv_files"
    csv_dir.mkdir()
    (csv_dir / "file1.csv").touch()
    (csv_dir / "file2.csv").touch()
    (csv_dir / "file3.txt").touch()  # not csv
    found_paths = get_file_paths(str(csv_dir), file_format="csv")
    assert len(found_paths) == 2
    assert all(p.endswith(".csv") for p in found_paths), "Should only find CSV files."
