import pytest
import pandas as pd
import arnio as ar
from pathlib import Path


def test_select_dtypes_include_single_dtype(tmp_path):
    """Test selecting columns by including a single dtype."""
    csv_file = tmp_path / "test.csv"
    df = pd.DataFrame({
        "int_col": [1, 2, 3],
        "float_col": [1.5, 2.5, 3.5],
        "str_col": ["a", "b", "c"]
    })
    df.to_csv(csv_file, index=False)
    
    frame = ar.read_csv(str(csv_file))
    result = frame.select_dtypes(include="int64")
    
    assert len(result.columns) == 1
    assert "int_col" in result.columns
    assert "float_col" not in result.columns
    assert "str_col" not in result.columns


def test_select_dtypes_include_multiple_dtypes_list(tmp_path):
    """Test selecting columns by including multiple dtypes as a list."""
    csv_file = tmp_path / "test.csv"
    df = pd.DataFrame({
        "int_col": [1, 2, 3],
        "float_col": [1.5, 2.5, 3.5],
        "str_col": ["a", "b", "c"],
        "bool_col": [True, False, True]
    })
    df.to_csv(csv_file, index=False)
    
    frame = ar.read_csv(str(csv_file))
    result = frame.select_dtypes(include=["int64", "float64"])
    
    assert len(result.columns) == 2
    assert "int_col" in result.columns
    assert "float_col" in result.columns


def test_select_dtypes_include_multiple_dtypes_tuple(tmp_path):
    """Test selecting columns by including multiple dtypes as a tuple."""
    csv_file = tmp_path / "test.csv"
    df = pd.DataFrame({
        "int_col": [1, 2, 3],
        "float_col": [1.5, 2.5, 3.5],
        "str_col": ["a", "b", "c"]
    })
    df.to_csv(csv_file, index=False)
    
    frame = ar.read_csv(str(csv_file))
    result = frame.select_dtypes(include=("int64", "float64"))
    
    assert len(result.columns) == 2
    assert "int_col" in result.columns
    assert "float_col" in result.columns


def test_select_dtypes_exclude_single_dtype(tmp_path):
    """Test excluding columns by a single dtype."""
    csv_file = tmp_path / "test.csv"
    df = pd.DataFrame({
        "int_col": [1, 2, 3],
        "float_col": [1.5, 2.5, 3.5],
        "str_col": ["a", "b", "c"]
    })
    df.to_csv(csv_file, index=False)
    
    frame = ar.read_csv(str(csv_file))
    result = frame.select_dtypes(exclude="string")
    
    assert len(result.columns) == 2
    assert "int_col" in result.columns
    assert "float_col" in result.columns
    assert "str_col" not in result.columns


def test_select_dtypes_exclude_multiple_dtypes(tmp_path):
    """Test excluding columns by multiple dtypes."""
    csv_file = tmp_path / "test.csv"
    df = pd.DataFrame({
        "int_col": [1, 2, 3],
        "float_col": [1.5, 2.5, 3.5],
        "str_col": ["a", "b", "c"],
        "bool_col": [True, False, True]
    })
    df.to_csv(csv_file, index=False)
    
    frame = ar.read_csv(str(csv_file))
    result = frame.select_dtypes(exclude=["string", "bool"])
    
    assert len(result.columns) == 2
    assert "int_col" in result.columns
    assert "float_col" in result.columns


def test_select_dtypes_include_and_exclude_no_overlap(tmp_path):
    """Test using both include and exclude without overlapping dtypes."""
    csv_file = tmp_path / "test.csv"
    df = pd.DataFrame({
        "int_col": [1, 2, 3],
        "float_col": [1.5, 2.5, 3.5],
        "str_col": ["a", "b", "c"],
        "bool_col": [True, False, True]
    })
    df.to_csv(csv_file, index=False)
    
    frame = ar.read_csv(str(csv_file))
    result = frame.select_dtypes(include=["int64", "float64", "bool"], exclude="bool")
    
    assert len(result.columns) == 2
    assert "int_col" in result.columns
    assert "float_col" in result.columns
    assert "bool_col" not in result.columns


def test_select_dtypes_no_include_no_exclude_raises_error(tmp_path):
    """Test that ValueError is raised when neither include nor exclude is provided."""
    csv_file = tmp_path / "test.csv"
    df = pd.DataFrame({"col": [1, 2, 3]})
    df.to_csv(csv_file, index=False)
    
    frame = ar.read_csv(str(csv_file))
    
    with pytest.raises(ValueError, match="select_dtypes.*requires at least one"):
        frame.select_dtypes()


def test_select_dtypes_include_exclude_overlap_raises_error(tmp_path):
    """Test that ValueError is raised when include and exclude overlap."""
    csv_file = tmp_path / "test.csv"
    df = pd.DataFrame({
        "int_col": [1, 2, 3],
        "float_col": [1.5, 2.5, 3.5]
    })
    df.to_csv(csv_file, index=False)
    
    frame = ar.read_csv(str(csv_file))
    
    with pytest.raises(ValueError, match="overlap"):
        frame.select_dtypes(include=["int64", "float64"], exclude="int64")


def test_select_dtypes_invalid_dtype_raises_error(tmp_path):
    """Test that ValueError is raised for unrecognised dtype strings."""
    csv_file = tmp_path / "test.csv"
    df = pd.DataFrame({"col": [1, 2, 3]})
    df.to_csv(csv_file, index=False)
    
    frame = ar.read_csv(str(csv_file))
    
    with pytest.raises(ValueError, match="Unrecognised dtype"):
        frame.select_dtypes(include="invalid_dtype")


def test_select_dtypes_invalid_type_include_raises_error(tmp_path):
    """Test that TypeError is raised for invalid type in include."""
    csv_file = tmp_path / "test.csv"
    df = pd.DataFrame({"col": [1, 2, 3]})
    df.to_csv(csv_file, index=False)
    
    frame = ar.read_csv(str(csv_file))
    
    with pytest.raises(TypeError, match="'include' must be a string, list, or tuple"):
        frame.select_dtypes(include=123)


def test_select_dtypes_non_string_in_list_raises_error(tmp_path):
    """Test that TypeError is raised for non-string elements in include list."""
    csv_file = tmp_path / "test.csv"
    df = pd.DataFrame({"col": [1, 2, 3]})
    df.to_csv(csv_file, index=False)
    
    frame = ar.read_csv(str(csv_file))
    
    with pytest.raises(TypeError, match="'include' must contain only strings"):
        frame.select_dtypes(include=["int64", 123])


def test_select_dtypes_no_matching_columns(tmp_path):
    """Test that an empty ArFrame is returned when no columns match."""
    csv_file = tmp_path / "test.csv"
    df = pd.DataFrame({
        "str_col1": ["a", "b", "c"],
        "str_col2": ["x", "y", "z"]
    })
    df.to_csv(csv_file, index=False)
    
    frame = ar.read_csv(str(csv_file))
    result = frame.select_dtypes(include="int64")
    
    assert len(result.columns) == 0
    assert len(result) == len(frame)


def test_select_dtypes_preserves_column_order(tmp_path):
    """Test that column order is preserved in the result."""
    csv_file = tmp_path / "test.csv"
    df = pd.DataFrame({
        "z_int": [1, 2, 3],
        "a_float": [1.5, 2.5, 3.5],
        "m_int": [4, 5, 6],
        "b_string": ["x", "y", "z"]
    })
    df.to_csv(csv_file, index=False)
    
    frame = ar.read_csv(str(csv_file))
    result = frame.select_dtypes(include=["int64", "float64"])
    
    expected_columns = ["z_int", "a_float", "m_int"]
    assert list(result.columns) == expected_columns


def test_select_dtypes_returns_arframe_and_writable(tmp_path):
    """Test that select_dtypes returns an ArFrame that can be written to CSV."""
    csv_file = tmp_path / "test.csv"
    df = pd.DataFrame({
        "int_col": [1, 2, 3],
        "float_col": [1.5, 2.5, 3.5]
    })
    df.to_csv(csv_file, index=False)
    
    frame = ar.read_csv(str(csv_file))
    result = frame.select_dtypes(include="int64")
    
    output_file = tmp_path / "output.csv"
    result.to_csv(str(output_file))
    
    assert output_file.exists()
    
    reloaded = ar.read_csv(str(output_file))
    assert "int_col" in reloaded.columns
    assert len(reloaded) == 3
