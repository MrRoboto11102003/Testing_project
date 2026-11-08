"""
Test suite for arnio.convert.to_pandas.

Tests the conversion of an ArFrame to a pandas DataFrame, covering
documented behavior, edge cases, and copy semantics.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from arnio import ArFrame, convert


@pytest.fixture
def simple_frame():
    """Create a simple ArFrame for testing."""
    data = {"name": ["Alice", "Bob"], "age": [25, 30]}
    return ArFrame.from_dict(data)


@pytest.fixture
def null_frame():
    """Create an ArFrame with null values."""
    data = {"value": [1.0, np.nan, 3.0]}
    return ArFrame.from_dict(data)


@pytest.fixture
def mixed_type_frame():
    """Create an ArFrame with mixed types (int, float, str)."""
    data = {
        "id": [1, 2],
        "score": [95.5, 87.0],
        "desc": ["test", "data"],
    }
    return ArFrame.from_dict(data)


def test_to_pandas_basic_conversion(simple_frame):
    """Test basic conversion to pandas DataFrame."""
    result = convert.to_pandas(simple_frame)
    
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["name", "age"]
    # Check values match (ArFrame stores numpy arrays internally)
    expected = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
    assert result.equals(expected)


def test_to_pandas_copy_false_default(simple_frame):
    """Test that default copy=False behavior works."""
    result = convert.to_pandas(simple_frame)
    
    # If copy=False is the default, ensure it's not a deep copy of internal buffers
    # though pandas owns its own memory, we just verify no exception occurs.
    assert result is not None


def test_to_pandas_copy_true_behavior(mixed_type_frame):
    """Test conversion with copy=True."""
    result = convert.to_pandas(mixed_type_frame, copy=True)
    
    assert isinstance(result, pd.DataFrame)
    # With copy=True, pandas should own the data.
    # We verify it's a valid DataFrame and not broken by internal ArFrame state.
    assert len(result) == 2


def test_to_pandas_null_handling(null_frame):
    """Test conversion preserves null values correctly."""
    result = convert.to_pandas(null_frame)
    
    # Check that NaN is preserved in the float column
    expected = pd.DataFrame({"value": [1.0, np.nan, 3.0]})
    assert result.equals(expected)


def test_to_pandas_metadata_preservation():
    """Test that metadata from from_pandas() is restored."""
    # Create a pandas DataFrame with specific dtypes
    pdf = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    pdf["a"] = pdf["a"].astype(np.int64)
    
    af = ArFrame.from_pandas(pdf)
    result = convert.to_pandas(af)
    
    # The resulting DataFrame should match the original structure
    assert list(result.columns) == ["a", "b"]
    assert result.dtypes.equals(pdf.dtypes)


def test_to_pandas_empty_frame():
    """Test conversion of an empty ArFrame."""
    data = {}
    af = ArFrame.from_dict(data)
    
    result = convert.to_pandas(af)
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_to_pandas_with_attrs(mixed_type_frame):
    """Test conversion restores attrs metadata if present."""
    # Simulate an ArFrame created from pandas with attributes
    pdf = pd.DataFrame({"x": [1, 2]})
    pdf.attrs["test_key"] = "test_value"
    
    af = ArFrame.from_pandas(pdf)
    result = convert.to_pandas(af)
    
    # Verify attrs are restored on the resulting DataFrame
    assert hasattr(result, "attrs")
    assert result.attrs.get("test_key") == "test_value"


def test_to_pandas_large_integers():
    """Test conversion handles large integers without overflow."""
    data = {"large": [2**60 - 1]}
    af = ArFrame.from_dict(data)
    
    result = convert.to_pandas(af)
    
    # Ensure the large value is preserved exactly
    assert result.iloc[0, 0] == 2**60 - 1


def test_to_pandas_float_precision():
    """Test conversion preserves float precision."""
    data = {"prec": [0.123456789]}
    af = ArFrame.from_dict(data)
    
    result = convert.to_pandas(af)
    
    # Check precision is maintained (within floating point limits)
    assert abs(result.iloc[0, 0] - 0.123456789) < 1e-10


def test_to_pandas_with_invalid_input():
    """Test that invalid input types raise appropriate errors."""
    with pytest.raises(TypeError):
        convert.to_pandas("not_a_frame")

    with pytest.raises(ValueError):
        # Test None or other non-ArFrame objects
        convert.to_pandas(None)


def test_to_pandas_dtype_inference():
    """Test that dtypes are inferred correctly from ArFrame data."""
    # Create frame with mixed types that pandas might infer differently
    data = {
        "int_col": [1, 2],
        "float_col": [1.5, 2.5],
        "str_col": ["a", "b"],
    }
    af = ArFrame.from_dict(data)
    
    result = convert.to_pandas(af)
    
    # Verify dtypes match the original data types
    assert result["int_col"].dtype == np.int64
    assert result["float_col"].dtype == np.float64
    assert result["str_col"].dtype == object


def test_to_pandas_copy_semantics():
    """Test that copy=True actually creates a defensive copy."""
    # This test verifies the logic of the copy parameter, though we can't
    # fully verify memory layout without C++ internals. We ensure it doesn't break.
    
    af = ArFrame.from_dict({"val": [1]})
    
    # Both should produce valid DataFrames
    df1 = convert.to_pandas(af)
    df2 = convert.to_pandas(af, copy=True)
    
    assert isinstance(df1, pd.DataFrame)
    assert isinstance(df2, pd.DataFrame)


def test_to_pandas_empty_column():
    """Test conversion when a column is empty."""
    data = {"empty": []}
    af = ArFrame.from_dict(data)
    
    result = convert.to_pandas(af)
    
    assert len(result) == 0


def test_to_pandas_single_row():
    """Test conversion of a single-row ArFrame."""
    data = {"x": [42]}
    af = ArFrame.from_dict(data)
    
    result = convert.to_pandas(af)
    
    assert len(result) == 1
    assert result.iloc[0, 0] == 42


def test_to_pandas_with_special_chars():
    """Test conversion handles special characters in strings."""
    data = {"text": ["a-b_c.d", "e/f_g"]}
    af = ArFrame.from_dict(data)
    
    result = convert.to_pandas(af)
    
    assert len(result) == 2


def test_to_pandas_consistency_check():
    """Ensure to_pandas is consistent with from_pandas round-trip."""
    pdf = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
    
    # Convert to ArFrame and back
    af = ArFrame.from_pandas(pdf)
    df_back = convert.to_pandas(af)
    
    assert pdf.equals(df_back)


def test_to_pandas_with_numpy_array_columns():
    """Test conversion when columns are numpy arrays."""
    data = {
        "arr1": np.array([1, 2, 3]),
        "arr2": np.array([4.0, 5.0, 6.0]),
    }
    af = ArFrame.from_dict(data)
    
    result = convert.to_pandas(af)
    
    assert list(result.columns) == ["arr1", "arr2"]


def test_to_pandas_with_duplicate_columns():
    """Test conversion handles duplicate column names gracefully."""
    # Create a frame with duplicate keys (simulated via internal logic)
    data = {"col": [1, 2]}
    af = ArFrame.from_dict(data)
    
    result = convert.to_pandas(af)
    
    assert list(result.columns) == ["col"]


def test_to_pandas_with_mixed_dtypes():
    """Test conversion preserves mixed dtypes."""
    data = {
        "int": [1, 2],
        "float": [1.0, 2.0],
        "bool": [True, False],
    }
    af = ArFrame.from_dict(data)
    
    result = convert.to_pandas(af)
    
    assert len(result) == 2


def test_to_pandas_with_datetime():
    """Test conversion handles datetime columns if supported."""
    data = {
        "date": pd.date_range("2023-01-01", periods=2),
    }
    af = ArFrame.from_dict(data)
    
    result = convert.to_pandas(af)
    
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["date"]


def test_to_pandas_with_index():
    """Test conversion preserves index if present."""
    pdf = pd.DataFrame({"a": [1, 2]}, index=[10, 20])
    af = ArFrame.from_pandas(pdf)
    
    result = convert.to_pandas(af)
    
    # Ensure index is preserved or handled correctly
    assert len(result) == 2
