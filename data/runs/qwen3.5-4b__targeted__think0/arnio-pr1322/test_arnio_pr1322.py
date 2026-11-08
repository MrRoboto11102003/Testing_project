import pytest
import pandas as pd
import numpy as np
from typing import Sequence, Optional

# Mock the ArFrame class to allow testing without full arnio installation
class MockArFrame:
    def __init__(self, data):
        self._data = data
    
    @property
    def columns(self):
        return list(self._data.columns)
    
    def __getitem__(self, key):
        return self._data[key]
    
    def copy(self):
        return MockArFrame(self._data.copy())

# Import the functions under test
from arnio.cleaning import coalesce_columns, standardize_missing_tokens


def test_coalesce_columns_basic():
    """Test basic coalescing of non-null values."""
    data = {"col_a": [None, 10, 20], "col_b": [5, None, 30]}
    frame = MockArFrame(data)
    
    result = coalesce_columns(frame, subset=["col_a", "col_b"], output_column="coalesced")
    
    assert "coalesced" in result.columns
    expected = [10, 20, 30]  # First non-null from col_a, then col_b
    assert result["coalesced"].tolist() == expected


def test_coalesce_columns_empty_subset():
    """Test that empty subset raises ValueError."""
    data = {"col_a": [1]}
    frame = MockArFrame(data)
    
    with pytest.raises(ValueError):
        coalesce_columns(frame, subset=[], output_column="coalesced")


def test_coalesce_columns_duplicate_output():
    """Test that duplicate output column name raises ValueError."""
    data = {"col_a": [1], "coalesced": [2]}
    frame = MockArFrame(data)
    
    with pytest.raises(ValueError):
        coalesce_columns(frame, subset=["col_a"], output_column="coalesced")


def test_coalesce_columns_missing_column():
    """Test that missing column in subset raises KeyError."""
    data = {"col_a": [1]}
    frame = MockArFrame(data)
    
    with pytest.raises(KeyError):
        coalesce_columns(frame, subset=["nonexistent"], output_column="coalesced")


def test_coalesce_columns_type_error():
    """Test that non-DataFrame input raises TypeError."""
    # Test with plain list instead of frame
    data = {"col_a": [1]}
    with pytest.raises(TypeError):
        coalesce_columns(data, subset=["col_a"], output_column="coalesced")


def test_standardize_missing_tokens_basic():
    """Test basic token standardization."""
    data = {"value": ["N/A", "null", "kept"]}
    frame = MockArFrame(data)
    
    result = standardize_missing_tokens(frame, tokens=["N/A", "null"])
    
    # Check that NaN values are present where expected
    assert pd.isna(result["value"][0])
    assert pd.isna(result["value"][1])
    assert not pd.isna(result["value"][2])


def test_standardize_missing_tokens_whitespace_handling():
    """Test that whitespace is trimmed before matching."""
    data = {"value": [" NULL ", "\\tNaN\\t", "kept"]}
    frame = MockArFrame(data)
    
    result = standardize_missing_tokens(frame, tokens=["NULL", "NaN"])
    
    # Whitespace should be trimmed for matching
    assert pd.isna(result["value"][0])  # " NULL " -> NULL
    assert pd.isna(result["value"][1])  # "\\tNaN\\t" -> NaN
    assert not pd.isna(result["value"][2])


def test_standardize_missing_tokens_case_insensitive():
    """Test that matching is case-insensitive."""
    data = {"value": ["n/a", "NULL", "Kept"]}
    frame = MockArFrame(data)
    
    result = standardize_missing_tokens(frame, tokens=["N/A"])
    
    assert pd.isna(result["value"][0])  # n/a -> N/A
    assert pd.isna(result["value"][1])  # NULL -> N/A
    assert not pd.isna(result["value"][2])


def test_standardize_missing_tokens_subset_filter():
    """Test that subset parameter filters columns."""
    data = {"col_a": ["N/A", "kept"], "col_b": [1, 2]}
    frame = MockArFrame(data)
    
    result = standardize_missing_tokens(frame, tokens=["N/A"], subset=["col_a"])
    
    # Only col_a should be modified
    assert pd.isna(result["col_a"][0])
    assert not pd.isna(result["col_a"][1])
    assert result["col_b"].tolist() == [1, 2]


def test_standardize_missing_tokens_no_subset():
    """Test that None subset applies to all columns."""
    data = {"col_a": ["N/A"], "col_b": [1]}
    frame = MockArFrame(data)
    
    result = standardize_missing_tokens(frame, tokens=["N/A"])
    
    assert pd.isna(result["col_a"][0])


def test_standardize_missing_tokens_default_tokens():
    """Test that None tokens uses default built-in list."""
    data = {"value": ["", "  ", "kept"]}
    frame = MockArFrame(data)
    
    result = standardize_missing_tokens(frame)
    
    # Empty strings and whitespace-only strings should be NaN by default
    assert pd.isna(result["value"][0])
    assert pd.isna(result["value"][1])


def test_standardize_missing_tokens_preserves_original_whitespace():
    """Test that non-matching values preserve original whitespace."""
    data = {"value": ["kept", "  kept  "]}
    frame = MockArFrame(data)
    
    result = standardize_missing_tokens(frame, tokens=["N/A"])
    
    # Values not matching tokens should keep their original form
    assert result["value"][0] == "kept"
    assert result["value"][1] == "  kept  "


def test_coalesce_columns_with_pandas():
    """Test coalesce_columns works with pandas DataFrame."""
    data = {"col_a": [None, 10], "col_b": [5, None]}
    df = pd.DataFrame(data)
    
    result = coalesce_columns(df, subset=["col_a", "col_b"], output_column="coalesced")
    
    assert isinstance(result, pd.DataFrame)
    assert "coalesced" in result.columns


def test_standardize_missing_tokens_with_pandas():
    """Test standardize_missing_tokens works with pandas DataFrame."""
    data = {"value": ["N/A", "kept"]}
    df = pd.DataFrame(data)
    
    result = standardize_missing_tokens(df, tokens=["N/A"])
    
    assert isinstance(result, pd.DataFrame)
    assert pd.isna(result["value"][0])


def test_coalesce_columns_multiple_non_nulls():
    """Test coalescing when multiple columns have non-null values."""
    data = {"col_a": [1, None], "col_b": [None, 2]}
    frame = MockArFrame(data)
    
    result = coalesce_columns(frame, subset=["col_a", "col_b"], output_column="coalesced")
    
    # First non-null in col_a (index 0), then first non-null in col_b (index 1)
    assert result["coalesced"].tolist() == [1, 2]


def test_standardize_missing_tokens_multiple_tokens():
    """Test standardizing multiple tokens at once."""
    data = {"value": ["N/A", "null", "n/a", "kept"]}
    frame = MockArFrame(data)
    
    result = standardize_missing_tokens(frame, tokens=["N/A", "null"])
    
    assert pd.isna(result["value"][0])
    assert pd.isna(result["value"][1])
    assert pd.isna(result["value"][2])
    assert not pd.isna(result["value"][3])


def test_standardize_missing_tokens_case_mismatch():
    """Test that case mismatch is not treated as missing token."""
    data = {"value": ["n/a", "N/A"]}
    frame = MockArFrame(data)
    
    result = standardize_missing_tokens(frame, tokens=["N/A"])
    
    # Case mismatch: 'n/a' should NOT be converted to NaN
    assert not pd.isna(result["value"][0])


def test_coalesce_columns_all_nulls():
    """Test coalescing when all columns are null."""
    data = {"col_a": [None, None], "col_b": [None, None]}
    frame = MockArFrame(data)
    
    result = coalesce_columns(frame, subset=["col_a", "col_b"], output_column="coalesced")
    
    # Should be all NaN
    assert pd.isna(result["coalesced"][0])
    assert pd.isna(result["coalesced"][1])


def test_standardize_missing_tokens_all_match():
    """Test standardizing when all values match tokens."""
    data = {"value": ["N/A", "null"]}
    frame = MockArFrame(data)
    
    result = standardize_missing_tokens(frame, tokens=["N/A", "null"])
    
    assert pd.isna(result["value"][0])
    assert pd.isna(result["value"][1])
