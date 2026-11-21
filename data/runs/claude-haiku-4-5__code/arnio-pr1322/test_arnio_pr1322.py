import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestCoalesceColumns:
    """Tests for arnio.cleaning.coalesce_columns."""
    
    def test_basic_coalesce(self):
        """Test selecting first non-null value across columns."""
        df = pd.DataFrame({
            "a": [1.0, None, 3.0],
            "b": [None, 2.0, None],
            "c": [10.0, 20.0, 30.0]
        })
        result = ar.coalesce_columns(df, subset=["a", "b", "c"])
        expected_values = [1.0, 2.0, 3.0]
        assert list(result["coalesced"]) == expected_values
    
    def test_coalesce_custom_output_column(self):
        """Test coalesce with custom output column name."""
        df = pd.DataFrame({
            "x": [None, 1.0],
            "y": [2.0, None]
        })
        result = ar.coalesce_columns(df, subset=["x", "y"], output_column="first_value")
        assert "first_value" in result.columns
        assert list(result["first_value"]) == [2.0, 1.0]
    
    def test_coalesce_all_none_row(self):
        """Test that all-None rows result in NaN."""
        df = pd.DataFrame({
            "a": [None, 1.0],
            "b": [None, 2.0]
        })
        result = ar.coalesce_columns(df, subset=["a", "b"])
        assert pd.isna(result["coalesced"].iloc[0])
        assert result["coalesced"].iloc[1] == 1.0
    
    def test_coalesce_string_columns(self):
        """Test coalesce with string values."""
        df = pd.DataFrame({
            "a": [None, "text"],
            "b": ["value", None]
        })
        result = ar.coalesce_columns(df, subset=["a", "b"])
        assert result["coalesced"].iloc[0] == "value"
        assert result["coalesced"].iloc[1] == "text"
    
    def test_coalesce_empty_subset_raises_error(self):
        """Test that empty subset raises ValueError."""
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="subset must contain at least one"):
            ar.coalesce_columns(df, subset=[])
    
    def test_coalesce_missing_column_raises_error(self):
        """Test that missing column in subset raises an error."""
        df = pd.DataFrame({"a": [1]})
        with pytest.raises((KeyError, ValueError)):
            ar.coalesce_columns(df, subset=["nonexistent"])
    
    def test_coalesce_duplicate_output_column_raises_error(self):
        """Test that duplicate output column raises ValueError."""
        df = pd.DataFrame({"a": [1], "coalesced": [0]})
        with pytest.raises(ValueError, match="already exists"):
            ar.coalesce_columns(df, subset=["a"], output_column="coalesced")


class TestStandardizeMissingTokens:
    """Tests for arnio.cleaning.standardize_missing_tokens."""
    
    def test_standardize_default_tokens(self):
        """Test standardization with default missing tokens."""
        df = pd.DataFrame({
            "col": ["N/A", "NA", "value", "null", ""]
        })
        result = ar.standardize_missing_tokens(df)
        assert pd.isna(result["col"].iloc[0])
        assert pd.isna(result["col"].iloc[1])
        assert result["col"].iloc[2] == "value"
        assert pd.isna(result["col"].iloc[3])
        assert pd.isna(result["col"].iloc[4])
    
    def test_standardize_case_insensitive(self):
        """Test that token matching is case-insensitive."""
        df = pd.DataFrame({"col": ["n/a", "N/A", "Na", "value"]})
        result = ar.standardize_missing_tokens(df)
        assert pd.isna(result["col"].iloc[0])
        assert pd.isna(result["col"].iloc[1])
        assert pd.isna(result["col"].iloc[2])
        assert result["col"].iloc[3] == "value"
    
    def test_standardize_whitespace_handling(self):
        """Test that whitespace is trimmed before checking tokens."""
        df = pd.DataFrame({
            "col": [" N/A ", "\tna\t", "  null  ", "value "]
        })
        result = ar.standardize_missing_tokens(df)
        assert pd.isna(result["col"].iloc[0])
        assert pd.isna(result["col"].iloc[1])
        assert pd.isna(result["col"].iloc[2])
        assert result["col"].iloc[3] == "value "
    
    def test_standardize_custom_tokens(self):
        """Test standardization with custom tokens list."""
        df = pd.DataFrame({"col": ["CUSTOM", "unknown", "keep"]})
        result = ar.standardize_missing_tokens(df, tokens=["CUSTOM", "unknown"])
        assert pd.isna(result["col"].iloc[0])
        assert pd.isna(result["col"].iloc[1])
        assert result["col"].iloc[2] == "keep"
    
    def test_standardize_subset_columns(self):
        """Test standardization with subset of columns."""
        df = pd.DataFrame({
            "a": ["N/A", "value"],
            "b": ["N/A", "value"]
        })
        result = ar.standardize_missing_tokens(df, subset=["a"])
        assert pd.isna(result["a"].iloc[0])
        assert result["b"].iloc[0] == "N/A"
    
    def test_standardize_subset_string_raises_error(self):
        """Test that passing subset as string raises TypeError."""
        df = pd.DataFrame({"col": ["N/A"]})
        with pytest.raises(TypeError, match="subset must be a list"):
            ar.standardize_missing_tokens(df, subset="col")
    
    def test_standardize_tokens_string_raises_error(self):
        """Test that passing tokens as string raises TypeError."""
        df = pd.DataFrame({"col": ["N/A"]})
        with pytest.raises(TypeError, match="tokens must be a list"):
            ar.standardize_missing_tokens(df, tokens="N/A")
    
    def test_standardize_tokens_non_list_raises_error(self):
        """Test that passing tokens as non-list raises TypeError."""
        df = pd.DataFrame({"col": ["N/A"]})
        with pytest.raises(TypeError, match="tokens must be a list"):
            ar.standardize_missing_tokens(df, tokens=("N/A",))
