import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestCoalesceColumns:
    """Tests for ar.cleaning.coalesce_columns"""

    def test_basic_coalesce_with_nulls(self):
        """Test basic coalescing of two columns with nulls."""
        df = pd.DataFrame({
            "col_a": [1.0, np.nan, 3.0],
            "col_b": [np.nan, 2.0, np.nan],
        })
        result = ar.cleaning.coalesce_columns(df, subset=["col_a", "col_b"])
        assert list(result["coalesced"]) == [1.0, 2.0, 3.0]

    def test_coalesce_three_columns_picks_first_non_null(self):
        """Test that coalesce picks first non-null value from left to right."""
        df = pd.DataFrame({
            "a": [np.nan, 1.0, np.nan],
            "b": [2.0, np.nan, 3.0],
            "c": [4.0, 5.0, 6.0],
        })
        result = ar.cleaning.coalesce_columns(df, subset=["a", "b", "c"])
        assert list(result["coalesced"]) == [2.0, 1.0, 3.0]

    def test_coalesce_all_nulls_produces_nan(self):
        """Test that rows with all nulls produce NaN."""
        df = pd.DataFrame({
            "x": [1.0, np.nan],
            "y": [np.nan, np.nan],
        })
        result = ar.cleaning.coalesce_columns(df, subset=["x", "y"])
        assert result["coalesced"].iloc[0] == 1.0
        assert pd.isna(result["coalesced"].iloc[1])

    def test_coalesce_custom_output_column_name(self):
        """Test using a custom output column name."""
        df = pd.DataFrame({"a": [1.0], "b": [2.0]})
        result = ar.cleaning.coalesce_columns(
            df, subset=["a", "b"], output_column="first_val"
        )
        assert "first_val" in result.columns

    def test_coalesce_empty_subset_raises_error(self):
        """Test that empty subset raises ValueError."""
        df = pd.DataFrame({"a": [1.0]})
        with pytest.raises(ValueError):
            ar.cleaning.coalesce_columns(df, subset=[])

    def test_coalesce_output_column_exists_raises_error(self):
        """Test that existing output_column raises ValueError."""
        df = pd.DataFrame({"a": [1.0], "coalesced": [2.0]})
        with pytest.raises(ValueError):
            ar.cleaning.coalesce_columns(df, subset=["a"])

    def test_coalesce_missing_column_raises_error(self):
        """Test that missing column in subset raises error."""
        df = pd.DataFrame({"a": [1.0]})
        with pytest.raises((KeyError, ValueError)):
            ar.cleaning.coalesce_columns(df, subset=["nonexistent"])


class TestStandardizeMissingTokens:
    """Tests for ar.cleaning.standardize_missing_tokens"""

    def test_standardize_default_tokens(self):
        """Test conversion of default missing tokens to NaN."""
        df = pd.DataFrame({
            "val": ["N/A", "na", "NA", "n/a", "valid"],
        })
        result = ar.cleaning.standardize_missing_tokens(df)
        assert pd.isna(result["val"].iloc[0])
        assert pd.isna(result["val"].iloc[1])
        assert pd.isna(result["val"].iloc[2])
        assert pd.isna(result["val"].iloc[3])
        assert result["val"].iloc[4] == "valid"

    def test_standardize_case_insensitive(self):
        """Test that matching is case-insensitive."""
        df = pd.DataFrame({"val": ["NONE", "None", "none", "VALID"]})
        result = ar.cleaning.standardize_missing_tokens(df)
        assert pd.isna(result["val"].iloc[0])
        assert pd.isna(result["val"].iloc[1])
        assert pd.isna(result["val"].iloc[2])
        assert result["val"].iloc[3] == "VALID"

    def test_standardize_whitespace_trimming(self):
        """Test that whitespace is trimmed before matching."""
        df = pd.DataFrame({
            "val": [" N/A ", "\tnan\t", "  value  "],
        })
        result = ar.cleaning.standardize_missing_tokens(df)
        assert pd.isna(result["val"].iloc[0])
        assert pd.isna(result["val"].iloc[1])
        assert result["val"].iloc[2] == "  value  "

    def test_standardize_custom_tokens(self):
        """Test using custom tokens."""
        df = pd.DataFrame({"val": ["custom_null", "valid"]})
        result = ar.cleaning.standardize_missing_tokens(
            df, tokens=["custom_null"]
        )
        assert pd.isna(result["val"].iloc[0])
        assert result["val"].iloc[1] == "valid"

    def test_standardize_subset_parameter(self):
        """Test that subset parameter limits which columns are affected."""
        df = pd.DataFrame({
            "col1": ["N/A", "value1"],
            "col2": ["N/A", "value2"],
        })
        result = ar.cleaning.standardize_missing_tokens(df, subset=["col1"])
        assert pd.isna(result["col1"].iloc[0])
        assert result["col2"].iloc[0] == "N/A"

    def test_standardize_subset_string_raises_error(self):
        """Test that passing subset as string raises TypeError."""
        df = pd.DataFrame({"a": ["N/A"]})
        with pytest.raises(TypeError):
            ar.cleaning.standardize_missing_tokens(df, subset="a")

    def test_standardize_tokens_string_raises_error(self):
        """Test that passing tokens as string raises TypeError."""
        df = pd.DataFrame({"a": ["N/A"]})
        with pytest.raises(TypeError):
            ar.cleaning.standardize_missing_tokens(df, tokens="N/A")

    def test_standardize_non_string_values_preserved(self):
        """Test that non-string values are not affected."""
        df = pd.DataFrame({
            "mixed": [1, 2.5, "N/A", None, "valid"],
        })
        result = ar.cleaning.standardize_missing_tokens(df)
        assert result["mixed"].iloc[0] == 1
        assert result["mixed"].iloc[1] == 2.5
        assert pd.isna(result["mixed"].iloc[2])
        assert pd.isna(result["mixed"].iloc[3])
        assert result["mixed"].iloc[4] == "valid"
