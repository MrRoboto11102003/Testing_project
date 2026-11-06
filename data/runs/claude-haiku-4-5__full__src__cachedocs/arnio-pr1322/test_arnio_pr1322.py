"""
Pytest tests for arnio.cleaning.coalesce_columns and 
arnio.cleaning.standardize_missing_tokens.

Tests cover documented behavior, edge cases, error conditions, and type handling.
"""

import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestCoalesceColumns:
    """Tests for ar.coalesce_columns function."""

    def test_basic_coalesce_selects_first_non_null(self):
        """Test that coalesce_columns selects the first non-null value in order."""
        df = pd.DataFrame({
            "a": [None, 1, None],
            "b": [2, None, 3],
            "c": [4, 5, 6],
        })
        frame = ar.from_pandas(df)
        result = ar.coalesce_columns(frame, subset=["a", "b", "c"])
        result_df = ar.to_pandas(result)
        
        assert result_df["coalesced"].tolist() == [2.0, 1.0, 3.0]

    def test_preserves_arframe_type(self):
        """Test that ArFrame input returns ArFrame output."""
        df = pd.DataFrame({"a": [None, 1], "b": [2, None]})
        frame = ar.from_pandas(df)
        result = ar.coalesce_columns(frame, subset=["a", "b"])
        
        assert isinstance(result, ar.ArFrame)

    def test_preserves_pandas_type(self):
        """Test that pandas DataFrame input returns pandas DataFrame output."""
        df = pd.DataFrame({"a": [None, 1], "b": [2, None]})
        result = ar.coalesce_columns(df, subset=["a", "b"])
        
        assert isinstance(result, pd.DataFrame)

    def test_custom_output_column_name(self):
        """Test custom output_column parameter."""
        df = pd.DataFrame({"a": [None, 1], "b": [2, None]})
        result = ar.coalesce_columns(df, subset=["a", "b"], output_column="my_col")
        
        assert "my_col" in result.columns
        assert "coalesced" not in result.columns

    def test_empty_subset_raises_value_error(self):
        """Test that empty subset raises ValueError."""
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(ValueError):
            ar.coalesce_columns(df, subset=[])

    def test_empty_output_column_raises_value_error(self):
        """Test that empty output_column raises ValueError."""
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(ValueError):
            ar.coalesce_columns(df, subset=["a"], output_column="")

    def test_existing_output_column_raises_value_error(self):
        """Test that output_column already existing raises ValueError."""
        df = pd.DataFrame({"a": [1, 2], "existing": [3, 4]})
        with pytest.raises(ValueError):
            ar.coalesce_columns(df, subset=["a"], output_column="existing")

    def test_missing_column_in_subset_raises_error(self):
        """Test that missing column in subset raises error."""
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises((KeyError, ValueError)):
            ar.coalesce_columns(df, subset=["a", "nonexistent"])

    def test_coalesce_order_is_significant(self):
        """Test that column order in subset determines precedence."""
        df = pd.DataFrame({
            "first": [1, None, 3],
            "second": [10, 20, 30],
        })
        result = ar.coalesce_columns(df, subset=["first", "second"])
        
        # When "first" is available, it should be used
        assert result["coalesced"].iloc[0] == 1.0
        # When "first" is None, "second" should be used
        assert result["coalesced"].iloc[1] == 20.0

    def test_all_null_row_returns_nan(self):
        """Test that rows with all null values result in NaN."""
        df = pd.DataFrame({
            "a": [None, 1],
            "b": [None, 2],
        })
        result = ar.coalesce_columns(df, subset=["a", "b"])
        
        assert pd.isna(result["coalesced"].iloc[0])
        assert not pd.isna(result["coalesced"].iloc[1])


class TestStandardizeMissingTokens:
    """Tests for ar.standardize_missing_tokens function."""

    def test_converts_default_tokens_to_nan(self):
        """Test that default missing tokens are converted to NaN."""
        df = pd.DataFrame({"col": ["N/A", "value", "none", "data"]})
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)
        
        assert pd.isna(result_df["col"].iloc[0])  # N/A
        assert result_df["col"].iloc[1] == "value"
        assert pd.isna(result_df["col"].iloc[2])  # none
        assert result_df["col"].iloc[3] == "data"

    def test_custom_tokens_list(self):
        """Test that custom tokens list is used."""
        df = pd.DataFrame({"col": ["MISSING", "value", "MISSING"]})
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame, tokens=["MISSING"])
        result_df = ar.to_pandas(result)
        
        assert pd.isna(result_df["col"].iloc[0])
        assert result_df["col"].iloc[1] == "value"
        assert pd.isna(result_df["col"].iloc[2])

    def test_case_insensitive_token_matching(self):
        """Test that token matching is case-insensitive."""
        df = pd.DataFrame({"col": ["null", "NULL", "Null", "value"]})
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)
        
        assert pd.isna(result_df["col"].iloc[0])
        assert pd.isna(result_df["col"].iloc[1])
        assert pd.isna(result_df["col"].iloc[2])
        assert result_df["col"].iloc[3] == "value"

    def test_whitespace_trimmed_before_matching(self):
        """Test that whitespace is trimmed before token matching."""
        df = pd.DataFrame({"col": [" N/A ", "\tnone\n", "  value  "]})
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)
        
        assert pd.isna(result_df["col"].iloc[0])  # " N/A " → N/A
        assert pd.isna(result_df["col"].iloc[1])  # "\tnone\n" → none
        assert result_df["col"].iloc[2] == "  value  "  # Non-matching preserves whitespace

    def test_subset_limits_affected_columns(self):
        """Test that subset parameter limits columns affected."""
        df = pd.DataFrame({
            "col_a": ["N/A", "value"],
            "col_b": ["N/A", "value"],
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame, subset=["col_a"])
        result_df = ar.to_pandas(result)
        
        assert pd.isna(result_df["col_a"].iloc[0])  # Processed
        assert result_df["col_b"].iloc[0] == "N/A"  # Not processed

    def test_subset_as_string_raises_type_error(self):
        """Test that subset as string (not list) raises TypeError."""
        df = pd.DataFrame({"col": ["N/A"]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(TypeError, match="list of column names"):
            ar.standardize_missing_tokens(frame, subset="col")

    def test_tokens_as_string_raises_type_error(self):
        """Test that tokens as string (not list) raises TypeError."""
        df = pd.DataFrame({"col": ["N/A"]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(TypeError, match="list of strings"):
            ar.standardize_missing_tokens(frame, tokens="N/A")

    def test_empty_string_is_default_missing_token(self):
        """Test that empty string is treated as a missing token by default."""
        df = pd.DataFrame({"col": ["", "value", ""]})
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)
        
        assert pd.isna(result_df["col"].iloc[0])
        assert result_df["col"].iloc[1] == "value"
        assert pd.isna(result_df["col"].iloc[2])

    def test_non_string_values_unaffected(self):
        """Test that non-string column values are not modified."""
        df = pd.DataFrame({
            "str_col": ["N/A", "value"],
            "int_col": [1, 2],
            "float_col": [1.5, 2.5],
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)
        
        assert pd.isna(result_df["str_col"].iloc[0])
        assert result_df["int_col"].tolist() == [1, 2]
        assert result_df["float_col"].tolist() == [1.5, 2.5]

    def test_preserves_arframe_type(self):
        """Test that function returns same type as input."""
        df = pd.DataFrame({"col": ["N/A", "value"]})
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame)
        
        assert isinstance(result, ar.ArFrame)
