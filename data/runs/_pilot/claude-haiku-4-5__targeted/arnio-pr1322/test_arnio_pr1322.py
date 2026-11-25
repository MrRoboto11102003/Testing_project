import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestCoalesceColumns:
    """Test suite for ar.cleaning.coalesce_columns"""

    def test_basic_coalesce_first_column(self):
        """Returns first non-null value when first column has data"""
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "c": [7, 8, 9]
        })
        result = ar.cleaning.coalesce_columns(df, subset=["a", "b", "c"])
        assert "coalesced" in result.columns
        assert list(result["coalesced"]) == [1, 2, 3]

    def test_coalesce_first_non_null_in_middle(self):
        """Returns first non-null value when earlier columns are null"""
        df = pd.DataFrame({
            "a": [None, None, 3],
            "b": [1, 2, None],
            "c": [4, 5, 6]
        })
        result = ar.cleaning.coalesce_columns(df, subset=["a", "b", "c"])
        assert result["coalesced"][0] == 1
        assert result["coalesced"][1] == 2
        assert result["coalesced"][2] == 3

    def test_coalesce_last_column_is_first_non_null(self):
        """Returns value from last column when all earlier columns are null"""
        df = pd.DataFrame({
            "a": [None, None, None],
            "b": [None, None, None],
            "c": [1, 2, 3]
        })
        result = ar.cleaning.coalesce_columns(df, subset=["a", "b", "c"])
        assert list(result["coalesced"]) == [1, 2, 3]

    def test_coalesce_all_nulls_returns_null(self):
        """Returns null when all values in a row are null"""
        df = pd.DataFrame({
            "a": [None, None],
            "b": [None, None],
            "c": [None, None]
        })
        result = ar.cleaning.coalesce_columns(df, subset=["a", "b", "c"])
        assert pd.isna(result["coalesced"]).all()

    def test_coalesce_custom_output_column(self):
        """Uses custom output column name"""
        df = pd.DataFrame({
            "a": [1, 2],
            "b": [3, 4]
        })
        result = ar.cleaning.coalesce_columns(
            df, subset=["a", "b"], output_column="first_value"
        )
        assert "first_value" in result.columns
        assert "coalesced" not in result.columns
        assert list(result["first_value"]) == [1, 2]

    def test_coalesce_single_column(self):
        """Works with a single column in subset"""
        df = pd.DataFrame({"a": [1, None, 3]})
        result = ar.cleaning.coalesce_columns(df, subset=["a"])
        assert "coalesced" in result.columns

    def test_coalesce_mixed_types(self):
        """Handles mixed data types across columns"""
        df = pd.DataFrame({
            "a": [None, "text", None],
            "b": [1.5, None, 2.5]
        })
        result = ar.cleaning.coalesce_columns(df, subset=["a", "b"])
        assert result["coalesced"][0] == 1.5
        assert result["coalesced"][1] == "text"
        assert result["coalesced"][2] == 2.5

    def test_coalesce_error_empty_subset(self):
        """Raises ValueError when subset is empty"""
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(ValueError):
            ar.cleaning.coalesce_columns(df, subset=[])

    def test_coalesce_error_output_column_exists(self):
        """Raises ValueError when output_column already exists"""
        df = pd.DataFrame({
            "a": [1, 2],
            "coalesced": [3, 4]
        })
        with pytest.raises(ValueError):
            ar.cleaning.coalesce_columns(df, subset=["a"])

    def test_coalesce_error_empty_output_column_name(self):
        """Raises ValueError when output_column is empty string"""
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(ValueError):
            ar.cleaning.coalesce_columns(df, subset=["a"], output_column="")

    def test_coalesce_error_missing_column(self):
        """Raises KeyError when a column in subset doesn't exist"""
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(KeyError):
            ar.cleaning.coalesce_columns(df, subset=["a", "missing"])

    def test_coalesce_error_subset_not_sequence(self):
        """Raises TypeError when subset is not a sequence"""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        with pytest.raises(TypeError):
            ar.cleaning.coalesce_columns(df, subset="ab")


class TestStandardizeMissingTokens:
    """Test suite for ar.cleaning.standardize_missing_tokens"""

    def test_default_tokens_converted_to_nan(self):
        """Default missing tokens are converted to NaN"""
        frame = ar.from_dict({"value": ["N/A", "NULL", "NaN"]})
        result = ar.cleaning.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)
        assert pd.isna(result_df["value"]).all()

    def test_case_insensitive_token_matching(self):
        """Token matching is case-insensitive"""
        frame = ar.from_dict({"value": ["null", "NULL", "Null", "nUlL"]})
        result = ar.cleaning.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)
        assert pd.isna(result_df["value"]).all()

    def test_whitespace_trimmed_before_matching(self):
        """Surrounding whitespace is trimmed before token matching"""
        frame = ar.from_dict({"value": [" NULL ", "\tNaN\t", "  N/A  "]})
        result = ar.cleaning.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)
        assert pd.isna(result_df["value"]).all()

    def test_whitespace_preserved_for_non_matching(self):
        """Non-matching values preserve their original whitespace"""
        frame = ar.from_dict({"value": ["  kept  ", "\t\tvalue\t\t"]})
        result = ar.cleaning.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)
        assert result_df["value"][0] == "  kept  "
        assert result_df["value"][1] == "\t\tvalue\t\t"

    def test_custom_tokens_list(self):
        """Custom token list replaces default tokens"""
        frame = ar.from_dict({"value": ["CUSTOM", "MISSING", "keep", "NULL"]})
        result = ar.cleaning.standardize_missing_tokens(
            frame, tokens=["CUSTOM", "MISSING"]
        )
        result_df = ar.to_pandas(result)
        assert pd.isna(result_df["value"][0])
        assert pd.isna(result_df["value"][1])
        assert result_df["value"][2] == "keep"
        assert result_df["value"][3] == "NULL"

    def test_subset_applies_to_specific_columns_only(self):
        """subset parameter applies standardization only to specified columns"""
        frame = ar.from_dict({
            "col_a": ["NULL"],
            "col_b": ["NULL"],
            "col_c": ["NULL"]
        })
        result = ar.cleaning.standardize_missing_tokens(frame, subset=["col_a"])
        result_df = ar.to_pandas(result)
        assert pd.isna(result_df["col_a"][0])
        assert result_df["col_b"][0] == "NULL"
        assert result_df["col_c"][0] == "NULL"

    def test_subset_none_applies_all_columns(self):
        """subset=None applies standardization to all columns"""
        frame = ar.from_dict({
            "a": ["NULL"],
            "b": ["N/A"],
            "c": ["NaN"]
        })
        result = ar.cleaning.standardize_missing_tokens(frame, subset=None)
        result_df = ar.to_pandas(result)
        assert pd.isna(result_df["a"][0])
        assert pd.isna(result_df["b"][0])
        assert pd.isna(result_df["c"][0])

    def test_mixed_matching_and_non_matching_values(self):
        """Some values match tokens and are converted; others remain unchanged"""
        frame = ar.from_dict({
            "value": ["NULL", "keep", "N/A", "another", "NaN"]
        })
        result = ar.cleaning.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)
        assert pd.isna(result_df["value"][0])
        assert result_df["value"][1] == "keep"
        assert pd.isna(result_df["value"][2])
        assert result_df["value"][3] == "another"
        assert pd.isna(result_df["value"][4])

    def test_no_matching_tokens_unchanged(self):
        """When no tokens match, frame is unchanged"""
        frame = ar.from_dict({"value": ["alpha", "beta", "gamma"]})
        result = ar.cleaning.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)
        assert list(result_df["value"]) == ["alpha", "beta", "gamma"]
