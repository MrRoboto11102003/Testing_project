import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestStandardizeMissingTokens:
    """Tests for ar.standardize_missing_tokens function."""

    def test_default_tokens_replaces_common_missing_values(self):
        """Test that default tokens replace common missing-value strings."""
        df = pd.DataFrame({
            "value": ["data", "N/A", "null", "MISSING", "valid"]
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)
        
        assert result_df.iloc[0, 0] == "data"
        assert pd.isna(result_df.iloc[1, 0])
        assert pd.isna(result_df.iloc[2, 0])
        assert pd.isna(result_df.iloc[3, 0])
        assert result_df.iloc[4, 0] == "valid"

    def test_custom_tokens_list(self):
        """Test function with custom tokens list."""
        df = pd.DataFrame({
            "value": ["data", "CUSTOM", "MISSING", "valid"]
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(
            frame,
            tokens=["CUSTOM"]
        )
        result_df = ar.to_pandas(result)
        
        assert result_df.iloc[0, 0] == "data"
        assert pd.isna(result_df.iloc[1, 0])
        assert result_df.iloc[2, 0] == "MISSING"
        assert result_df.iloc[3, 0] == "valid"

    def test_case_insensitive_matching(self):
        """Test that matching is case-insensitive."""
        df = pd.DataFrame({
            "value": ["n/a", "N/A", "N/a", "nUlL", "valid"]
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)
        
        assert pd.isna(result_df.iloc[0, 0])
        assert pd.isna(result_df.iloc[1, 0])
        assert pd.isna(result_df.iloc[2, 0])
        assert pd.isna(result_df.iloc[3, 0])
        assert result_df.iloc[4, 0] == "valid"

    def test_whitespace_trimmed_before_matching(self):
        """Test that surrounding whitespace is trimmed before token matching."""
        df = pd.DataFrame({
            "value": [" N/A ", "\tN/A\t", "  null  ", "\n\nnull\n\n", "valid"]
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)
        
        assert pd.isna(result_df.iloc[0, 0])
        assert pd.isna(result_df.iloc[1, 0])
        assert pd.isna(result_df.iloc[2, 0])
        assert pd.isna(result_df.iloc[3, 0])
        assert result_df.iloc[4, 0] == "valid"

    def test_non_matching_values_preserve_whitespace(self):
        """Test that non-matching values keep their whitespace unchanged."""
        df = pd.DataFrame({
            "value": [" data ", "\tvalue\t", "  text  "]
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)
        
        assert result_df.iloc[0, 0] == " data "
        assert result_df.iloc[1, 0] == "\tvalue\t"
        assert result_df.iloc[2, 0] == "  text  "

    def test_subset_single_column(self):
        """Test applying replacement to a single column via subset."""
        df = pd.DataFrame({
            "col1": ["N/A", "valid"],
            "col2": ["N/A", "valid"]
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame, subset=["col1"])
        result_df = ar.to_pandas(result)
        
        assert pd.isna(result_df.iloc[0, 0])
        assert result_df.iloc[0, 1] == "N/A"

    def test_subset_multiple_columns(self):
        """Test applying replacement to multiple columns via subset."""
        df = pd.DataFrame({
            "col1": ["N/A", "valid"],
            "col2": ["null", "valid"],
            "col3": ["N/A", "valid"]
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame, subset=["col1", "col2"])
        result_df = ar.to_pandas(result)
        
        assert pd.isna(result_df.iloc[0, 0])
        assert pd.isna(result_df.iloc[0, 1])
        assert result_df.iloc[0, 2] == "N/A"

    def test_subset_none_applies_to_all_columns(self):
        """Test that None for subset applies the operation to all columns."""
        df = pd.DataFrame({
            "col1": ["N/A", "valid"],
            "col2": ["null", "valid"],
            "col3": ["N/A", "valid"]
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame, subset=None)
        result_df = ar.to_pandas(result)
        
        assert pd.isna(result_df.iloc[0, 0])
        assert pd.isna(result_df.iloc[0, 1])
        assert pd.isna(result_df.iloc[0, 2])

    def test_empty_frame(self):
        """Test with an empty frame."""
        df = pd.DataFrame({"col": []})
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)
        
        assert len(result_df) == 0
        assert list(result_df.columns) == ["col"]

    def test_no_matching_tokens_leaves_values_unchanged(self):
        """Test that values not matching any token remain unchanged."""
        df = pd.DataFrame({
            "value": ["data", "valid", "kept"]
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)
        
        assert result_df.iloc[0, 0] == "data"
        assert result_df.iloc[1, 0] == "valid"
        assert result_df.iloc[2, 0] == "kept"

    def test_all_values_are_tokens(self):
        """Test when all values in a column are tokens."""
        df = pd.DataFrame({
            "value": ["N/A", "null", "MISSING"]
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)
        
        assert all(pd.isna(result_df["value"]))

    def test_numeric_columns_not_affected(self):
        """Test that numeric columns are left unchanged by the function."""
        df = pd.DataFrame({
            "numbers": [1, 2, 3],
            "text": ["N/A", "valid", "null"]
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)
        
        assert result_df.iloc[0, 0] == 1
        assert result_df.iloc[1, 0] == 2
        assert result_df.iloc[2, 0] == 3
        assert pd.isna(result_df.iloc[0, 1])
        assert result_df.iloc[1, 1] == "valid"
        assert pd.isna(result_df.iloc[2, 1])

    def test_empty_tokens_list_replaces_nothing(self):
        """Test that an empty tokens list results in no replacements."""
        df = pd.DataFrame({
            "value": ["N/A", "valid"]
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame, tokens=[])
        result_df = ar.to_pandas(result)
        
        assert result_df.iloc[0, 0] == "N/A"
        assert result_df.iloc[1, 0] == "valid"

    def test_returns_new_frame_without_mutating_input(self):
        """Test that the function returns a new frame without mutating the input."""
        df = pd.DataFrame({
            "value": ["N/A", "valid"]
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame)
        
        original_df = ar.to_pandas(frame)
        assert original_df.iloc[0, 0] == "N/A"
        
        result_df = ar.to_pandas(result)
        assert pd.isna(result_df.iloc[0, 0])
