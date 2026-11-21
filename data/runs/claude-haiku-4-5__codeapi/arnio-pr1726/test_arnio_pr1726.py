import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestStandardizeMissingTokens:
    """Test suite for standardize_missing_tokens function."""

    def test_default_tokens_basic(self):
        """Test conversion with default tokens."""
        df = pd.DataFrame({
            "col1": ["1", "N/A", "3"],
            "col2": ["a", "NA", "c"]
        })
        frame = ar.from_pandas(df)
        result = ar.cleaning.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)
        
        assert pd.isna(result_df.loc[1, "col1"])
        assert pd.isna(result_df.loc[1, "col2"])
        assert result_df.loc[0, "col1"] == "1"
        assert result_df.loc[2, "col1"] == "3"

    def test_case_insensitive_matching(self):
        """Test that token matching is case-insensitive."""
        df = pd.DataFrame({"col": ["N/A", "n/a", "N/a", "value"]})
        frame = ar.from_pandas(df)
        result = ar.cleaning.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)
        
        assert pd.isna(result_df.loc[0, "col"])
        assert pd.isna(result_df.loc[1, "col"])
        assert pd.isna(result_df.loc[2, "col"])
        assert result_df.loc[3, "col"] == "value"

    def test_whitespace_trimming_before_check(self):
        """Test that whitespace is trimmed before checking token membership."""
        df = pd.DataFrame({
            "col": [" NULL ", "\tNaN\t", "  N/A  ", "preserved  "]
        })
        frame = ar.from_pandas(df)
        result = ar.cleaning.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)
        
        assert pd.isna(result_df.loc[0, "col"])
        assert pd.isna(result_df.loc[1, "col"])
        assert pd.isna(result_df.loc[2, "col"])
        assert result_df.loc[3, "col"] == "preserved  "

    def test_custom_tokens(self):
        """Test with custom token list."""
        df = pd.DataFrame({"col": ["custom", "CUSTOM", "value"]})
        frame = ar.from_pandas(df)
        result = ar.cleaning.standardize_missing_tokens(frame, tokens=["custom"])
        result_df = ar.to_pandas(result)
        
        assert pd.isna(result_df.loc[0, "col"])
        assert pd.isna(result_df.loc[1, "col"])
        assert result_df.loc[2, "col"] == "value"

    def test_subset_parameter_single_column(self):
        """Test subset parameter to apply to specific columns only."""
        df = pd.DataFrame({
            "col1": ["N/A", "value"],
            "col2": ["N/A", "value"]
        })
        frame = ar.from_pandas(df)
        result = ar.cleaning.standardize_missing_tokens(frame, subset=["col1"])
        result_df = ar.to_pandas(result)
        
        assert pd.isna(result_df.loc[0, "col1"])
        assert result_df.loc[0, "col2"] == "N/A"

    def test_all_default_tokens(self):
        """Test all default tokens are recognized."""
        tokens_to_test = [
            "N/A", "NA", "n/a", "na", "nan", "-", "none", "nil", "null", "", "?"
        ]
        df = pd.DataFrame({"col": tokens_to_test})
        frame = ar.from_pandas(df)
        result = ar.cleaning.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)
        
        for i in range(len(tokens_to_test)):
            assert pd.isna(result_df.loc[i, "col"])

    def test_non_string_values_preserved(self):
        """Test that non-string values are preserved."""
        df = pd.DataFrame({
            "col": [1, 2.5, None, "N/A"]
        })
        frame = ar.from_pandas(df)
        result = ar.cleaning.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)
        
        assert result_df.loc[0, "col"] == 1
        assert result_df.loc[1, "col"] == 2.5
        assert pd.isna(result_df.loc[2, "col"])
        assert pd.isna(result_df.loc[3, "col"])

    def test_empty_frame(self):
        """Test with empty frame."""
        df = pd.DataFrame({"col": []})
        frame = ar.from_pandas(df)
        result = ar.cleaning.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)
        
        assert len(result_df) == 0
        assert "col" in result_df.columns

    def test_subset_string_raises_error(self):
        """Test that passing subset as string raises TypeError."""
        df = pd.DataFrame({"col": ["N/A", "value"]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(TypeError, match="subset must be a list"):
            ar.cleaning.standardize_missing_tokens(frame, subset="col")

    def test_tokens_string_raises_error(self):
        """Test that passing tokens as string raises TypeError."""
        df = pd.DataFrame({"col": ["N/A", "value"]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(TypeError, match="tokens must be a list"):
            ar.cleaning.standardize_missing_tokens(frame, tokens="N/A")

    def test_tokens_invalid_type_raises_error(self):
        """Test that passing tokens as invalid type raises TypeError."""
        df = pd.DataFrame({"col": ["N/A", "value"]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(TypeError, match="tokens must be a list"):
            ar.cleaning.standardize_missing_tokens(frame, tokens={"N/A"})

    def test_tokens_with_non_string_item_raises_error(self):
        """Test that tokens with non-string item raises TypeError."""
        df = pd.DataFrame({"col": ["N/A", "value"]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(TypeError, match="tokens must be a list of strings"):
            ar.cleaning.standardize_missing_tokens(frame, tokens=["N/A", 123])

    def test_subset_nonexistent_column_raises_error(self):
        """Test that subset with nonexistent column raises error."""
        df = pd.DataFrame({"col1": ["N/A", "value"]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(ValueError, match="Unknown columns in subset"):
            ar.cleaning.standardize_missing_tokens(frame, subset=["nonexistent"])

    def test_multiple_columns_subset(self):
        """Test subset with multiple columns."""
        df = pd.DataFrame({
            "col1": ["N/A", "value"],
            "col2": ["NULL", "value"],
            "col3": ["N/A", "value"]
        })
        frame = ar.from_pandas(df)
        result = ar.cleaning.standardize_missing_tokens(frame, subset=["col1", "col3"])
        result_df = ar.to_pandas(result)
        
        assert pd.isna(result_df.loc[0, "col1"])
        assert result_df.loc[0, "col2"] == "NULL"
        assert pd.isna(result_df.loc[0, "col3"])

    def test_pandas_dataframe_input(self):
        """Test that function works with pandas DataFrame directly."""
        df = pd.DataFrame({"col": ["N/A", "value"]})
        result = ar.cleaning.standardize_missing_tokens(df)
        
        assert pd.isna(result.loc[0, "col"])
        assert result.loc[1, "col"] == "value"
        assert isinstance(result, pd.DataFrame)

    def test_return_type_arframe(self):
        """Test that return type is ArFrame when input is ArFrame."""
        df = pd.DataFrame({"col": ["N/A", "value"]})
        frame = ar.from_pandas(df)
        result = ar.cleaning.standardize_missing_tokens(frame)
        
        assert isinstance(result, ar.frame.ArFrame)
