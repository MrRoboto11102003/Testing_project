import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestStandardizeMissingTokens:
    """Tests for arnio.cleaning.standardize_missing_tokens function."""

    def test_default_tokens_basic(self):
        """Test that default tokens include common NA variants."""
        df = pd.DataFrame({
            "col": ["N/A", "NA", "n/a", "na", "nan", "-", "none", "nil", "null", "", "?", "value"]
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)
        
        # All default tokens (first 11 rows) should become NaN
        for i in range(11):
            assert pd.isna(result_df.iloc[i, 0]), f"Row {i} should be NaN"
        # Non-matching values should be preserved
        assert result_df.iloc[11, 0] == "value"

    def test_case_insensitive_matching(self):
        """Test that token matching is case-insensitive."""
        df = pd.DataFrame({
            "col": ["N/A", "n/a", "N/a", "nA", "NULL", "NuLl"]
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)
        
        # All case variants should become NaN
        assert all(pd.isna(result_df["col"]))

    def test_whitespace_trimming_and_preservation(self):
        """Test whitespace trimming for matching and preservation for non-matching."""
        df = pd.DataFrame({
            "col": [" N/A ", "\tNA\t", "  null  ", " hello ", "  world  ", "\tkeep\t"]
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)
        
        # Whitespace-padded tokens should match and become NaN
        assert pd.isna(result_df.iloc[0, 0])
        assert pd.isna(result_df.iloc[1, 0])
        assert pd.isna(result_df.iloc[2, 0])
        # Non-matching values should preserve their whitespace
        assert result_df.iloc[3, 0] == " hello "
        assert result_df.iloc[4, 0] == "  world  "
        assert result_df.iloc[5, 0] == "\tkeep\t"

    def test_custom_tokens_override_defaults(self):
        """Test that custom tokens list replaces defaults."""
        df = pd.DataFrame({
            "col": ["MISSING", "UNKNOWN", "N/A", "value"]
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame, tokens=["MISSING", "UNKNOWN"])
        result_df = ar.to_pandas(result)
        
        assert pd.isna(result_df.iloc[0, 0])
        assert pd.isna(result_df.iloc[1, 0])
        # N/A should NOT be converted with custom tokens
        assert result_df.iloc[2, 0] == "N/A"
        assert result_df.iloc[3, 0] == "value"

    def test_custom_tokens_case_insensitive(self):
        """Test that custom tokens are also case-insensitive."""
        df = pd.DataFrame({
            "col": ["Missing", "MISSING", "missing", "data"]
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame, tokens=["MISSING"])
        result_df = ar.to_pandas(result)
        
        # All case variants should match
        assert all(pd.isna(result_df.iloc[:3, 0]))
        assert result_df.iloc[3, 0] == "data"

    def test_subset_single_column(self):
        """Test applying standardization to a single column."""
        df = pd.DataFrame({
            "col1": ["N/A", "value"],
            "col2": ["N/A", "value"]
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame, subset=["col1"])
        result_df = ar.to_pandas(result)
        
        # col1 should be converted
        assert pd.isna(result_df.iloc[0, 0])
        assert result_df.iloc[1, 0] == "value"
        # col2 should be unchanged
        assert result_df.iloc[0, 1] == "N/A"
        assert result_df.iloc[1, 1] == "value"

    def test_subset_multiple_columns(self):
        """Test applying standardization to multiple columns."""
        df = pd.DataFrame({
            "col1": ["N/A", "value"],
            "col2": ["null", "data"],
            "col3": ["N/A", "text"]
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame, subset=["col1", "col2"])
        result_df = ar.to_pandas(result)
        
        # col1 and col2 should be converted
        assert pd.isna(result_df.iloc[0, 0])
        assert pd.isna(result_df.iloc[0, 1])
        # col3 should be unchanged
        assert result_df.iloc[0, 2] == "N/A"

    def test_numeric_and_non_string_values_unchanged(self):
        """Test that numeric and non-string values are not affected."""
        df = pd.DataFrame({
            "col": [1, 2.5, None, "N/A", 0, -999]
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)
        
        assert result_df.iloc[0, 0] == 1
        assert result_df.iloc[1, 0] == 2.5
        assert pd.isna(result_df.iloc[2, 0])  # Original None
        assert pd.isna(result_df.iloc[3, 0])  # Converted from "N/A"
        assert result_df.iloc[4, 0] == 0
        assert result_df.iloc[5, 0] == -999

    def test_returns_arframe_for_arframe_input(self):
        """Test that input ArFrame returns ArFrame."""
        df = pd.DataFrame({"col": ["N/A", "value"]})
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame)
        
        assert isinstance(result, ar.ArFrame)

    def test_returns_dataframe_for_dataframe_input(self):
        """Test that input DataFrame returns DataFrame."""
        df = pd.DataFrame({"col": ["N/A", "value"]})
        result = ar.standardize_missing_tokens(df)
        
        assert isinstance(result, pd.DataFrame)

    def test_error_subset_as_string(self):
        """Test that subset as string raises TypeError with helpful message."""
        df = pd.DataFrame({"col1": ["N/A", "value"]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(TypeError) as exc_info:
            ar.standardize_missing_tokens(frame, subset="col1")
        
        assert "subset must be a list" in str(exc_info.value)

    def test_error_tokens_as_string(self):
        """Test that tokens as string raises TypeError with helpful message."""
        df = pd.DataFrame({"col": ["N/A", "value"]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(TypeError) as exc_info:
            ar.standardize_missing_tokens(frame, tokens="N/A")
        
        assert "tokens must be a list" in str(exc_info.value)

    def test_error_tokens_invalid_types(self):
        """Test that tokens must be a list of strings."""
        df = pd.DataFrame({"col": ["N/A", "value"]})
        frame = ar.from_pandas(df)
        
        # tokens as set
        with pytest.raises(TypeError) as exc_info:
            ar.standardize_missing_tokens(frame, tokens={"N/A"})
        assert "tokens must be a list" in str(exc_info.value)
        
        # tokens with non-string item
        with pytest.raises(TypeError) as exc_info:
            ar.standardize_missing_tokens(frame, tokens=["N/A", 123])
        assert "tokens must be a list of strings" in str(exc_info.value)

    def test_error_missing_column_in_subset(self):
        """Test that missing columns in subset raise ValueError."""
        df = pd.DataFrame({"col1": ["N/A", "value"]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(ValueError) as exc_info:
            ar.standardize_missing_tokens(frame, subset=["nonexistent"])
        
        assert "Unknown columns" in str(exc_info.value)

    def test_empty_dataframe(self):
        """Test that empty DataFrame is handled gracefully."""
        df = pd.DataFrame({"col": pd.Series([], dtype="object")})
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)
        
        assert len(result_df) == 0
        assert list(result_df.columns) == ["col"]

    def test_all_missing_tokens_in_column(self):
        """Test column with all missing tokens becomes all NaN."""
        df = pd.DataFrame({
            "col": ["N/A", "null", "nan", "-", ""]
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)
        
        assert all(pd.isna(result_df["col"]))
