import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestStandardizeMissingTokens:
    """End-to-end tests for arnio.cleaning.standardize_missing_tokens"""

    def test_basic_standardize_defaults(self, tmp_path):
        """Test basic standardization with default tokens list."""
        # Create test file on disk
        csv_path = tmp_path / "input.csv"
        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "value": ["10", "N/A", "20", "na", "30"],
            "category": ["A", "null", "B", "nil", "C"],
        })
        df.to_csv(csv_path, index=False)
        
        # Load, process, and verify
        df_loaded = pd.read_csv(csv_path)
        frame = ar.from_pandas(df_loaded)
        result = ar.cleaning.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)
        
        assert result_df.loc[0, "value"] == "10"
        assert pd.isna(result_df.loc[1, "value"])  # N/A
        assert result_df.loc[2, "value"] == "20"
        assert pd.isna(result_df.loc[3, "value"])  # na
        assert pd.isna(result_df.loc[1, "category"])  # null
        assert pd.isna(result_df.loc[3, "category"])  # nil
        
        # Save result
        output_path = tmp_path / "output.csv"
        result_df.to_csv(output_path, index=False)
        assert output_path.exists()

    def test_custom_tokens(self):
        """Test standardization with custom token list."""
        df = pd.DataFrame({
            "value": ["valid", "CUSTOM_NA", "another", "custom_na"],
        })
        frame = ar.from_pandas(df)
        
        result = ar.cleaning.standardize_missing_tokens(
            frame,
            tokens=["CUSTOM_NA"]
        )
        result_df = ar.to_pandas(result)
        
        assert result_df.loc[0, "value"] == "valid"
        assert pd.isna(result_df.loc[1, "value"])
        assert result_df.loc[2, "value"] == "another"
        assert pd.isna(result_df.loc[3, "value"])

    def test_subset_columns(self):
        """Test applying standardization to subset of columns only."""
        df = pd.DataFrame({
            "col_a": ["NA", "value", "null"],
            "col_b": ["NA", "value", "null"],
            "col_c": ["NA", "value", "null"],
        })
        frame = ar.from_pandas(df)
        
        result = ar.cleaning.standardize_missing_tokens(
            frame,
            subset=["col_a", "col_b"]
        )
        result_df = ar.to_pandas(result)
        
        assert pd.isna(result_df.loc[0, "col_a"])
        assert pd.isna(result_df.loc[0, "col_b"])
        assert result_df.loc[0, "col_c"] == "NA"  # Not in subset
        assert result_df.loc[2, "col_c"] == "null"

    def test_case_insensitive_matching(self):
        """Test that token matching is case-insensitive."""
        df = pd.DataFrame({
            "value": ["N/A", "n/a", "N/a", "n/A", "NaN", "nan"],
        })
        frame = ar.from_pandas(df)
        
        result = ar.cleaning.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)
        
        assert all(pd.isna(result_df["value"]))

    def test_whitespace_trimming_before_matching(self):
        """Test that whitespace is trimmed before checking token membership."""
        df = pd.DataFrame({
            "value": [" NULL ", "\tNaN\t", "  na  ", "normal"],
        })
        frame = ar.from_pandas(df)
        
        result = ar.cleaning.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)
        
        assert pd.isna(result_df.loc[0, "value"])  # " NULL " trimmed to "null"
        assert pd.isna(result_df.loc[1, "value"])  # "\tNaN\t" trimmed to "nan"
        assert pd.isna(result_df.loc[2, "value"])  # "  na  " trimmed to "na"
        assert result_df.loc[3, "value"] == "normal"

    def test_preserve_whitespace_nonmatching(self):
        """Test that non-matching values preserve original whitespace."""
        df = pd.DataFrame({
            "value": [" kept ", "\ttabbed\t", "  spaced  "],
        })
        frame = ar.from_pandas(df)
        
        result = ar.cleaning.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)
        
        assert result_df.loc[0, "value"] == " kept "
        assert result_df.loc[1, "value"] == "\ttabbed\t"
        assert result_df.loc[2, "value"] == "  spaced  "

    def test_mixed_types_in_column(self):
        """Test standardization with mixed data types in a column."""
        df = pd.DataFrame({
            "mixed": [1, "NA", 3.14, "null", None, "valid"],
        })
        frame = ar.from_pandas(df)
        
        result = ar.cleaning.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)
        
        assert result_df.loc[0, "mixed"] == 1
        assert pd.isna(result_df.loc[1, "mixed"])
        assert result_df.loc[2, "mixed"] == 3.14
        assert pd.isna(result_df.loc[3, "mixed"])
        assert pd.isna(result_df.loc[4, "mixed"])
        assert result_df.loc[5, "mixed"] == "valid"

    def test_empty_dataframe(self):
        """Test standardization on empty dataframe."""
        df = pd.DataFrame({"col_a": [], "col_b": []})
        frame = ar.from_pandas(df)
        
        result = ar.cleaning.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)
        
        assert len(result_df) == 0
        assert list(result_df.columns) == ["col_a", "col_b"]

    def test_all_default_missing_tokens(self):
        """Test that all default missing tokens are recognized."""
        # Default tokens from source: ["N/A", "NA", "n/a", "na", "nan", "-", "none", "nil", "null", "", "?"]
        df = pd.DataFrame({
            "tokens": ["N/A", "NA", "n/a", "na", "nan", "-", "none", "nil", "null", "", "?"],
        })
        frame = ar.from_pandas(df)
        
        result = ar.cleaning.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)
        
        assert all(pd.isna(result_df["tokens"]))

    def test_error_subset_string_not_list(self):
        """Test that passing subset as string raises TypeError."""
        df = pd.DataFrame({"col": ["NA"]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(TypeError, match="subset must be a list"):
            ar.cleaning.standardize_missing_tokens(frame, subset="col")

    def test_error_tokens_string_not_list(self):
        """Test that passing tokens as string raises TypeError."""
        df = pd.DataFrame({"col": ["NA"]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(TypeError, match="tokens must be a list of strings"):
            ar.cleaning.standardize_missing_tokens(frame, tokens="NA")

    def test_error_tokens_not_list_type(self):
        """Test that passing tokens as non-list iterable raises TypeError."""
        df = pd.DataFrame({"col": ["NA"]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(TypeError, match="tokens must be a list"):
            ar.cleaning.standardize_missing_tokens(frame, tokens=("NA", "null"))

    def test_error_tokens_non_string_item(self):
        """Test that tokens list with non-string item raises TypeError."""
        df = pd.DataFrame({"col": ["NA"]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(TypeError, match="tokens must be a list of strings"):
            ar.cleaning.standardize_missing_tokens(frame, tokens=["NA", 123])

    def test_error_invalid_subset_column(self):
        """Test that unknown column in subset raises ValueError."""
        df = pd.DataFrame({"col_a": ["NA"]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(ValueError, match="Unknown columns in subset"):
            ar.cleaning.standardize_missing_tokens(frame, subset=["nonexistent"])
