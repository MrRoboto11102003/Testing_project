import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestCoalesceColumns:
    """Tests for ar.coalesce_columns()"""
    
    def test_basic_coalesce_selects_first_non_null(self):
        """Coalesce returns first non-null value in column order."""
        df = pd.DataFrame({
            "a": [None, 2, None],
            "b": [1, None, 3],
            "c": [5, 6, 7]
        })
        frame = ar.from_pandas(df)
        result = ar.coalesce_columns(
            frame, 
            subset=["a", "b", "c"], 
            output_column="first"
        )
        result_df = ar.to_pandas(result)
        
        assert result_df["first"].tolist() == [1.0, 2.0, 3.0]
    
    def test_coalesce_all_nulls_produces_null(self):
        """When all subset columns are null, result is null."""
        df = pd.DataFrame({"a": [None], "b": [None], "c": [1]})
        frame = ar.from_pandas(df)
        result = ar.coalesce_columns(frame, subset=["a", "b"], output_column="out")
        result_df = ar.to_pandas(result)
        
        assert pd.isna(result_df["out"].iloc[0])
    
    def test_coalesce_default_output_column_name(self):
        """Default output_column is 'coalesced'."""
        df = pd.DataFrame({"a": [None], "b": [1]})
        frame = ar.from_pandas(df)
        result = ar.coalesce_columns(frame, subset=["a", "b"])
        result_df = ar.to_pandas(result)
        
        assert "coalesced" in result_df.columns
        assert result_df["coalesced"].iloc[0] == 1.0
    
    def test_coalesce_empty_subset_raises_value_error(self):
        """Empty subset raises ValueError."""
        df = pd.DataFrame({"a": [1]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(ValueError):
            ar.coalesce_columns(frame, subset=[])
    
    def test_coalesce_empty_output_column_raises_value_error(self):
        """Empty output_column name raises ValueError."""
        df = pd.DataFrame({"a": [1]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(ValueError):
            ar.coalesce_columns(frame, subset=["a"], output_column="")
    
    def test_coalesce_output_column_exists_raises_value_error(self):
        """If output_column already exists, raises ValueError."""
        df = pd.DataFrame({"a": [1], "existing": [2]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(ValueError):
            ar.coalesce_columns(frame, subset=["a"], output_column="existing")
    
    def test_coalesce_missing_column_raises_key_error(self):
        """Non-existent column in subset raises KeyError."""
        df = pd.DataFrame({"a": [1]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(KeyError):
            ar.coalesce_columns(frame, subset=["a", "missing"])
    
    def test_coalesce_non_sequence_subset_raises_type_error(self):
        """Non-sequence subset raises TypeError."""
        df = pd.DataFrame({"a": [1]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(TypeError):
            ar.coalesce_columns(frame, subset="a", output_column="out")
    
    def test_coalesce_invalid_frame_type_raises_type_error(self):
        """Non-ArFrame/DataFrame input raises TypeError."""
        with pytest.raises(TypeError):
            ar.coalesce_columns([1, 2], subset=["a"])


class TestStandardizeMissingTokens:
    """Tests for ar.standardize_missing_tokens()"""
    
    def test_standardize_converts_default_missing_tokens(self):
        """Built-in missing tokens (e.g., N/A, NA, null) are converted."""
        df = pd.DataFrame({"val": ["1", "N/A", "2", "null"]})
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)
        
        assert result_df["val"].iloc[0] == "1"
        assert pd.isna(result_df["val"].iloc[1])
        assert result_df["val"].iloc[2] == "2"
        assert pd.isna(result_df["val"].iloc[3])
    
    def test_standardize_case_insensitive_matching(self):
        """Token matching is case-insensitive."""
        df = pd.DataFrame({"val": ["NULL", "Null", "null", "valid"]})
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)
        
        assert pd.isna(result_df["val"].iloc[0])
        assert pd.isna(result_df["val"].iloc[1])
        assert pd.isna(result_df["val"].iloc[2])
        assert result_df["val"].iloc[3] == "valid"
    
    def test_standardize_strips_whitespace_before_matching(self):
        """Whitespace is trimmed before matching token membership."""
        df = pd.DataFrame({"val": [" NULL ", "\tNA\t", "  null  "]})
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)
        
        assert pd.isna(result_df["val"].iloc[0])
        assert pd.isna(result_df["val"].iloc[1])
        assert pd.isna(result_df["val"].iloc[2])
    
    def test_standardize_preserves_non_matching_whitespace(self):
        """Non-matching values preserve original whitespace."""
        df = pd.DataFrame({"val": [" kept ", "\t preserved\t"]})
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)
        
        assert result_df["val"].iloc[0] == " kept "
        assert result_df["val"].iloc[1] == "\t preserved\t"
    
    def test_standardize_custom_tokens(self):
        """Custom tokens list overrides defaults."""
        df = pd.DataFrame({"val": ["CUSTOM", "valid", "CUSTOM"]})
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame, tokens=["CUSTOM"])
        result_df = ar.to_pandas(result)
        
        assert pd.isna(result_df["val"].iloc[0])
        assert result_df["val"].iloc[1] == "valid"
        assert pd.isna(result_df["val"].iloc[2])
    
    def test_standardize_subset_parameter(self):
        """Subset parameter applies transformation to only specified columns."""
        df = pd.DataFrame({
            "a": ["NULL", "1"],
            "b": ["NULL", "2"]
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame, subset=["a"])
        result_df = ar.to_pandas(result)
        
        assert pd.isna(result_df["a"].iloc[0])
        assert result_df["b"].iloc[0] == "NULL"  # Unchanged
    
    def test_standardize_returns_arframe(self):
        """Return type is ArFrame."""
        df = pd.DataFrame({"val": ["NULL", "1"]})
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame)
        
        assert isinstance(result, ar.ArFrame)
