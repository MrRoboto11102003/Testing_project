import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestStandardizeMissingTokens:
    """Test suite for arnio.cleaning.standardize_missing_tokens"""
    
    def test_basic_standardization_nslash_a(self):
        """Test that N/A is replaced with NaN"""
        df = pd.DataFrame({
            "name": ["Alice", "N/A", "Bob"],
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)
        
        assert result_df.loc[0, "name"] == "Alice"
        assert pd.isna(result_df.loc[1, "name"])
        assert result_df.loc[2, "name"] == "Bob"
    
    def test_multiple_common_tokens(self):
        """Test that multiple common missing-value tokens are standardized"""
        df = pd.DataFrame({
            "col": ["value", "N/A", "null", "NA", "MISSING"]
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)
        
        assert result_df.loc[0, "col"] == "value"
        for i in [1, 2, 3, 4]:
            assert pd.isna(result_df.loc[i, "col"])
    
    def test_custom_tokens_parameter(self):
        """Test that custom token list is respected"""
        df = pd.DataFrame({
            "col": ["value", "UNKNOWN", "N/A", "CUSTOM_NULL"]
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame, tokens=["UNKNOWN", "CUSTOM_NULL"])
        result_df = ar.to_pandas(result)
        
        assert result_df.loc[0, "col"] == "value"
        assert pd.isna(result_df.loc[1, "col"])
        assert result_df.loc[2, "col"] == "N/A"  # Not in custom tokens
        assert pd.isna(result_df.loc[3, "col"])
    
    def test_subset_single_column(self):
        """Test that subset parameter limits affected columns"""
        df = pd.DataFrame({
            "name": ["Alice", "N/A"],
            "status": ["active", "N/A"]
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame, subset=["name"])
        result_df = ar.to_pandas(result)
        
        assert pd.isna(result_df.loc[1, "name"])
        assert result_df.loc[1, "status"] == "N/A"  # Not affected
    
    def test_subset_multiple_columns(self):
        """Test subset with multiple column names"""
        df = pd.DataFrame({
            "a": ["N/A", "1"],
            "b": ["NULL", "2"],
            "c": ["N/A", "3"]
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame, subset=["a", "c"])
        result_df = ar.to_pandas(result)
        
        assert pd.isna(result_df.loc[0, "a"])
        assert result_df.loc[0, "b"] == "NULL"
        assert pd.isna(result_df.loc[0, "c"])
    
    def test_empty_dataframe(self):
        """Test with empty dataframe returns ArFrame"""
        df = pd.DataFrame()
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame)
        
        assert isinstance(result, ar.ArFrame)
        assert result.shape[0] == 0
    
    def test_no_missing_values_unchanged(self):
        """Test dataframe with no missing tokens remains unchanged"""
        df = pd.DataFrame({
            "col1": ["a", "b", "c"],
            "col2": ["1", "2", "3"]
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)
        
        pd.testing.assert_frame_equal(result_df, df, check_dtype=True)
    
    def test_all_values_standardized(self):
        """Test dataframe where all values are missing tokens"""
        df = pd.DataFrame({
            "col": ["N/A", "NA", "null", "MISSING"]
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)
        
        assert result_df["col"].isna().all()
    
    def test_case_insensitive_matching(self):
        """Test that token matching is case-insensitive"""
        df = pd.DataFrame({
            "col": ["N/A", "n/a", "N/a", "value", "NA", "na"]
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)
        
        # Case variants of N/A and NA should be standardized
        for i in [0, 1, 2, 4, 5]:
            assert pd.isna(result_df.loc[i, "col"]), f"Row {i} should be NaN"
        assert result_df.loc[3, "col"] == "value"
    
    def test_returns_arframe_instance(self):
        """Test that result is an ArFrame"""
        df = pd.DataFrame({"col": ["N/A", "value"]})
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame)
        
        assert isinstance(result, ar.ArFrame)
    
    def test_numeric_columns_unaffected(self):
        """Test that numeric columns pass through unchanged"""
        df = pd.DataFrame({
            "num_int": [1, 2, 3],
            "num_float": [1.5, 2.5, 3.5],
            "text": ["N/A", "valid", "NA"]
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)
        
        assert result_df["num_int"].tolist() == [1, 2, 3]
        assert result_df["num_float"].tolist() == [1.5, 2.5, 3.5]
        assert pd.isna(result_df.loc[0, "text"])
        assert result_df.loc[1, "text"] == "valid"
    
    def test_empty_tokens_no_replacement(self):
        """Test that empty tokens list results in no replacements"""
        df = pd.DataFrame({
            "col": ["N/A", "null", "value"]
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame, tokens=[])
        result_df = ar.to_pandas(result)
        
        pd.testing.assert_frame_equal(result_df, df)
    
    def test_exact_match_only(self):
        """Test that only exact matches are replaced, not substrings"""
        df = pd.DataFrame({
            "col": ["N/A", "N/A suffix", "prefix N/A", "not_null", "valid"]
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)
        
        assert pd.isna(result_df.loc[0, "col"])  # Exact match
        assert result_df.loc[1, "col"] == "N/A suffix"  # Not exact
        assert result_df.loc[2, "col"] == "prefix N/A"  # Not exact
        assert result_df.loc[3, "col"] == "not_null"
        assert result_df.loc[4, "col"] == "valid"
    
    def test_preserves_other_column_values(self):
        """Test that non-targeted columns retain all values"""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["N/A", "Bob", "N/A"],
            "score": ["A", "N/A", "C"]
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame, subset=["name"])
        result_df = ar.to_pandas(result)
        
        assert result_df["id"].tolist() == [1, 2, 3]
        assert result_df["score"].tolist() == ["A", "N/A", "C"]
        assert pd.isna(result_df.loc[0, "name"])
        assert result_df.loc[1, "name"] == "Bob"
