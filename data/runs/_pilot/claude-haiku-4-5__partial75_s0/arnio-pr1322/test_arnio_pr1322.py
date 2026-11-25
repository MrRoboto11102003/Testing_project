import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestCoalesceColumns:
    """Tests for arnio.cleaning.coalesce_columns."""

    def test_basic_coalesce_first_non_null(self):
        """Test selecting first non-null value from multiple columns."""
        df = pd.DataFrame({
            "phone": ["123-456", None, None],
            "mobile": [None, "987-654", None],
            "fax": [None, None, "555-1234"],
        })
        frame = ar.from_pandas(df)
        result = ar.coalesce_columns(frame, subset=["phone", "mobile", "fax"])
        result_df = ar.to_pandas(result)
        
        assert "coalesced" in result_df.columns
        assert result_df["coalesced"].tolist() == ["123-456", "987-654", "555-1234"]

    def test_coalesce_custom_output_column(self):
        """Test custom output column name."""
        df = pd.DataFrame({
            "col1": ["a", None],
            "col2": [None, "b"],
        })
        frame = ar.from_pandas(df)
        result = ar.coalesce_columns(
            frame, 
            subset=["col1", "col2"], 
            output_column="contact"
        )
        result_df = ar.to_pandas(result)
        
        assert "contact" in result_df.columns
        assert result_df["contact"].tolist() == ["a", "b"]

    def test_coalesce_all_nulls(self):
        """Test when all values in subset are null."""
        df = pd.DataFrame({
            "a": [None, None],
            "b": [None, None],
        })
        frame = ar.from_pandas(df)
        result = ar.coalesce_columns(frame, subset=["a", "b"])
        result_df = ar.to_pandas(result)
        
        assert pd.isna(result_df["coalesced"].iloc[0])
        assert pd.isna(result_df["coalesced"].iloc[1])

    def test_coalesce_single_column(self):
        """Test coalesce with single column in subset."""
        df = pd.DataFrame({
            "name": ["Alice", None, "Charlie"],
        })
        frame = ar.from_pandas(df)
        result = ar.coalesce_columns(frame, subset=["name"])
        result_df = ar.to_pandas(result)
        
        assert result_df["coalesced"].tolist()[0] == "Alice"
        assert pd.isna(result_df["coalesced"].iloc[1])
        assert result_df["coalesced"].tolist()[2] == "Charlie"

    def test_coalesce_preserves_other_columns(self):
        """Test that coalesced column is added without removing others."""
        df = pd.DataFrame({
            "id": [1, 2],
            "phone": ["123", None],
            "mobile": [None, "456"],
        })
        frame = ar.from_pandas(df)
        result = ar.coalesce_columns(frame, subset=["phone", "mobile"])
        result_df = ar.to_pandas(result)
        
        assert all(col in result_df.columns for col in ["id", "phone", "mobile", "coalesced"])

    def test_coalesce_nonexistent_column_raises(self):
        """Test that referencing non-existent column raises error."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        frame = ar.from_pandas(df)
        
        with pytest.raises((ValueError, KeyError)):
            ar.coalesce_columns(frame, subset=["a", "nonexistent"])

    def test_coalesce_empty_subset_raises(self):
        """Test that empty subset raises error."""
        df = pd.DataFrame({"a": [1, 2]})
        frame = ar.from_pandas(df)
        
        with pytest.raises((ValueError, TypeError)):
            ar.coalesce_columns(frame, subset=[])

    def test_coalesce_respects_column_order(self):
        """Test that first column in subset takes priority."""
        df = pd.DataFrame({
            "col1": [None, "first"],
            "col2": ["second", "also_second"],
            "col3": ["third", "also_third"],
        })
        frame = ar.from_pandas(df)
        result = ar.coalesce_columns(frame, subset=["col1", "col2", "col3"])
        result_df = ar.to_pandas(result)
        
        assert result_df["coalesced"].iloc[0] == "second"
        assert result_df["coalesced"].iloc[1] == "first"


class TestStandardizeMissingTokens:
    """Tests for arnio.cleaning.standardize_missing_tokens."""

    def test_basic_standardize_default_tokens(self):
        """Test replacing default missing-value tokens with null."""
        df = pd.DataFrame({
            "name": ["Alice", "N/A", "MISSING", "Bob"],
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)
        
        assert result_df["name"].iloc[0] == "Alice"
        assert pd.isna(result_df["name"].iloc[1])
        assert pd.isna(result_df["name"].iloc[2])
        assert result_df["name"].iloc[3] == "Bob"

    def test_standardize_custom_tokens(self):
        """Test standardize with custom missing-value tokens."""
        df = pd.DataFrame({
            "value": ["data", "???", "---", "more_data"],
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame, tokens=["???", "---"])
        result_df = ar.to_pandas(result)
        
        assert result_df["value"].iloc[0] == "data"
        assert pd.isna(result_df["value"].iloc[1])
        assert pd.isna(result_df["value"].iloc[2])
        assert result_df["value"].iloc[3] == "more_data"

    def test_standardize_subset_columns(self):
        """Test standardize only on specified columns."""
        df = pd.DataFrame({
            "col1": ["data", "N/A"],
            "col2": ["N/A", "other"],
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame, subset=["col1"])
        result_df = ar.to_pandas(result)
        
        assert result_df["col1"].iloc[0] == "data"
        assert pd.isna(result_df["col1"].iloc[1])
        assert result_df["col2"].iloc[0] == "N/A"

    def test_standardize_no_matches(self):
        """Test when no matching tokens are found."""
        df = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie"],
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame, tokens=["???"])
        result_df = ar.to_pandas(result)
        
        assert result_df["name"].tolist() == ["Alice", "Bob", "Charlie"]

    def test_standardize_empty_string_token(self):
        """Test handling of empty strings as missing token."""
        df = pd.DataFrame({
            "value": ["data", "", "more"],
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame, tokens=[""])
        result_df = ar.to_pandas(result)
        
        assert result_df["value"].iloc[0] == "data"
        assert pd.isna(result_df["value"].iloc[1])
        assert result_df["value"].iloc[2] == "more"

    def test_standardize_preserves_other_columns(self):
        """Test that standardize doesn't affect unspecified columns."""
        df = pd.DataFrame({
            "id": [1, 2],
            "text": ["valid", "N/A"],
            "numeric": [10.5, 20.0],
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame, subset=["text"])
        result_df = ar.to_pandas(result)
        
        assert result_df["id"].tolist() == [1, 2]
        assert result_df["numeric"].tolist() == [10.5, 20.0]
        assert result_df["text"].iloc[0] == "valid"
        assert pd.isna(result_df["text"].iloc[1])
