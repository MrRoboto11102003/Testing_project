import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestNormalizeWhitespace:
    """Test suite for arnio.cleaning.normalize_whitespace function."""
    
    def test_collapse_multiple_spaces(self):
        """Test collapsing multiple internal spaces to single space."""
        df = pd.DataFrame({
            "name": ["John  Doe", "Jane    Smith"]
        })
        result = ar.cleaning.normalize_whitespace(df)
        assert result["name"].iloc[0] == "John Doe"
        assert result["name"].iloc[1] == "Jane Smith"
    
    def test_strip_leading_whitespace(self):
        """Test removal of leading whitespace including tabs."""
        df = pd.DataFrame({
            "text": ["  hello", "\t\tworld"]
        })
        result = ar.cleaning.normalize_whitespace(df)
        assert result["text"].iloc[0] == "hello"
        assert result["text"].iloc[1] == "world"
    
    def test_strip_trailing_whitespace(self):
        """Test removal of trailing whitespace including tabs."""
        df = pd.DataFrame({
            "text": ["hello  ", "world\t\t"]
        })
        result = ar.cleaning.normalize_whitespace(df)
        assert result["text"].iloc[0] == "hello"
        assert result["text"].iloc[1] == "world"
    
    def test_normalize_mixed_whitespace_types(self):
        """Test collapsing mixed whitespace types (spaces, tabs, newlines)."""
        df = pd.DataFrame({
            "text": ["hello \t\n world", "foo  \t  bar"]
        })
        result = ar.cleaning.normalize_whitespace(df)
        assert result["text"].iloc[0] == "hello world"
        assert result["text"].iloc[1] == "foo bar"
    
    def test_process_specified_columns_only(self):
        """Test that only specified columns are processed."""
        df = pd.DataFrame({
            "name": ["  John  Doe  "],
            "age": ["  30  "]
        })
        result = ar.cleaning.normalize_whitespace(df, columns=["name"])
        assert result["name"].iloc[0] == "John Doe"
        assert result["age"].iloc[0] == "  30  "
    
    def test_default_processes_all_string_columns(self):
        """Test that all string columns are processed by default."""
        df = pd.DataFrame({
            "name": ["  John  Doe  "],
            "city": ["  New  York  "],
            "age": [25]
        })
        result = ar.cleaning.normalize_whitespace(df)
        assert result["name"].iloc[0] == "John Doe"
        assert result["city"].iloc[0] == "New York"
        assert result["age"].iloc[0] == 25
    
    def test_pass_through_non_string_values(self):
        """Test that non-string values in object columns pass through unchanged."""
        df = pd.DataFrame({
            "mixed": ["  hello  ", 123, None, 45.67, True]
        })
        result = ar.cleaning.normalize_whitespace(df)
        assert result["mixed"].iloc[0] == "hello"
        assert result["mixed"].iloc[1] == 123
        assert pd.isna(result["mixed"].iloc[2])
        assert result["mixed"].iloc[3] == 45.67
        assert result["mixed"].iloc[4] is True
    
    def test_null_values_preserved(self):
        """Test that null/None values are preserved unchanged."""
        df = pd.DataFrame({
            "text": ["  hello  ", None, "  world  ", np.nan]
        })
        result = ar.cleaning.normalize_whitespace(df)
        assert result["text"].iloc[0] == "hello"
        assert pd.isna(result["text"].iloc[1])
        assert result["text"].iloc[2] == "world"
        assert pd.isna(result["text"].iloc[3])
    
    def test_internal_single_spaces_preserved(self):
        """Test that single spaces within words are preserved."""
        df = pd.DataFrame({
            "text": ["hello world", "foo bar baz"]
        })
        result = ar.cleaning.normalize_whitespace(df)
        assert result["text"].iloc[0] == "hello world"
        assert result["text"].iloc[1] == "foo bar baz"
    
    def test_only_whitespace_becomes_empty(self):
        """Test that strings of only whitespace become empty strings."""
        df = pd.DataFrame({
            "text": ["   ", "\t\t\t", "  \n  "]
        })
        result = ar.cleaning.normalize_whitespace(df)
        assert result["text"].iloc[0] == ""
        assert result["text"].iloc[1] == ""
        assert result["text"].iloc[2] == ""
    
    def test_empty_string_unchanged(self):
        """Test that empty strings remain empty."""
        df = pd.DataFrame({
            "text": [""]
        })
        result = ar.cleaning.normalize_whitespace(df)
        assert result["text"].iloc[0] == ""
    
    def test_invalid_column_raises_error(self):
        """Test that specifying non-existent columns raises ValueError."""
        df = pd.DataFrame({
            "name": ["  John  "],
            "age": [25]
        })
        with pytest.raises(ValueError, match="Missing columns"):
            ar.cleaning.normalize_whitespace(df, columns=["nonexistent"])
    
    def test_multiple_columns_specified(self):
        """Test processing multiple specified columns while leaving others unchanged."""
        df = pd.DataFrame({
            "first": ["  John  "],
            "last": ["  Doe  "],
            "age": ["  30  "]
        })
        result = ar.cleaning.normalize_whitespace(df, columns=["first", "last"])
        assert result["first"].iloc[0] == "John"
        assert result["last"].iloc[0] == "Doe"
        assert result["age"].iloc[0] == "  30  "
    
    def test_returns_pandas_dataframe(self):
        """Test that pandas DataFrame input returns pandas DataFrame output."""
        df = pd.DataFrame({
            "text": ["  hello  world  "]
        })
        result = ar.cleaning.normalize_whitespace(df)
        assert isinstance(result, pd.DataFrame)
        assert result["text"].iloc[0] == "hello world"
