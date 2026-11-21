import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestDropEmptyColumns:
    """Test suite for arnio.cleaning.drop_empty_columns function."""
    
    def test_no_empty_columns(self):
        """DataFrame with no empty columns should be unchanged."""
        df = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": ["a", "b", "c"]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        
        assert set(result.columns) == {"col1", "col2"}
        assert result.shape[0] == 3
    
    def test_remove_all_null_column(self):
        """Column with all null values should be removed."""
        df = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": [None, None, None],
            "col3": ["a", "b", "c"]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        
        assert set(result.columns) == {"col1", "col3"}
        assert result.shape[0] == 3
    
    def test_remove_all_empty_strings(self):
        """Column with all empty strings should be removed."""
        df = pd.DataFrame({
            "col1": [1, 2],
            "col2": ["", "", ""],
            "col3": ["x", "y", "z"]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        
        assert set(result.columns) == {"col1", "col3"}
    
    def test_remove_all_whitespace_strings(self):
        """Column with only whitespace strings should be removed."""
        df = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": ["   ", "\t", "\n"],
            "col3": ["a", "b", "c"]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        
        assert set(result.columns) == {"col1", "col3"}
    
    def test_mixed_null_and_empty_string(self):
        """Column with both nulls and empty strings should be removed."""
        df = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": [None, "", None],
            "col3": ["x", "y", "z"]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        
        assert set(result.columns) == {"col1", "col3"}
    
    def test_preserve_column_with_some_data(self):
        """Column with at least one non-empty value should be preserved."""
        df = pd.DataFrame({
            "col1": [None, None, "value"],
            "col2": ["", "", ""]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        
        assert "col1" in result.columns
        assert "col2" not in result.columns
    
    def test_empty_dataframe_zero_rows(self):
        """Empty dataframe (0 rows) should preserve column structure."""
        df = pd.DataFrame({
            "col1": pd.Series([], dtype=object),
            "col2": pd.Series([], dtype=object),
            "col3": pd.Series([], dtype=object)
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        
        assert result.shape[0] == 0
        assert set(result.columns) == {"col1", "col2", "col3"}
    
    def test_all_columns_empty(self):
        """When all columns are empty, result should have 0 columns."""
        df = pd.DataFrame({
            "col1": [None, None],
            "col2": ["", ""],
            "col3": ["  ", "\t"]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        
        assert result.shape[1] == 0
        assert result.shape[0] == 2
    
    def test_single_row_single_null(self):
        """Single row with null value in a column."""
        df = pd.DataFrame({
            "col1": [1],
            "col2": [None]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        
        assert list(result.columns) == ["col1"]
        assert result.shape[0] == 1
    
    def test_single_row_whitespace(self):
        """Single row with whitespace value."""
        df = pd.DataFrame({
            "col1": ["value"],
            "col2": ["   "]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        
        assert list(result.columns) == ["col1"]
    
    def test_preserve_strings_with_content(self):
        """Strings containing non-whitespace content should be preserved."""
        df = pd.DataFrame({
            "col1": ["hello world", "  a  "],
            "col2": ["", ""]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        
        assert "col1" in result.columns
        assert "col2" not in result.columns
    
    def test_column_order_preserved(self):
        """Order of columns should be preserved in result."""
        df = pd.DataFrame({
            "a": [1, 2],
            "b": [None, None],
            "c": [3, 4],
            "d": ["", ""],
            "e": [5, 6]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        
        assert list(result.columns) == ["a", "c", "e"]
    
    def test_numeric_zero_preserved(self):
        """Numeric zero should not be treated as empty."""
        df = pd.DataFrame({
            "col1": [0, 0, 0],
            "col2": [None, None, None]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        
        assert "col1" in result.columns
        assert "col2" not in result.columns
    
    def test_boolean_false_preserved(self):
        """Boolean False should not be treated as empty."""
        df = pd.DataFrame({
            "col1": [False, False, False],
            "col2": [None, None, None]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        
        assert "col1" in result.columns
        assert "col2" not in result.columns
