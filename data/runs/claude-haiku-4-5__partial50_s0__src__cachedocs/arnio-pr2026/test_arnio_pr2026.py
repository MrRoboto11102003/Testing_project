import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestDropConstantColumns:
    """Tests for ar.drop_constant_columns()"""
    
    def test_drop_single_constant_column(self):
        """Single column with one unique value should be dropped."""
        df = pd.DataFrame({"const": [1, 1, 1], "var": [1, 2, 3]})
        frame = ar.from_pandas(df)
        result = ar.drop_constant_columns(frame)
        result_df = ar.to_pandas(result)
        
        assert "const" not in result_df.columns
        assert "var" in result_df.columns
        assert len(result_df) == 3
    
    def test_drop_all_null_column(self):
        """Column with all nulls should be dropped (1 unique value)."""
        df = pd.DataFrame({"all_null": [None, None, None], "other": [1, 2, 3]})
        frame = ar.from_pandas(df)
        result = ar.drop_constant_columns(frame)
        result_df = ar.to_pandas(result)
        
        assert "all_null" not in result_df.columns
        assert "other" in result_df.columns
    
    def test_keep_null_with_other_value(self):
        """[1, 1, None] should be kept (2 unique values: 1 and None)."""
        df = pd.DataFrame({"col": [1, 1, None]})
        frame = ar.from_pandas(df)
        result = ar.drop_constant_columns(frame)
        result_df = ar.to_pandas(result)
        
        assert "col" in result_df.columns
        assert len(result_df) == 3
    
    def test_keep_column_with_multiple_values(self):
        """Column with multiple unique values should be kept."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        frame = ar.from_pandas(df)
        result = ar.drop_constant_columns(frame)
        result_df = ar.to_pandas(result)
        
        assert "col" in result_df.columns
    
    def test_drop_all_constant_preserves_row_count(self):
        """When all columns constant, drop all but preserve row count."""
        df = pd.DataFrame({"a": [1, 1, 1], "b": [2, 2, 2]})
        frame = ar.from_pandas(df)
        result = ar.drop_constant_columns(frame)
        result_df = ar.to_pandas(result)
        
        assert len(result_df.columns) == 0
        assert len(result_df) == 3
    
    def test_zero_row_frame_keeps_all_columns(self):
        """Empty frame keeps all columns (0 unique values, not constant)."""
        df = pd.DataFrame({"a": [], "b": []})
        frame = ar.from_pandas(df)
        result = ar.drop_constant_columns(frame)
        result_df = ar.to_pandas(result)
        
        assert set(result_df.columns) == {"a", "b"}
    
    def test_mixed_constant_and_variable_columns(self):
        """Mix of constant and variable columns."""
        df = pd.DataFrame({
            "const1": [5, 5, 5],
            "var": [1, 2, 3],
            "const2": ["x", "x", "x"]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_constant_columns(frame)
        result_df = ar.to_pandas(result)
        
        assert result_df.columns.tolist() == ["var"]
    
    def test_pandas_dataframe_input_returns_dataframe(self):
        """Works with pd.DataFrame input and returns pd.DataFrame."""
        df = pd.DataFrame({"const": [1, 1], "var": [1, 2]})
        result = ar.drop_constant_columns(df)
        
        assert isinstance(result, pd.DataFrame)
        assert "const" not in result.columns
        assert "var" in result.columns


class TestDropEmptyColumns:
    """Tests for ar.drop_empty_columns()"""
    
    def test_drop_all_null_column(self):
        """Column with all nulls should be dropped."""
        df = pd.DataFrame({
            "all_null": [None, None],
            "has_value": [1, 2]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        result_df = ar.to_pandas(result)
        
        assert "all_null" not in result_df.columns
        assert "has_value" in result_df.columns
    
    def test_drop_all_empty_string_column(self):
        """Column with all empty strings should be dropped."""
        df = pd.DataFrame({
            "empty": ["", "", ""],
            "has_value": ["a", "b", "c"]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        result_df = ar.to_pandas(result)
        
        assert "empty" not in result_df.columns
        assert "has_value" in result_df.columns
    
    def test_drop_all_whitespace_column(self):
        """Column with all whitespace treated as empty should be dropped."""
        df = pd.DataFrame({
            "spaces": ["   ", " ", "  "],
            "has_value": ["a", "b", "c"]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        result_df = ar.to_pandas(result)
        
        assert "spaces" not in result_df.columns
        assert "has_value" in result_df.columns
    
    def test_keep_column_with_non_empty_value(self):
        """Column with at least one non-empty value should be kept."""
        df = pd.DataFrame({
            "mostly_empty": ["", "", "x"],
            "other": [1, 2, 3]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        result_df = ar.to_pandas(result)
        
        assert "mostly_empty" in result_df.columns
    
    def test_drop_mixed_null_and_empty_strings(self):
        """Column with only nulls and empty strings should be dropped."""
        df = pd.DataFrame({
            "mixed": [None, "", None, ""],
            "has_value": [1, 2, 3, 4]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        result_df = ar.to_pandas(result)
        
        assert "mixed" not in result_df.columns
        assert "has_value" in result_df.columns
    
    def test_drop_all_empty_preserves_row_count(self):
        """When all columns empty, drop all but preserve row count."""
        df = pd.DataFrame({
            "empty1": [None, None],
            "empty2": ["", ""]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        result_df = ar.to_pandas(result)
        
        assert len(result_df.columns) == 0
        assert len(result_df) == 2
    
    def test_zero_row_frame_keeps_columns(self):
        """Zero-row frame should keep all columns."""
        df = pd.DataFrame({"a": [], "b": []})
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        result_df = ar.to_pandas(result)
        
        assert set(result_df.columns) == {"a", "b"}
        assert len(result_df) == 0
