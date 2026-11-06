import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestDropConstantColumns:
    """Tests for drop_constant_columns function."""

    def test_drop_single_constant_column(self):
        """Remove a column with exactly one unique value."""
        df = pd.DataFrame({
            "const": [1, 1, 1],
            "var": [1, 2, 3],
        })
        frame = ar.from_pandas(df)
        result = ar.drop_constant_columns(frame)
        
        assert result.columns == ["var"]

    def test_keep_column_with_null_and_value(self):
        """Keep column with one value and nulls (2 unique values when counting None)."""
        df = pd.DataFrame({
            "const_with_null": [1, 1, None],
            "var": [1, 2, 3],
        })
        frame = ar.from_pandas(df)
        result = ar.drop_constant_columns(frame)
        
        assert "const_with_null" in result.columns
        assert "var" in result.columns

    def test_drop_all_nulls_column(self):
        """Drop column with only nulls (1 unique value: None)."""
        df = pd.DataFrame({
            "all_null": [None, None, None],
            "var": [1, 2, 3],
        })
        frame = ar.from_pandas(df)
        result = ar.drop_constant_columns(frame)
        
        assert result.columns == ["var"]

    def test_multiple_constant_columns(self):
        """Remove multiple constant columns."""
        df = pd.DataFrame({
            "const1": ["a", "a", "a"],
            "const2": [100, 100, 100],
            "var": [1, 2, 3],
        })
        frame = ar.from_pandas(df)
        result = ar.drop_constant_columns(frame)
        
        assert result.columns == ["var"]

    def test_all_columns_constant_preserves_row_count(self):
        """If all columns are constant, preserve row count with zero columns."""
        df = pd.DataFrame({
            "const1": [1, 1, 1],
            "const2": ["x", "x", "x"],
        })
        frame = ar.from_pandas(df)
        result = ar.drop_constant_columns(frame)
        
        assert result.shape == (3, 0)

    def test_pandas_dataframe_input_and_output(self):
        """Accept and return pandas DataFrame (not ArFrame)."""
        df = pd.DataFrame({
            "const": [5, 5, 5],
            "var": ["a", "b", "c"],
        })
        result = ar.drop_constant_columns(df)
        
        assert isinstance(result, pd.DataFrame)
        assert result.columns.tolist() == ["var"]

    def test_empty_frame_with_columns(self):
        """Zero-row frame preserves all columns (0 unique values, not 1)."""
        df = pd.DataFrame({
            "col1": pd.Series([], dtype=int),
            "col2": pd.Series([], dtype=str),
        })
        frame = ar.from_pandas(df)
        result = ar.drop_constant_columns(frame)
        
        assert result.shape == (0, 2)


class TestDropEmptyColumns:
    """Tests for drop_empty_columns function."""

    def test_drop_all_nulls_column(self):
        """Remove column containing only None values."""
        df = pd.DataFrame({
            "empty": [None, None, None],
            "filled": [1, 2, 3],
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        
        assert result.columns == ["filled"]

    def test_drop_empty_strings_and_nulls(self):
        """Remove column with only empty strings and nulls."""
        df = pd.DataFrame({
            "empty": ["", None, ""],
            "filled": ["a", "b", "c"],
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        
        assert result.columns == ["filled"]

    def test_treat_whitespace_only_as_empty(self):
        """Column with only whitespace (spaces, tabs, newlines) is empty."""
        df = pd.DataFrame({
            "spaces": ["   ", "\t", "\n"],
            "filled": ["x", "y", "z"],
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        
        assert result.columns == ["filled"]

    def test_keep_column_with_any_content(self):
        """Keep column if at least one non-empty value exists."""
        df = pd.DataFrame({
            "mixed": ["", None, "value"],
            "filled": [1, 2, 3],
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        
        assert set(result.columns) == {"mixed", "filled"}

    def test_all_columns_empty_returns_zero_columns(self):
        """If all columns are empty, return zero-column frame preserving rows."""
        df = pd.DataFrame({
            "empty1": [None, None],
            "empty2": ["", ""],
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        
        assert result.shape == (2, 0)

    def test_empty_frame_unchanged(self):
        """Zero-row frame returns unchanged (no rows to check emptiness)."""
        df = pd.DataFrame({
            "col1": pd.Series([], dtype=object),
            "col2": pd.Series([], dtype=object),
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        
        assert result.shape == (0, 2)

    def test_no_empty_columns(self):
        """No columns removed when none are empty."""
        df = pd.DataFrame({
            "col1": ["a", "b"],
            "col2": [1, 2],
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        
        assert result.columns.tolist() == ["col1", "col2"]
