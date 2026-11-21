import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestDropConstantColumns:
    """Tests for arnio.cleaning.drop_constant_columns."""

    def test_drop_single_constant_numeric_column_pandas(self):
        """Remove pandas column with one unique numeric value."""
        df = pd.DataFrame({'A': [1, 1, 1], 'B': [1, 2, 3]})
        result = ar.drop_constant_columns(df)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ['B']
        assert len(result) == 3

    def test_drop_single_constant_string_column_pandas(self):
        """Remove pandas column with one unique string value."""
        df = pd.DataFrame({'A': ['x', 'x', 'x'], 'B': ['a', 'b', 'c']})
        result = ar.drop_constant_columns(df)
        assert list(result.columns) == ['B']

    def test_keep_multi_value_columns_pandas(self):
        """Keep pandas columns with multiple unique values."""
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [1, 1, 2]})
        result = ar.drop_constant_columns(df)
        assert set(result.columns) == {'A', 'B'}

    def test_constant_all_nulls_dropped_pandas(self):
        """Drop pandas column containing only None (1 unique value)."""
        df = pd.DataFrame({'A': [None, None, None], 'B': [1, 2, 3]})
        result = ar.drop_constant_columns(df)
        assert list(result.columns) == ['B']

    def test_keep_value_with_nulls_pandas(self):
        """Keep pandas column with value and nulls (2 unique: value + None)."""
        df = pd.DataFrame({'A': [1, 1, None], 'B': [1, 2, 3]})
        result = ar.drop_constant_columns(df)
        assert set(result.columns) == {'A', 'B'}

    def test_empty_frame_preserves_columns_pandas(self):
        """Zero-row pandas frame keeps all columns (0 unique values ≠ 1)."""
        df = pd.DataFrame({'A': pd.Series([], dtype='int64'),
                          'B': pd.Series([], dtype='int64')})
        result = ar.drop_constant_columns(df)
        assert list(result.columns) == ['A', 'B']
        assert len(result) == 0

    def test_all_constant_removes_all_preserves_rows_pandas(self):
        """When all columns constant, preserve row count with zero columns."""
        df = pd.DataFrame({'A': [1, 1, 1], 'B': [2, 2, 2]})
        result = ar.drop_constant_columns(df)
        assert len(result.columns) == 0
        assert len(result) == 3

    def test_drop_constant_arframe(self):
        """Remove constant column from ArFrame."""
        frame = ar.from_dict({'A': [1, 1, 1], 'B': [1, 2, 3]})
        result = ar.drop_constant_columns(frame)
        assert isinstance(result, ar.frame.ArFrame)
        assert list(result.columns) == ['B']

    def test_single_row_all_constant_pandas(self):
        """Single-row frame makes all columns constant."""
        df = pd.DataFrame({'A': [5], 'B': [10], 'C': [15]})
        result = ar.drop_constant_columns(df)
        assert len(result.columns) == 0
        assert len(result) == 1

    def test_multiple_constant_columns_pandas(self):
        """Drop multiple constant columns, keep varying ones."""
        df = pd.DataFrame({
            'A': [1, 1, 1],
            'B': [2, 2, 2],
            'C': [1, 2, 3],
            'D': [5, 5, 5]
        })
        result = ar.drop_constant_columns(df)
        assert list(result.columns) == ['C']

    def test_value_and_nan_not_constant_pandas(self):
        """Column with single value and NaN has 2 unique values, not dropped."""
        df = pd.DataFrame({'A': [5, 5, np.nan], 'B': [1, 2, 3]})
        result = ar.drop_constant_columns(df)
        assert set(result.columns) == {'A', 'B'}

    def test_pandas_input_returns_pandas_type(self):
        """pd.DataFrame input should return pd.DataFrame type."""
        df = pd.DataFrame({'X': [1, 1, 1]})
        result = ar.drop_constant_columns(df)
        assert isinstance(result, pd.DataFrame)
        assert not isinstance(result, ar.frame.ArFrame)

    def test_arframe_input_returns_arframe_type(self):
        """ArFrame input should return ArFrame type."""
        frame = ar.from_dict({'X': [1, 1, 1]})
        result = ar.drop_constant_columns(frame)
        assert isinstance(result, ar.frame.ArFrame)


class TestDropEmptyColumns:
    """Tests for arnio.cleaning.drop_empty_columns."""

    def test_drop_all_null_column(self):
        """Remove column with all null values."""
        frame = ar.from_dict({'A': [None, None, None], 'B': [1, 2, 3]})
        result = ar.drop_empty_columns(frame)
        assert list(result.columns) == ['B']
        assert result.shape[0] == 3

    def test_drop_all_empty_strings(self):
        """Remove column with all empty strings."""
        frame = ar.from_dict({'A': ['', '', ''], 'B': ['a', 'b', 'c']})
        result = ar.drop_empty_columns(frame)
        assert list(result.columns) == ['B']

    def test_drop_whitespace_only_strings(self):
        """Remove column where all strings are whitespace-only."""
        frame = ar.from_dict({'A': [' ', '  ', '\t'], 'B': [1, 2, 3]})
        result = ar.drop_empty_columns(frame)
        assert list(result.columns) == ['B']

    def test_keep_nonempty_columns(self):
        """Keep columns with actual values."""
        frame = ar.from_dict({'A': ['x', 'y', 'z'], 'B': [1, 2, 3]})
        result = ar.drop_empty_columns(frame)
        assert set(result.columns) == {'A', 'B'}

    def test_drop_mixed_null_empty_whitespace(self):
        """Remove column with mixed nulls, empty strings, whitespace."""
        frame = ar.from_dict({'A': [None, '', '  '], 'B': [1, 2, 3]})
        result = ar.drop_empty_columns(frame)
        assert list(result.columns) == ['B']

    def test_keep_column_with_one_value(self):
        """Keep column if it has at least one non-empty value."""
        frame = ar.from_dict({'A': [None, 'x', None], 'B': [1, 2, 3]})
        result = ar.drop_empty_columns(frame)
        assert set(result.columns) == {'A', 'B'}

    def test_zero_row_frame_preserves_columns(self):
        """Zero-row frame should preserve column structure."""
        frame = ar.from_dict({'A': [], 'B': []})
        result = ar.drop_empty_columns(frame)
        assert result.shape == (0, 2)
        assert set(result.columns) == {'A', 'B'}

    def test_all_empty_columns_removes_all(self):
        """Remove all columns if all empty, but preserve row count."""
        frame = ar.from_dict({'A': [None, None], 'B': ['', '']})
        result = ar.drop_empty_columns(frame)
        assert result.shape == (2, 0)

    def test_returns_arframe_type(self):
        """drop_empty_columns always returns ArFrame."""
        frame = ar.from_dict({'A': [1, 2, 3], 'B': ['x', 'y', 'z']})
        result = ar.drop_empty_columns(frame)
        assert isinstance(result, ar.frame.ArFrame)

    def test_multiple_empty_columns_mixed_types(self):
        """Drop multiple empty columns with different empty indicators."""
        frame = ar.from_dict({
            'A': [None, None, None],
            'B': ['', '', ''],
            'C': [1, 2, 3],
            'D': [' ', '\t', '  ']
        })
        result = ar.drop_empty_columns(frame)
        assert list(result.columns) == ['C']

    def test_preserves_column_order(self):
        """Remaining columns maintain their original order."""
        frame = ar.from_dict({
            'X': [1, 2],
            'Y': [None, None],
            'Z': ['a', 'b']
        })
        result = ar.drop_empty_columns(frame)
        assert list(result.columns) == ['X', 'Z']
