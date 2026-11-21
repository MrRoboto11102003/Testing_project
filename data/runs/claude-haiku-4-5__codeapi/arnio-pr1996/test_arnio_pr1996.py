import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestDropEmptyColumns:
    """Test suite for drop_empty_columns function."""

    def test_remove_fully_null_column(self):
        """Remove a column with all null values."""
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [None, None, None],
            'c': ['x', 'y', 'z']
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        assert result.columns == ['a', 'c']
        assert result.shape == (3, 2)

    def test_remove_fully_empty_string_column(self):
        """Remove a column with all empty strings."""
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': ['', '', ''],
            'c': ['x', 'y', 'z']
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        assert result.columns == ['a', 'c']
        assert result.shape == (3, 2)

    def test_remove_whitespace_only_column(self):
        """Treat whitespace-only strings as empty."""
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': ['   ', '\t', '\n'],
            'c': ['x', 'y', 'z']
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        assert result.columns == ['a', 'c']
        assert result.shape == (3, 2)

    def test_mixed_null_and_empty_string_column(self):
        """Remove column with mix of nulls and empty strings."""
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [None, '', None],
            'c': ['x', 'y', 'z']
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        assert result.columns == ['a', 'c']
        assert result.shape == (3, 2)

    def test_keep_partially_empty_column(self):
        """Columns with some non-empty values are preserved."""
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': ['x', '', None],
            'c': ['y', 'z', 'w']
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        assert result.columns == ['a', 'b', 'c']
        assert result.shape == (3, 3)

    def test_no_empty_columns(self):
        """Frame with no empty columns remains unchanged."""
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': ['x', 'y', 'z'],
            'c': [4.5, 6.7, 8.9]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        assert result.columns == ['a', 'b', 'c']
        assert result.shape == (3, 3)

    def test_all_columns_empty(self):
        """All columns are empty."""
        df = pd.DataFrame({
            'a': [None, None, None],
            'b': ['', '', ''],
            'c': ['  ', '  ', '  ']
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        assert result.columns == []
        assert result.shape == (3, 0)

    def test_single_empty_column(self):
        """Frame with only one empty column."""
        df = pd.DataFrame({
            'a': [None, None, None]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        assert result.columns == []
        assert result.shape == (1, 0)

    def test_empty_frame_zero_rows(self):
        """Empty frame with zero rows maintains columns."""
        df = pd.DataFrame({
            'a': pd.Series([], dtype='int64'),
            'b': pd.Series([], dtype='object'),
            'c': pd.Series([], dtype='float64')
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        assert result.columns == ['a', 'b', 'c']
        assert result.shape == (0, 3)

    def test_column_order_preserved(self):
        """Column order is preserved."""
        df = pd.DataFrame({
            'z': [1, 2, 3],
            'empty1': [None, None, None],
            'y': ['a', 'b', 'c'],
            'empty2': ['', '', ''],
            'x': [4.5, 6.7, 8.9]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        assert result.columns == ['z', 'y', 'x']

    def test_multiple_empty_columns(self):
        """Multiple empty columns are removed."""
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [None, None, None],
            'c': ['', '', ''],
            'd': ['x', 'y', 'z']
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        assert result.columns == ['a', 'd']
        assert result.shape == (3, 2)

    def test_nan_values_treated_as_empty(self):
        """NaN values are treated as empty."""
        df = pd.DataFrame({
            'a': [1.0, 2.0, 3.0],
            'b': [np.nan, np.nan, np.nan],
            'c': ['x', 'y', 'z']
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        assert result.columns == ['a', 'c']
        assert result.shape == (3, 2)

    def test_numeric_zero_not_empty(self):
        """Numeric zero values are not treated as empty."""
        df = pd.DataFrame({
            'a': [0, 1, 2],
            'b': [0.0, 0.0, 0.0],
            'c': [None, None, None]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        assert result.columns == ['a', 'b']
        assert result.shape == (3, 2)

    def test_single_row_frame(self):
        """Frame with single row."""
        df = pd.DataFrame({
            'a': [1],
            'b': [None],
            'c': ['x']
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        assert result.columns == ['a', 'c']
        assert result.shape == (1, 2)

    def test_data_values_preserved(self):
        """Data values in remaining columns are preserved."""
        df = pd.DataFrame({
            'keep': [10, 20, 30],
            'drop': [None, None, None],
            'keep2': ['a', 'b', 'c']
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        result_dict = result.to_dict(orient='list')
        assert result_dict['keep'] == [10, 20, 30]
        assert result_dict['keep2'] == ['a', 'b', 'c']
