import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestDropEmptyColumns:
    """Test suite for ar.drop_empty_columns()"""

    def test_drop_single_null_column(self):
        """Remove a column containing only null values."""
        df = pd.DataFrame({
            "name": ["Alice", "Bob"],
            "empty": [None, None],
            "age": [25, 30]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        result_df = ar.to_pandas(result)
        
        assert list(result_df.columns) == ["name", "age"]
        assert result_df.shape[0] == 2

    def test_drop_column_with_whitespace_only(self):
        """Remove column containing only whitespace strings."""
        df = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie"],
            "spaces": ["   ", "\t", "  \n  "],
            "age": [25, 30, 35]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        result_df = ar.to_pandas(result)
        
        assert list(result_df.columns) == ["name", "age"]

    def test_drop_multiple_empty_columns(self):
        """Remove multiple empty columns from same frame."""
        df = pd.DataFrame({
            "name": ["Alice", "Bob"],
            "empty1": [None, None],
            "age": [25, 30],
            "empty2": ["", ""],
            "city": ["NY", "LA"]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        result_df = ar.to_pandas(result)
        
        assert sorted(result_df.columns) == ["age", "city", "name"]
        assert result_df.shape[0] == 2

    def test_no_empty_columns(self):
        """Preserve all columns when none are empty."""
        df = pd.DataFrame({
            "name": ["Alice", "Bob"],
            "age": [25, 30],
            "active": [True, False]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        result_df = ar.to_pandas(result)
        
        assert set(result_df.columns) == {"name", "age", "active"}
        assert result_df.shape == (2, 3)

    def test_all_columns_empty(self):
        """Handle case where all columns are empty."""
        df = pd.DataFrame({
            "col1": [None, None, None],
            "col2": ["", "", ""],
            "col3": ["  ", "\t", "\n"]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        result_df = ar.to_pandas(result)
        
        assert result_df.shape[1] == 0
        assert result_df.shape[0] == 3

    def test_empty_frame_zero_rows(self):
        """Preserve column structure for empty frame."""
        df = pd.DataFrame({
            "name": pd.Series([], dtype="object"),
            "empty": pd.Series([], dtype="object"),
            "age": pd.Series([], dtype="object")
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        result_df = ar.to_pandas(result)
        
        assert result_df.shape[0] == 0
        assert set(result_df.columns) == {"name", "empty", "age"}

    def test_mixed_null_and_whitespace_in_column(self):
        """Remove column with mix of null and whitespace values."""
        df = pd.DataFrame({
            "data": ["value", "text", "content"],
            "empty": [None, "  ", ""]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        result_df = ar.to_pandas(result)
        
        assert list(result_df.columns) == ["data"]

    def test_single_row_frame(self):
        """Handle single-row frame correctly."""
        df = pd.DataFrame({
            "name": ["Alice"],
            "empty": [None],
            "age": [25]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        result_df = ar.to_pandas(result)
        
        assert set(result_df.columns) == {"name", "age"}
        assert result_df.shape[0] == 1

    def test_column_with_one_nonempty_value(self):
        """Keep column with even a single non-empty value."""
        df = pd.DataFrame({
            "mostly_empty": [None, None, "value", None],
            "truly_empty": [None, "", "  ", None]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        result_df = ar.to_pandas(result)
        
        assert list(result_df.columns) == ["mostly_empty"]

    def test_preserves_data_integrity(self):
        """Ensure non-empty column data is unchanged."""
        df = pd.DataFrame({
            "id": [1, 2, 3, 4],
            "name": ["Alice", "Bob", "Charlie", "David"],
            "empty_col": [None, None, None, None]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        result_df = ar.to_pandas(result)
        
        pd.testing.assert_series_equal(result_df["id"], df["id"], check_names=True)
        pd.testing.assert_series_equal(result_df["name"], df["name"], check_names=True)

    def test_column_order_preserved(self):
        """Verify column order matches original when dropping middle columns."""
        df = pd.DataFrame({
            "first": ["a", "b"],
            "empty1": [None, None],
            "second": ["c", "d"],
            "empty2": ["", ""],
            "third": ["e", "f"]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        result_df = ar.to_pandas(result)
        
        assert list(result_df.columns) == ["first", "second", "third"]

    def test_numeric_nan_column(self):
        """Handle numeric column with all NaN values."""
        df = pd.DataFrame({
            "numbers": [np.nan, np.nan, np.nan],
            "values": [1, 2, 3]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        result_df = ar.to_pandas(result)
        
        assert list(result_df.columns) == ["values"]

    def test_returns_new_frame(self):
        """Function returns a new ArFrame object."""
        df = pd.DataFrame({
            "name": ["Alice"],
            "empty": [None]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        
        assert frame is not result

    def test_column_with_zero_string(self):
        """Keep column with numeric string '0'."""
        df = pd.DataFrame({
            "data": ["0", "1", "0"],
            "empty": [None, None, None]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        result_df = ar.to_pandas(result)
        
        assert list(result_df.columns) == ["data"]

    def test_single_empty_column_only(self):
        """Handle frame with only one empty column."""
        df = pd.DataFrame({
            "empty": ["", None, "   "]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        result_df = ar.to_pandas(result)
        
        assert result_df.shape[1] == 0
        assert result_df.shape[0] == 3
