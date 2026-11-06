import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestDropEmptyColumns:
    """Tests for ar.drop_empty_columns()"""

    def test_drop_all_null_column(self):
        """Should remove a column containing only null values."""
        frame = ar.from_pandas(pd.DataFrame({
            "id": [1, 2, 3],
            "all_nulls": [None, None, None],
            "name": ["a", "b", "c"]
        }))
        result = ar.drop_empty_columns(frame)
        
        assert result.shape == (3, 2)
        assert "all_nulls" not in result.columns
        assert set(result.columns) == {"id", "name"}

    def test_drop_all_empty_strings_column(self):
        """Should remove a column containing only empty strings."""
        frame = ar.from_pandas(pd.DataFrame({
            "id": [1, 2, 3],
            "all_empty": ["", "", ""],
            "name": ["a", "b", "c"]
        }))
        result = ar.drop_empty_columns(frame)
        
        assert result.shape == (3, 2)
        assert "all_empty" not in result.columns

    def test_drop_whitespace_only_column(self):
        """Should treat whitespace-only strings as empty."""
        frame = ar.from_pandas(pd.DataFrame({
            "id": [1, 2, 3],
            "whitespace": [" ", "  ", "\t"],
            "name": ["a", "b", "c"]
        }))
        result = ar.drop_empty_columns(frame)
        
        assert result.shape == (3, 2)
        assert "whitespace" not in result.columns
        assert "name" in result.columns

    def test_drop_mixed_null_and_empty_column(self):
        """Should remove column with mix of nulls and empty strings."""
        frame = ar.from_pandas(pd.DataFrame({
            "id": [1, 2, 3],
            "mixed": [None, "", "  "],
            "name": ["a", "b", "c"]
        }))
        result = ar.drop_empty_columns(frame)
        
        assert result.shape == (3, 2)
        assert "mixed" not in result.columns

    def test_keep_column_with_single_value(self):
        """Should keep columns with at least one non-empty value."""
        frame = ar.from_pandas(pd.DataFrame({
            "id": [1, 2, 3],
            "mostly_empty": [None, "x", None],
            "name": ["a", "b", "c"]
        }))
        result = ar.drop_empty_columns(frame)
        
        assert result.shape == (3, 3)
        assert "mostly_empty" in result.columns

    def test_no_empty_columns(self):
        """Should return unchanged frame when no columns are empty."""
        frame = ar.from_pandas(pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["a", "b", "c"],
            "score": [10.5, 20.5, 30.5]
        }))
        result = ar.drop_empty_columns(frame)
        
        assert result.shape == (3, 3)
        assert set(result.columns) == {"id", "name", "score"}

    def test_all_columns_empty(self):
        """Should return frame with no columns when all are empty."""
        frame = ar.from_pandas(pd.DataFrame({
            "col1": [None, None, None],
            "col2": ["", "", ""],
            "col3": [" ", "  ", "\t"]
        }))
        result = ar.drop_empty_columns(frame)
        
        assert result.shape == (3, 0)
        assert len(result.columns) == 0

    def test_empty_frame_zero_rows(self):
        """Should keep all columns when frame has 0 rows (can't determine emptiness)."""
        frame = ar.from_pandas(pd.DataFrame({
            "col1": pd.Series([], dtype="object"),
            "col2": pd.Series([], dtype="object"),
            "col3": pd.Series([], dtype="object")
        }))
        result = ar.drop_empty_columns(frame)
        
        # When frame is empty, columns are kept since emptiness is indeterminate
        assert result.shape == (0, 3)
        assert set(result.columns) == {"col1", "col2", "col3"}

    def test_single_row_frame(self):
        """Should work correctly with single-row frame."""
        frame = ar.from_pandas(pd.DataFrame({
            "id": [1],
            "all_null": [None],
            "name": ["a"]
        }))
        result = ar.drop_empty_columns(frame)
        
        assert result.shape == (1, 2)
        assert set(result.columns) == {"id", "name"}

    def test_preserves_data_values(self):
        """Should preserve values in non-empty columns."""
        frame = ar.from_pandas(pd.DataFrame({
            "id": [1, 2, 3],
            "empty": [None, None, None],
            "name": ["alice", "bob", "charlie"],
            "score": [95.5, 87.0, 92.3]
        }))
        result = ar.drop_empty_columns(frame)
        result_df = ar.to_pandas(result)
        
        assert list(result_df["id"]) == [1, 2, 3]
        assert list(result_df["name"]) == ["alice", "bob", "charlie"]
        assert list(result_df["score"]) == [95.5, 87.0, 92.3]

    def test_preserves_column_order(self):
        """Should preserve order of remaining columns."""
        frame = ar.from_pandas(pd.DataFrame({
            "a": [1, 2, 3],
            "b": [None, None, None],
            "c": ["x", "y", "z"],
            "d": ["", "", ""],
            "e": [10, 20, 30]
        }))
        result = ar.drop_empty_columns(frame)
        
        assert result.columns == ["a", "c", "e"]

    def test_numeric_column_all_nan(self):
        """Should remove numeric column containing only NaNs."""
        frame = ar.from_pandas(pd.DataFrame({
            "id": [1, 2, 3],
            "all_nan": [np.nan, np.nan, np.nan],
            "value": [10.5, 20.5, 30.5]
        }))
        result = ar.drop_empty_columns(frame)
        
        assert result.shape == (3, 2)
        assert "all_nan" not in result.columns

    def test_mixed_type_column_non_empty(self):
        """Should keep columns with mixed types if they have non-empty values."""
        frame = ar.from_pandas(pd.DataFrame({
            "mixed": [1, "string", None],
            "all_null": [None, None, None]
        }))
        result = ar.drop_empty_columns(frame)
        
        assert result.shape == (3, 1)
        assert result.columns == ["mixed"]

    def test_various_whitespace_characters(self):
        """Should treat all whitespace characters (newline, tab) as empty."""
        frame = ar.from_pandas(pd.DataFrame({
            "id": [1, 2, 3, 4],
            "whitespace": ["\n", "\t", "\r\n", "\v"],
            "name": ["a", "b", "c", "d"]
        }))
        result = ar.drop_empty_columns(frame)
        
        assert result.shape == (4, 2)
        assert "whitespace" not in result.columns

    def test_returns_new_frame_not_mutated_input(self):
        """Should return a new frame without modifying the input frame."""
        original = ar.from_pandas(pd.DataFrame({
            "id": [1, 2, 3],
            "empty": [None, None, None],
            "name": ["a", "b", "c"]
        }))
        
        original_columns_count = len(original.columns)
        result = ar.drop_empty_columns(original)
        
        # Input should be unchanged
        assert len(original.columns) == original_columns_count
        assert "empty" in original.columns
        # Result should differ
        assert "empty" not in result.columns
