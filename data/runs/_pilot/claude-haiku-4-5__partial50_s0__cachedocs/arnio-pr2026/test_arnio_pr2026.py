import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestDropConstantColumns:
    """Test drop_constant_columns with various scenarios."""

    def test_drop_single_constant_column_from_arframe(self):
        """Should remove a single column with one unique value."""
        frame = ar.from_dict({
            "id": [1, 2, 3],
            "status": ["active", "active", "active"],
            "name": ["Alice", "Bob", "Charlie"]
        })
        result = ar.drop_constant_columns(frame)
        assert "status" not in result.columns
        assert "id" in result.columns
        assert "name" in result.columns
        assert result.shape == (3, 2)

    def test_drop_multiple_constant_columns(self):
        """Should remove multiple constant columns."""
        frame = ar.from_dict({
            "id": [1, 2, 3],
            "status": ["active", "active", "active"],
            "version": ["1.0", "1.0", "1.0"],
            "name": ["Alice", "Bob", "Charlie"]
        })
        result = ar.drop_constant_columns(frame)
        assert set(result.columns) == {"id", "name"}
        assert result.shape == (3, 2)

    def test_keep_all_non_constant_columns(self):
        """Should keep columns with multiple unique values."""
        frame = ar.from_dict({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "score": [100, 95, 88]
        })
        result = ar.drop_constant_columns(frame)
        assert set(result.columns) == {"id", "name", "score"}

    def test_all_columns_are_constant(self):
        """Should handle DataFrame where all columns are constant."""
        frame = ar.from_dict({
            "status": ["active", "active", "active"],
            "version": ["1.0", "1.0", "1.0"]
        })
        result = ar.drop_constant_columns(frame)
        assert len(result.columns) == 0

    def test_works_with_pandas_dataframe(self):
        """Should preserve DataFrame type when input is pandas DataFrame."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "const": ["x", "x", "x"],
            "name": ["A", "B", "C"]
        })
        result = ar.drop_constant_columns(df)
        assert isinstance(result, pd.DataFrame)
        assert "const" not in result.columns
        assert set(result.columns) == {"id", "name"}

    def test_constant_column_with_single_row(self):
        """Single-row frames: all columns technically constant (single value)."""
        frame = ar.from_dict({
            "a": [1],
            "b": ["x"],
            "c": [3.14]
        })
        result = ar.drop_constant_columns(frame)
        # All columns have only one unique value
        assert len(result.columns) == 0

    def test_constant_numeric_zero(self):
        """Should correctly identify numeric zero as a repeated constant value."""
        frame = ar.from_dict({
            "id": [1, 2, 3],
            "zeros": [0, 0, 0],
            "name": ["A", "B", "C"]
        })
        result = ar.drop_constant_columns(frame)
        assert "zeros" not in result.columns
        assert "id" in result.columns


class TestDropEmptyColumns:
    """Test drop_empty_columns with various scenarios."""

    def test_drop_single_all_null_column(self):
        """Should remove a column with all null values."""
        frame = ar.from_dict({
            "id": [1, 2, 3],
            "empty": [None, None, None],
            "name": ["Alice", "Bob", "Charlie"]
        })
        result = ar.drop_empty_columns(frame)
        assert "empty" not in result.columns
        assert "id" in result.columns
        assert "name" in result.columns

    def test_drop_multiple_empty_columns(self):
        """Should remove multiple all-null columns."""
        frame = ar.from_dict({
            "id": [1, 2, 3],
            "empty1": [None, None, None],
            "empty2": [None, None, None],
            "name": ["A", "B", "C"]
        })
        result = ar.drop_empty_columns(frame)
        assert set(result.columns) == {"id", "name"}

    def test_keep_columns_with_any_data(self):
        """Should keep columns that have at least one non-null value."""
        frame = ar.from_dict({
            "id": [1, 2, 3],
            "mostly_empty": [None, None, "value"],
            "name": ["A", "B", "C"]
        })
        result = ar.drop_empty_columns(frame)
        assert "mostly_empty" in result.columns
        assert set(result.columns) == {"id", "mostly_empty", "name"}

    def test_all_columns_are_empty(self):
        """Should handle DataFrame where all columns are empty."""
        frame = ar.from_dict({
            "empty1": [None, None, None],
            "empty2": [None, None, None]
        })
        result = ar.drop_empty_columns(frame)
        assert len(result.columns) == 0

    def test_sparse_column_with_single_value(self):
        """Column with mostly nulls but one actual value should be kept."""
        frame = ar.from_dict({
            "id": [1, 2, 3],
            "sparse": [None, "only_value", None],
            "empty": [None, None, None]
        })
        result = ar.drop_empty_columns(frame)
        assert "sparse" in result.columns
        assert "empty" not in result.columns
        assert "id" in result.columns

    def test_empty_frame(self):
        """Should handle completely empty frame gracefully."""
        frame = ar.from_dict({"a": [], "b": []})
        result = ar.drop_empty_columns(frame)
        # Empty frame with no rows - columns might be considered empty
        assert isinstance(result, type(frame))
        assert result.shape[0] == 0

    def test_column_with_mixed_numeric_types(self):
        """Column with null and numeric values should be kept."""
        frame = ar.from_dict({
            "id": [1, 2, 3],
            "scores": [100.5, None, 88.0],
            "empty": [None, None, None]
        })
        result = ar.drop_empty_columns(frame)
        assert "scores" in result.columns
        assert "empty" not in result.columns
