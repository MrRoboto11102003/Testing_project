import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestDropEmptyColumns:
    """Tests for arnio.cleaning.drop_empty_columns()"""

    def test_removes_column_with_all_nulls(self):
        """Column with entirely null values should be removed."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "empty": [None, None, None],
            "name": ["a", "b", "c"]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        
        assert "empty" not in result.columns
        assert set(result.columns) == {"id", "name"}

    def test_removes_column_with_all_empty_strings(self):
        """Column with entirely empty strings should be removed."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "empty": ["", "", ""],
            "name": ["a", "b", "c"]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        
        assert "empty" not in result.columns
        assert set(result.columns) == {"id", "name"}

    def test_removes_column_with_only_whitespace(self):
        """Column with only whitespace should be removed."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "whitespace": ["   ", "\t", "\n"],
            "name": ["a", "b", "c"]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        
        assert "whitespace" not in result.columns
        assert set(result.columns) == {"id", "name"}

    def test_keeps_column_with_single_nonempty_value(self):
        """Column with at least one non-empty value should be kept."""
        df = pd.DataFrame({
            "mostly_empty": ["", "", "x"],
            "id": [1, 2, 3]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        
        assert "mostly_empty" in result.columns
        assert "id" in result.columns

    def test_keeps_column_with_mixed_nulls_and_values(self):
        """Column with mix of nulls and non-empty values should be kept."""
        df = pd.DataFrame({
            "mixed": [None, "value", None],
            "id": [1, 2, 3]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        
        assert "mixed" in result.columns

    def test_removes_multiple_empty_columns(self):
        """Multiple empty columns should all be removed."""
        df = pd.DataFrame({
            "empty1": [None, None, None],
            "empty2": ["", "", ""],
            "id": [1, 2, 3],
            "empty3": ["   ", "   ", "   "]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        
        assert result.columns == ["id"]

    def test_returns_arframe_type(self):
        """Result should be an ArFrame instance."""
        df = pd.DataFrame({"id": [1, 2], "empty": [None, None]})
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        
        assert isinstance(result, ar.ArFrame)

    def test_preserves_all_columns_when_none_empty(self):
        """When all columns have values, all should be kept."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["a", "b", "c"],
            "value": [10.5, 20.3, 30.1]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        
        assert set(result.columns) == {"id", "name", "value"}

    def test_removes_all_columns_when_all_empty(self):
        """When all columns are empty, result should have no columns."""
        df = pd.DataFrame({
            "empty1": [None, None],
            "empty2": ["", ""],
            "empty3": ["   ", "   "]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        
        assert len(result.columns) == 0
        assert result.shape[1] == 0

    def test_handles_empty_dataframe(self):
        """Empty dataframe should be handled gracefully."""
        df = pd.DataFrame()
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        
        assert result.is_empty
        assert len(result.columns) == 0

    def test_single_empty_column_removed(self):
        """Single column that is empty should be removed."""
        df = pd.DataFrame({"empty": [None, None, None]})
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        
        assert len(result.columns) == 0

    def test_numeric_zeros_not_treated_as_empty(self):
        """Numeric columns with zero values should not be removed."""
        df = pd.DataFrame({
            "zeros": [0, 0, 0],
            "empty": [None, None, None],
            "id": [1, 2, 3]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        
        assert "zeros" in result.columns
        assert "empty" not in result.columns

    def test_boolean_false_not_treated_as_empty(self):
        """Boolean False values should not be treated as empty."""
        df = pd.DataFrame({
            "flags": [False, False, False],
            "empty": [None, None, None]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        
        assert "flags" in result.columns
        assert "empty" not in result.columns

    def test_preserves_column_order(self):
        """Column order should be preserved in result."""
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "empty": [None, None, None],
            "b": ["x", "y", "z"],
            "c": [1.1, 2.2, 3.3]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        
        assert result.columns == ["a", "b", "c"]

    def test_mixed_null_and_empty_string_treated_as_empty(self):
        """Column with only nulls and empty strings should be removed."""
        df = pd.DataFrame({
            "mixed_empty": [None, "", None, ""],
            "id": [1, 2, 3, 4]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        
        assert "mixed_empty" not in result.columns
        assert "id" in result.columns
