import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestDropConstantColumns:
    """Tests for arnio.cleaning.drop_constant_columns"""

    def test_basic_constant_column_removal(self):
        """Remove a column with a single unique value."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "constant": ["A", "A", "A"],
            "varied": ["X", "Y", "Z"],
        })
        frame = ar.from_pandas(df)
        result = ar.drop_constant_columns(frame)
        result_df = ar.to_pandas(result)
        
        assert list(result_df.columns) == ["id", "varied"]
        assert result_df.shape[0] == 3

    def test_constant_columns_different_types(self):
        """Remove constant columns with int, str, float, and bool types."""
        df = pd.DataFrame({
            "const_int": [5, 5, 5],
            "const_str": ["A", "A", "A"],
            "const_float": [1.5, 1.5, 1.5],
            "const_bool": [True, True, True],
            "varied": [1, 2, 3],
        })
        frame = ar.from_pandas(df)
        result = ar.drop_constant_columns(frame)
        result_df = ar.to_pandas(result)
        
        assert list(result_df.columns) == ["varied"]

    def test_all_null_column_is_constant(self):
        """All-null columns have one unique value (None) and are dropped."""
        df = pd.DataFrame({
            "all_null": [None, None, None],
            "varied": [1, 2, 3],
        })
        frame = ar.from_pandas(df)
        result = ar.drop_constant_columns(frame)
        result_df = ar.to_pandas(result)
        
        assert list(result_df.columns) == ["varied"]

    def test_null_with_value_is_not_constant(self):
        """Columns with null + value have two unique values and are kept."""
        df = pd.DataFrame({
            "null_and_value": [1, 1, None],
            "constant": ["A", "A", "A"],
        })
        frame = ar.from_pandas(df)
        result = ar.drop_constant_columns(frame)
        result_df = ar.to_pandas(result)
        
        assert list(result_df.columns) == ["null_and_value"]

    def test_zero_row_frame_keeps_columns(self):
        """Zero-row columns have zero unique values (not one), so are kept."""
        df = pd.DataFrame({
            "col1": pd.Series([], dtype=object),
            "col2": pd.Series([], dtype=int),
        })
        frame = ar.from_pandas(df)
        result = ar.drop_constant_columns(frame)
        result_df = ar.to_pandas(result)
        
        assert set(result_df.columns) == {"col1", "col2"}
        assert result_df.shape[0] == 0

    def test_all_columns_constant_preserves_rows(self):
        """Dropping all constant columns preserves row count."""
        df = pd.DataFrame({
            "const1": ["A", "A", "A"],
            "const2": [1, 1, 1],
            "const3": [True, True, True],
        })
        frame = ar.from_pandas(df)
        result = ar.drop_constant_columns(frame)
        result_df = ar.to_pandas(result)
        
        assert result_df.shape[0] == 3  # rows preserved
        assert result_df.shape[1] == 0  # all columns dropped

    def test_no_constant_columns(self):
        """When no columns are constant, all columns are returned."""
        df = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": ["A", "B", "C"],
            "col3": [1.1, 2.2, 3.3],
        })
        frame = ar.from_pandas(df)
        result = ar.drop_constant_columns(frame)
        result_df = ar.to_pandas(result)
        
        assert set(result_df.columns) == {"col1", "col2", "col3"}
        assert result_df.shape[0] == 3

    def test_preserves_input_type_pandas(self):
        """Function returns pd.DataFrame when given pd.DataFrame."""
        df = pd.DataFrame({
            "constant": ["X", "X"],
            "varied": [1, 2],
        })
        result = ar.drop_constant_columns(df)
        
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["varied"]

    def test_preserves_input_type_arframe(self):
        """Function returns ArFrame when given ArFrame."""
        df = pd.DataFrame({
            "constant": ["X", "X"],
            "varied": [1, 2],
        })
        frame = ar.from_pandas(df)
        result = ar.drop_constant_columns(frame)
        
        assert isinstance(result, ar.ArFrame)
        result_df = ar.to_pandas(result)
        assert list(result_df.columns) == ["varied"]

    def test_mixed_constant_and_varied(self):
        """Realistic scenario with mix of constant and varied columns."""
        df = pd.DataFrame({
            "id": [1, 2, 3, 4],
            "region": ["US", "US", "US", "US"],  # constant
            "sales": [100, 200, 150, 300],  # varied
            "source": ["web", "web", "web", "web"],  # constant
            "price": [10.0, 20.0, 15.0, 30.0],  # varied
        })
        frame = ar.from_pandas(df)
        result = ar.drop_constant_columns(frame)
        result_df = ar.to_pandas(result)
        
        assert set(result_df.columns) == {"id", "sales", "price"}


class TestDropEmptyColumns:
    """Tests for arnio.cleaning.drop_empty_columns"""

    def test_remove_all_null_columns(self):
        """Remove columns that are entirely null."""
        df = pd.DataFrame({
            "all_null": [None, None, None],
            "has_value": [1, 2, 3],
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        result_df = ar.to_pandas(result)
        
        assert list(result_df.columns) == ["has_value"]
        assert result_df.shape[0] == 3

    def test_remove_all_empty_string_columns(self):
        """Remove columns that are entirely empty strings."""
        df = pd.DataFrame({
            "all_empty": ["", "", ""],
            "has_value": ["A", "B", "C"],
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        result_df = ar.to_pandas(result)
        
        assert list(result_df.columns) == ["has_value"]

    def test_remove_all_whitespace_columns(self):
        """Remove columns with only whitespace strings."""
        df = pd.DataFrame({
            "whitespace": ["   ", "\t", "\n"],
            "has_value": ["A", "B", "C"],
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        result_df = ar.to_pandas(result)
        
        assert list(result_df.columns) == ["has_value"]

    def test_keep_column_with_one_non_empty_value(self):
        """Keep columns that have at least one non-empty value."""
        df = pd.DataFrame({
            "mostly_empty": ["", "", "X"],
            "has_values": [1, 2, 3],
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        result_df = ar.to_pandas(result)
        
        assert set(result_df.columns) == {"mostly_empty", "has_values"}

    def test_mixed_null_and_empty_is_empty(self):
        """Columns with only nulls and empty strings are removed."""
        df = pd.DataFrame({
            "mixed_empty": [None, "", None, ""],
            "has_value": [1, 2, 3, 4],
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        result_df = ar.to_pandas(result)
        
        assert list(result_df.columns) == ["has_value"]

    def test_no_empty_columns(self):
        """When no columns are empty, all columns are returned."""
        df = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": ["A", "B", "C"],
            "col3": [1.1, 2.2, 3.3],
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        result_df = ar.to_pandas(result)
        
        assert set(result_df.columns) == {"col1", "col2", "col3"}

    def test_all_columns_empty(self):
        """When all columns are empty, result has zero columns."""
        df = pd.DataFrame({
            "empty1": [None, None, None],
            "empty2": ["", "", ""],
            "empty3": ["  ", "\t", "\n"],
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        result_df = ar.to_pandas(result)
        
        assert result_df.shape[1] == 0
        assert result_df.shape[0] == 3

    def test_zero_row_frame_with_empty_columns(self):
        """Zero-row frame with all-null columns are removed."""
        df = pd.DataFrame({
            "empty": pd.Series([], dtype=object),
            "also_empty": pd.Series([], dtype=str),
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        result_df = ar.to_pandas(result)
        
        assert result_df.shape == (0, 0)

    def test_numeric_columns_not_affected(self):
        """Numeric columns are not removed (no empty/whitespace concept)."""
        df = pd.DataFrame({
            "numbers": [0, 0, 0],
            "empty_str": ["", "", ""],
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        result_df = ar.to_pandas(result)
        
        assert list(result_df.columns) == ["numbers"]
