import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestDropConstantColumns:
    """Tests for arnio.cleaning.drop_constant_columns"""

    def test_basic_remove_constant_column(self):
        """Should remove columns with exactly one unique value."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [5, 5, 5]})
        frame = ar.from_pandas(df)
        result = ar.drop_constant_columns(frame)
        assert result.columns == ["a"]
        assert ar.to_pandas(result)["a"].tolist() == [1, 2, 3]

    def test_null_handling(self):
        """Null-only columns are constant; partly null columns are not."""
        df = pd.DataFrame({
            "null_only": [None, None],
            "partly_null": [1, None],
            "no_null": [2, 3]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_constant_columns(frame)
        # null_only is dropped; partly_null and no_null kept (2 unique values each)
        assert set(result.columns) == {"partly_null", "no_null"}

    def test_zero_row_keeps_all_columns(self):
        """Zero-row frames keep all columns (0 unique values, not 1)."""
        df = pd.DataFrame({
            "a": pd.Series([], dtype="int64"),
            "b": pd.Series([], dtype="float64")
        })
        frame = ar.from_pandas(df)
        result = ar.drop_constant_columns(frame)
        assert result.columns == ["a", "b"]
        assert result.shape[0] == 0

    def test_single_row_drops_all(self):
        """Single row: all columns have exactly 1 unique value and are dropped."""
        df = pd.DataFrame({"a": [10], "b": ["x"]})
        frame = ar.from_pandas(df)
        result = ar.drop_constant_columns(frame)
        assert result.columns == []
        assert result.shape[0] == 1

    def test_column_order_preserved(self):
        """Non-constant columns maintain original order."""
        df = pd.DataFrame({"x": [1, 2], "const": [0, 0], "y": [3, 4]})
        frame = ar.from_pandas(df)
        result = ar.drop_constant_columns(frame)
        assert result.columns == ["x", "y"]

    def test_pandas_dataframe_input_and_output(self):
        """Accepts and returns pandas DataFrame unchanged in type."""
        df = pd.DataFrame({"a": [1, 2], "b": [5, 5]})
        result = ar.drop_constant_columns(df)
        assert isinstance(result, pd.DataFrame)
        assert result.columns.tolist() == ["a"]

    def test_type_error_on_invalid_input(self):
        """Raises TypeError for non-frame inputs."""
        with pytest.raises(TypeError):
            ar.drop_constant_columns({"a": [1]})
        with pytest.raises(TypeError):
            ar.drop_constant_columns(None)
        with pytest.raises(TypeError):
            ar.drop_constant_columns([1, 2])


class TestDropEmptyColumns:
    """Tests for arnio.cleaning.drop_empty_columns"""

    def test_basic_remove_null_column(self):
        """Should remove columns that are entirely null."""
        df = pd.DataFrame({"a": [1, 2], "b": [None, None]})
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        assert result.columns == ["a"]

    def test_whitespace_only_is_empty(self):
        """Columns with only whitespace strings are considered empty."""
        df = pd.DataFrame({"a": [1, 2], "b": ["", " ", "\t"]})
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        assert result.columns == ["a"]

    def test_zero_row_keeps_all_columns(self):
        """Zero-row frames keep all columns even if conceptually empty."""
        df = pd.DataFrame({
            "a": pd.Series([], dtype="int64"),
            "b": pd.Series([], dtype="float64")
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        assert result.columns == ["a", "b"]
        assert result.shape[0] == 0

    def test_non_empty_values_preserved(self):
        """Zero, False, and mixed-null columns are not empty."""
        df = pd.DataFrame({
            "zeros": [0, 0],
            "bools": [False, False],
            "mixed": [None, "text"]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        assert set(result.columns) == {"zeros", "bools", "mixed"}

    def test_column_order_preserved(self):
        """Remaining columns maintain original order."""
        df = pd.DataFrame({"z": [1], "empty": [None], "a": ["x"]})
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        assert result.columns == ["z", "a"]

    def test_dtype_preservation(self):
        """Remaining columns preserve original dtypes."""
        df = pd.DataFrame({
            "int_col": pd.array([1, 2], dtype="int64"),
            "empty": [None, None],
            "float_col": pd.array([1.5, 2.5], dtype="float64")
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        assert "int_col" in result.columns
        assert "float_col" in result.columns
        # Verify dtypes match
        assert "int64" in str(result.dtypes.get("int_col", ""))
        assert "float" in str(result.dtypes.get("float_col", ""))

    def test_attrs_deep_copy(self):
        """Result _attrs are deep copy, not shared reference."""
        df = pd.DataFrame({"a": [1], "b": [None]})
        frame = ar.from_pandas(df)
        frame._attrs = {"key": [1]}
        result = ar.drop_empty_columns(frame)
        
        # Should be equal
        assert result._attrs == frame._attrs
        # But not the same object
        assert result._attrs is not frame._attrs
        # Modifying result should not affect frame
        result._attrs["key"].append(2)
        assert frame._attrs["key"] == [1]

    def test_type_error_on_invalid_input(self):
        """Raises TypeError for non-ArFrame inputs."""
        with pytest.raises(TypeError):
            ar.drop_empty_columns(pd.DataFrame({"a": [1]}))
        with pytest.raises(TypeError):
            ar.drop_empty_columns({"a": [1]})
        with pytest.raises(TypeError):
            ar.drop_empty_columns(None)
