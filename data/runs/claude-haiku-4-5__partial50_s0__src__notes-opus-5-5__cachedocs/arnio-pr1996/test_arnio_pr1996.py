import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestDropEmptyColumns:
    """Tests for arnio.cleaning.drop_empty_columns"""

    def test_basic_mixed_columns(self):
        """Test with mix of empty and non-empty columns."""
        df = pd.DataFrame({
            "a": [1, 2],
            "b": [None, None],
            "c": ["x", "y"]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        
        assert result.columns == ["a", "c"]
        assert result.shape == (2, 2)

    def test_no_empty_columns(self):
        """When no columns are empty, all are kept as new object."""
        df = pd.DataFrame({
            "x": [1, 2, 3],
            "y": ["a", "b", "c"]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        
        assert result.columns == ["x", "y"]
        assert result is not frame

    def test_all_columns_empty(self):
        """When all columns are empty, result has no columns."""
        df = pd.DataFrame({
            "a": [None, None],
            "b": ["", " "]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        
        assert result.columns == []
        assert result.shape[1] == 0

    def test_zero_rows_keeps_all(self):
        """Zero-row frame: all columns kept regardless of dtype."""
        df = pd.DataFrame({
            "a": pd.Series([], dtype="int64"),
            "b": pd.Series([], dtype="string")
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        
        assert result.columns == ["a", "b"]
        assert result.shape[0] == 0

    def test_whitespace_only_dropped(self):
        """Columns with only whitespace strings are dropped."""
        df = pd.DataFrame({
            "a": [" ", "\t", "\n"],
            "b": [1, 2, 3]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        
        assert result.columns == ["b"]

    def test_mixed_null_whitespace_dropped(self):
        """Column with mix of nulls and whitespace is dropped."""
        df = pd.DataFrame({
            "a": [None, " ", None],
            "b": ["x", "y", "z"]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        
        assert result.columns == ["b"]

    def test_keep_zero_and_false(self):
        """Columns with 0 or False values are not empty."""
        df = pd.DataFrame({
            "zeros": [0, 0],
            "bools": [False, False],
            "empty": [None, None]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        
        assert result.columns == ["zeros", "bools"]

    def test_column_order_preserved(self):
        """Original column order is preserved when removing empties."""
        df = pd.DataFrame({
            "z": [1],
            "e": [None],
            "a": ["k"],
            "m": [" "]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        
        assert result.columns == ["z", "a"]

    def test_dtype_preservation(self):
        """Kept columns preserve their original dtypes."""
        df = pd.DataFrame({
            "int_col": [1, 2],
            "empty": [None, None],
            "float_col": [1.5, 2.5]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        
        assert result.dtypes["int_col"] == frame.dtypes["int_col"]
        assert result.dtypes["float_col"] == frame.dtypes["float_col"]

    def test_whitespace_preserved_in_kept_columns(self):
        """Whitespace in non-empty columns is not stripped."""
        df = pd.DataFrame({
            "keep": ["  hello  ", "world"],
            "drop": ["", " "]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        
        assert result.columns == ["keep"]
        result_df = ar.to_pandas(result)
        assert len(result_df["keep"].tolist()) == 2

    def test_attrs_deep_copied(self):
        """Frame attrs are deep copied, not referenced."""
        df = pd.DataFrame({"a": [1], "b": [None]})
        frame = ar.from_pandas(df)
        frame._attrs = {"src": {"x": [1, 2]}}
        
        result = ar.drop_empty_columns(frame)
        
        assert result._attrs == frame._attrs
        assert result._attrs is not frame._attrs
        result._attrs["src"]["x"].append(3)
        assert len(frame._attrs["src"]["x"]) == 2

    def test_non_arframe_raises(self):
        """Non-ArFrame input raises TypeError."""
        df = pd.DataFrame({"a": [1]})
        
        with pytest.raises(TypeError):
            ar.drop_empty_columns(df)

    def test_single_value_column_kept(self):
        """Column with one non-null value among nulls is kept."""
        df = pd.DataFrame({
            "has_value": [None, "x", None],
            "all_null": [None, None, None]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        
        assert result.columns == ["has_value"]

    def test_empty_string_and_whitespace_equivalent(self):
        """Both empty strings and whitespace are treated as empty."""
        df = pd.DataFrame({
            "empties": ["", " ", "\t"],
            "values": ["a", "b", "c"]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        
        assert result.columns == ["values"]

    def test_values_unchanged(self):
        """Non-empty column values are unchanged."""
        df = pd.DataFrame({
            "keep": [1, 2, 3],
            "drop": [None, None, None]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        
        result_df = ar.to_pandas(result)
        assert result_df["keep"].tolist() == [1, 2, 3]
