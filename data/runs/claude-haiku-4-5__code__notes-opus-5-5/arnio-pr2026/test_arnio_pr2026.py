import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestDropConstantColumns:
    """Tests for arnio.cleaning.drop_constant_columns."""
    
    def test_basic_mixed_constant_and_varying_columns(self):
        """Test basic case: some columns constant, some varying."""
        df = pd.DataFrame({
            "const_int": [1, 1, 1],
            "varying_int": [1, 2, 3],
            "const_str": ["x", "x", "x"],
            "varying_str": ["a", "b", "c"]
        })
        result = ar.drop_constant_columns(df)
        assert result.columns.tolist() == ["varying_int", "varying_str"]
        assert len(result) == 3
    
    def test_all_columns_constant_preserves_row_count(self):
        """Test that all constant columns dropped preserves row count."""
        df = pd.DataFrame({"a": [5, 5], "b": ["z", "z"]})
        result = ar.drop_constant_columns(df)
        assert result.columns.tolist() == []
        assert len(result) == 2
    
    def test_single_row_all_columns_dropped(self):
        """Test that single row means all columns are dropped (nunique==1)."""
        df = pd.DataFrame({"a": [1], "b": ["x"]})
        result = ar.drop_constant_columns(df)
        assert result.columns.tolist() == []
        assert len(result) == 1
    
    def test_zero_rows_keeps_all_columns(self):
        """Test zero-row frame keeps all columns."""
        df = pd.DataFrame({
            "a": pd.Series([], dtype="int64"),
            "b": pd.Series([], dtype="str")
        })
        result = ar.drop_constant_columns(df)
        assert result.columns.tolist() == ["a", "b"]
        assert len(result) == 0
    
    def test_null_handling_all_null_dropped_partial_null_kept(self):
        """Test: all-null column dropped, partial-null column kept."""
        df = pd.DataFrame({
            "all_null": [None, None],
            "partial_null": [1, None],
            "no_null": [1, 2]
        })
        result = ar.drop_constant_columns(df)
        assert result.columns.tolist() == ["partial_null", "no_null"]
    
    def test_pandas_input_returns_pandas_dataframe(self):
        """Test return type: pandas DataFrame input → pandas DataFrame."""
        df = pd.DataFrame({"a": [1, 1], "b": [1, 2]})
        result = ar.drop_constant_columns(df)
        assert isinstance(result, pd.DataFrame)
    
    def test_arframe_input_returns_arframe(self):
        """Test return type: ArFrame input → ArFrame."""
        df = pd.DataFrame({"a": [1, 1], "b": [1, 2]})
        frame = ar.from_pandas(df)
        result = ar.drop_constant_columns(frame)
        assert isinstance(result, ar.ArFrame)
    
    def test_column_order_preserved(self):
        """Test that surviving columns maintain original order."""
        df = pd.DataFrame({
            "x": [1, 2],
            "const": [0, 0],
            "y": [3, 4]
        })
        result = ar.drop_constant_columns(df)
        assert result.columns.tolist() == ["x", "y"]


class TestDropEmptyColumns:
    """Tests for arnio.cleaning.drop_empty_columns."""
    
    def test_basic_mixed_empty_and_nonempty_columns(self):
        """Test basic case: some empty, some non-empty columns."""
        df = pd.DataFrame({
            "data": [1, 2],
            "nulls": [None, None],
            "text": ["a", "b"],
            "blanks": ["", " "]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        assert result.columns == ["data", "text"]
        assert result.shape[0] == 2
    
    def test_all_columns_empty(self):
        """Test when all columns are empty."""
        df = pd.DataFrame({
            "nulls": [None, None],
            "blanks": ["  ", ""]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        assert result.columns == []
        assert result.shape[0] == 2
    
    def test_zero_rows_keeps_all_columns_despite_emptiness(self):
        """Test that zero-row frame keeps all columns even if empty."""
        df = pd.DataFrame({
            "a": pd.Series([], dtype="int64"),
            "b": pd.Series([], dtype="str")
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        assert result.columns == ["a", "b"]
        assert result.shape[0] == 0
    
    def test_null_column_is_empty(self):
        """Test that all-null column is dropped as empty."""
        df = pd.DataFrame({
            "nulls": [None, None],
            "data": [1, 2]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        assert result.columns == ["data"]
    
    def test_whitespace_only_column_is_empty(self):
        """Test that whitespace-only strings are considered empty."""
        df = pd.DataFrame({
            "blanks": ["", " ", "\t"],
            "data": ["a", "b", "c"]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        assert result.columns == ["data"]
    
    def test_column_order_and_dtypes_preserved(self):
        """Test that column order and dtypes are preserved."""
        df = pd.DataFrame({
            "z": pd.Series([1, 2], dtype="int64"),
            "empty": pd.Series([None, None], dtype="float64"),
            "a": pd.Series([3.5, 4.5], dtype="float64")
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        assert result.columns == ["z", "a"]
        result_df = ar.to_pandas(result)
        assert result_df["z"].dtype == np.dtype("int64")
        assert result_df["a"].dtype == np.dtype("float64")
    
    def test_attrs_deep_copied(self):
        """Test that attrs are deep copied, not shared."""
        df = pd.DataFrame({"a": [1, 2], "b": [None, None]})
        frame = ar.from_pandas(df)
        frame._attrs = {"key": [1, 2]}
        
        result = ar.drop_empty_columns(frame)
        
        # Same value, different object
        assert result._attrs == {"key": [1, 2]}
        assert result._attrs is not frame._attrs
        
        # Mutation of result doesn't affect original
        result._attrs["key"].append(3)
        assert frame._attrs["key"] == [1, 2]
