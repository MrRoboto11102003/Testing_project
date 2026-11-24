import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestDropEmptyColumns:
    
    def test_basic_drop_empty_columns(self):
        """Test basic functionality of removing empty columns."""
        df = pd.DataFrame({
            "a": [1, 2],
            "b": [None, None],
            "c": ["x", "y"]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        
        assert list(result.columns) == ["a", "c"]
        assert result.shape == (2, 2)
    
    def test_whitespace_only_strings_are_empty(self):
        """Test that whitespace-only strings are treated as empty."""
        df = pd.DataFrame({
            "a": ["x", "y"],
            "whitespace": [" ", "\t"]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        
        assert list(result.columns) == ["a"]
    
    def test_zero_rows_keeps_all_columns(self):
        """Test that zero-row frames keep all columns."""
        df = pd.DataFrame({
            "a": pd.Series([], dtype="int64"),
            "b": pd.Series([], dtype="float64")
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        
        assert list(result.columns) == ["a", "b"]
        assert result.shape[0] == 0
    
    def test_zero_rows_dtype_preservation(self):
        """Test that dtypes are preserved in zero-row frames."""
        df = pd.DataFrame({
            "int_col": pd.Series([], dtype="int64"),
            "float_col": pd.Series([], dtype="float64")
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        
        assert result.dtypes["int_col"] == frame.dtypes["int_col"]
        assert result.dtypes["float_col"] == frame.dtypes["float_col"]
    
    def test_non_arframe_input_raises_typeerror(self):
        """Test that non-ArFrame input raises TypeError."""
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(TypeError):
            ar.drop_empty_columns(df)
    
    def test_returns_new_arframe_object(self):
        """Test that a new ArFrame is returned, not the input."""
        df = pd.DataFrame({"a": [1, 2]})
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        
        assert result is not frame
    
    def test_column_order_preserved(self):
        """Test that column order is preserved after dropping empty columns."""
        df = pd.DataFrame({
            "z": [1, 2],
            "e": [None, None],
            "a": ["x", "y"]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        
        assert list(result.columns) == ["z", "a"]
    
    def test_dtype_preservation(self):
        """Test that dtypes are preserved for kept columns."""
        df = pd.DataFrame({
            "int_col": [1, 2],
            "empty_col": [None, None],
            "str_col": ["x", "y"]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        
        assert result.dtypes["int_col"] == frame.dtypes["int_col"]
        assert result.dtypes["str_col"] == frame.dtypes["str_col"]
    
    def test_nulls_in_kept_columns_preserved(self):
        """Test that null values in kept columns are preserved."""
        df = pd.DataFrame({
            "a": [1, None, 3],
            "b": [None, None, None]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        
        result_pd = ar.to_pandas(result)
        assert pd.isna(result_pd["a"].iloc[1])
        assert result.shape == (3, 1)
    
    def test_attrs_deep_copied(self):
        """Test that attrs are deep copied."""
        df = pd.DataFrame({"a": [1, 2]})
        frame = ar.from_pandas(df)
        frame._attrs = {"src": {"x": [1]}}
        
        result = ar.drop_empty_columns(frame)
        
        assert result._attrs == frame._attrs
        assert result._attrs is not frame._attrs
        
        result._attrs["src"]["x"].append(2)
        assert frame._attrs["src"]["x"] == [1]
    
    def test_attrs_none_preserved(self):
        """Test that None attrs stay None."""
        df = pd.DataFrame({"a": [1, 2]})
        frame = ar.from_pandas(df)
        frame._attrs = None
        
        result = ar.drop_empty_columns(frame)
        
        assert result._attrs is None
    
    def test_all_columns_empty(self):
        """Test when all columns are empty."""
        df = pd.DataFrame({
            "a": [None, None],
            "b": [" ", "\t"]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        
        assert result.shape[1] == 0
    
    def test_no_empty_columns(self):
        """Test when no columns are empty."""
        df = pd.DataFrame({
            "a": [1, 2],
            "b": ["x", "y"]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        
        assert list(result.columns) == ["a", "b"]
        assert result.shape == (2, 2)
    
    def test_numeric_zeros_and_booleans_not_empty(self):
        """Test that numeric zeros and False are not treated as empty."""
        df = pd.DataFrame({
            "zeros": [0, 0],
            "bools": [False, False],
            "empty": [None, None]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        
        assert list(result.columns) == ["zeros", "bools"]
    
    def test_mixed_null_and_whitespace_in_same_column(self):
        """Test column with mixed nulls and whitespace is dropped."""
        df = pd.DataFrame({
            "a": ["x", "y"],
            "mixed": [None, " ", "\t"]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_empty_columns(frame)
        
        assert list(result.columns) == ["a"]
