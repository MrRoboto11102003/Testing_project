import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestFromDict:
    """Test suite for arnio.from_dict function."""
    
    def test_basic_dict_conversion(self):
        """Test converting a simple dict to ArFrame."""
        data = {"name": ["Alice", "Bob"], "age": [25, 30]}
        frame = ar.from_dict(data)
        assert frame.shape == (2, 2)
        assert set(frame.columns) == {"name", "age"}
    
    def test_empty_dict(self):
        """Test converting an empty dict."""
        data = {}
        frame = ar.from_dict(data)
        assert frame.shape == (0, 0)
    
    def test_single_column_dict(self):
        """Test converting a dict with a single column."""
        data = {"id": [1, 2, 3]}
        frame = ar.from_dict(data)
        assert frame.shape == (3, 1)
        assert frame.columns == ["id"]
    
    def test_mixed_types_in_columns(self):
        """Test dict with various data types."""
        data = {
            "integers": [1, 2, 3],
            "floats": [1.5, 2.5, 3.5],
            "bools": [True, False, True],
            "strings": ["a", "b", "c"],
        }
        frame = ar.from_dict(data)
        assert frame.shape == (3, 4)
    
    def test_none_values_in_columns(self):
        """Test dict with None/null values."""
        data = {
            "col1": [1, None, 3],
            "col2": ["a", "b", None],
        }
        frame = ar.from_dict(data)
        assert frame.shape == (3, 2)
    
    def test_input_not_dict_raises_typeerror(self):
        """Test that non-dict input raises TypeError."""
        with pytest.raises(TypeError, match="Expected dict datatype"):
            ar.from_dict([("name", ["Alice", "Bob"])])
        
        with pytest.raises(TypeError, match="Expected dict datatype"):
            ar.from_dict("not a dict")
        
        with pytest.raises(TypeError, match="Expected dict datatype"):
            ar.from_dict(42)
    
    def test_non_string_keys_raises_typeerror(self):
        """Test that non-string keys raise TypeError."""
        with pytest.raises(TypeError, match="All dictionary keys must be strings"):
            ar.from_dict({1: ["a", "b"], "name": ["Alice", "Bob"]})
        
        with pytest.raises(TypeError, match="All dictionary keys must be strings"):
            ar.from_dict({("tuple", "key"): [1, 2]})
    
    def test_nested_dict_in_column_raises_valueerror(self):
        """Test that nested dicts in column values raise ValueError."""
        with pytest.raises(ValueError, match="Nested objects are not supported"):
            ar.from_dict({"data": [{"nested": "value"}, {"nested": "value2"}]})
    
    def test_string_column_value_raises_typeerror(self):
        """Test that string column values (not list of strings) raise TypeError."""
        with pytest.raises(TypeError, match="must be a sequence of values"):
            ar.from_dict({"name": "AliceBob"})
    
    def test_bytes_column_value_raises_typeerror(self):
        """Test that bytes column values raise TypeError."""
        with pytest.raises(TypeError, match="must be a sequence of values"):
            ar.from_dict({"data": b"bytes"})
    
    def test_non_sequence_column_value_raises_typeerror(self):
        """Test that non-sequence column values raise TypeError."""
        with pytest.raises(TypeError, match="must be a sequence of values"):
            ar.from_dict({"num": 42})
    
    def test_mismatched_column_lengths_raises_valueerror(self):
        """Test that columns with different lengths raise ValueError."""
        with pytest.raises(ValueError, match="column lengths differ"):
            ar.from_dict({
                "col1": [1, 2, 3],
                "col2": ["a", "b"],
            })
    
    def test_result_converts_to_pandas(self):
        """Test that result can be converted back to pandas DataFrame."""
        data = {"name": ["Alice", "Bob"], "age": [25, 30]}
        frame = ar.from_dict(data)
        df = ar.to_pandas(frame)
        assert isinstance(df, pd.DataFrame)
        assert list(df["name"]) == ["Alice", "Bob"]
        assert list(df["age"]) == [25, 30]
    
    def test_column_order_preserved(self):
        """Test that column order is preserved from dict."""
        data = {"z": [1, 2], "a": [3, 4], "m": [5, 6]}
        frame = ar.from_dict(data)
        assert frame.columns == ["z", "a", "m"]
