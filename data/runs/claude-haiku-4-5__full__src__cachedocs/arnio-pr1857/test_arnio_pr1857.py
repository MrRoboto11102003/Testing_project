import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestFromDict:
    """Tests for arnio.convert.from_dict"""

    def test_basic_dict_with_lists(self):
        """Test basic dictionary with equal-length lists"""
        data = {"name": ["Alice", "Bob"], "age": [25, 30]}
        frame = ar.from_dict(data)
        assert frame.shape == (2, 2)
        assert set(frame.columns) == {"name", "age"}

    def test_single_column_dict(self):
        """Test dictionary with a single column"""
        data = {"values": [1, 2, 3]}
        frame = ar.from_dict(data)
        assert frame.shape == (3, 1)

    def test_empty_dict_creates_empty_frame(self):
        """Test empty dictionary creates empty ArFrame"""
        data = {}
        frame = ar.from_dict(data)
        assert frame.is_empty

    def test_dict_with_integer_column(self):
        """Test dict with integer data"""
        data = {"ids": [1, 2, 3, 4, 5]}
        frame = ar.from_dict(data)
        result = ar.to_pandas(frame)
        assert result["ids"].tolist() == [1, 2, 3, 4, 5]

    def test_dict_with_float_column(self):
        """Test dict with float data"""
        data = {"scores": [1.1, 2.2, 3.3]}
        frame = ar.from_dict(data)
        result = ar.to_pandas(frame)
        np.testing.assert_array_almost_equal(result["scores"].tolist(), [1.1, 2.2, 3.3])

    def test_dict_with_boolean_column(self):
        """Test dict with boolean data"""
        data = {"flags": [True, False, True]}
        frame = ar.from_dict(data)
        result = ar.to_pandas(frame)
        assert result["flags"].tolist() == [True, False, True]

    def test_dict_with_null_values(self):
        """Test dict with None values in column"""
        data = {"values": [1, None, 3]}
        frame = ar.from_dict(data)
        result = ar.to_pandas(frame)
        assert pd.isna(result["values"].iloc[1])
        assert result["values"].iloc[0] == 1

    def test_non_dict_input_raises_type_error(self):
        """Test that non-dict input raises TypeError"""
        with pytest.raises(TypeError, match="Expected dict datatype"):
            ar.from_dict([1, 2, 3])
        with pytest.raises(TypeError, match="Expected dict datatype"):
            ar.from_dict("not a dict")

    def test_non_string_keys_raise_type_error(self):
        """Test that non-string keys raise TypeError"""
        with pytest.raises(TypeError, match="All dictionary keys must be strings"):
            ar.from_dict({1: [1, 2], 2: [3, 4]})

    def test_string_column_value_raises_type_error(self):
        """Test that string column value raises TypeError"""
        with pytest.raises(TypeError, match="must be a sequence of values"):
            ar.from_dict({"col": "not a list"})

    def test_bytes_column_value_raises_type_error(self):
        """Test that bytes column value raises TypeError"""
        with pytest.raises(TypeError, match="must be a sequence of values"):
            ar.from_dict({"col": b"bytes"})

    def test_nested_dict_raises_value_error(self):
        """Test that nested dicts in values raise ValueError"""
        with pytest.raises(ValueError, match="Nested objects are not supported"):
            ar.from_dict({"col": [{"key": "value"}]})

    def test_mismatched_column_lengths_raise_value_error(self):
        """Test that columns with different lengths raise ValueError"""
        with pytest.raises(ValueError, match="column lengths differ"):
            ar.from_dict({"col1": [1, 2], "col2": [1, 2, 3]})

    def test_tuple_as_column_value(self):
        """Test that tuples work as column values"""
        data = {"col": (1, 2, 3)}
        frame = ar.from_dict(data)
        assert frame.shape == (3, 1)

    def test_numpy_array_as_column_value(self):
        """Test that numpy arrays work as column values"""
        data = {"values": np.array([10, 20, 30])}
        frame = ar.from_dict(data)
        result = ar.to_pandas(frame)
        assert result["values"].tolist() == [10, 20, 30]
