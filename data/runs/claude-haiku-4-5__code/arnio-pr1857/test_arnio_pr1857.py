import pytest
import pandas as pd
import numpy as np
import arnio as ar
from arnio.convert import from_dict


class TestFromDict:
    """Test suite for arnio.convert.from_dict function"""
    
    def test_basic_dict_to_arframe(self):
        """Test basic dictionary with same-length lists converts to ArFrame"""
        data = {'col1': [1, 2, 3], 'col2': [4, 5, 6]}
        result = from_dict(data)
        assert len(result) == 3
    
    def test_single_column_dict(self):
        """Test dictionary with a single column"""
        data = {'single': [10, 20, 30]}
        result = from_dict(data)
        assert len(result) == 3
    
    def test_multiple_data_types(self):
        """Test with columns of different data types"""
        data = {
            'ints': [1, 2, 3],
            'floats': [1.1, 2.2, 3.3],
            'strings': ['a', 'b', 'c']
        }
        result = from_dict(data)
        assert len(result) == 3
    
    def test_empty_dict(self):
        """Test empty dictionary"""
        result = from_dict({})
        assert len(result) == 0
    
    def test_empty_columns(self):
        """Test with empty lists as column values"""
        data = {'a': [], 'b': []}
        result = from_dict(data)
        assert len(result) == 0
    
    def test_numpy_array_values(self):
        """Test using numpy arrays as column values"""
        data = {
            'arr1': np.array([1, 2, 3]),
            'arr2': np.array([4.0, 5.0, 6.0])
        }
        result = from_dict(data)
        assert len(result) == 3
    
    def test_tuple_values(self):
        """Test using tuples as column values (alternative sequence type)"""
        data = {
            'col1': (1, 2, 3),
            'col2': (4, 5, 6)
        }
        result = from_dict(data)
        assert len(result) == 3
    
    def test_none_values_in_columns(self):
        """Test with None values in column data"""
        data = {'a': [1, None, 3], 'b': [None, 5, 6]}
        result = from_dict(data)
        assert len(result) == 3
    
    def test_type_error_not_dict_input(self):
        """Test TypeError when input is not a dictionary"""
        with pytest.raises(TypeError, match="Expected dict datatype"):
            from_dict([1, 2, 3])
        
        with pytest.raises(TypeError, match="Expected dict datatype"):
            from_dict((1, 2, 3))
    
    def test_type_error_non_string_keys(self):
        """Test TypeError when dictionary keys are not strings"""
        with pytest.raises(TypeError, match="All dictionary keys must be strings"):
            from_dict({1: [1, 2], 2: [3, 4]})
        
        with pytest.raises(TypeError, match="All dictionary keys must be strings"):
            from_dict({('a', 'b'): [1, 2]})
    
    def test_type_error_string_column_value(self):
        """Test TypeError when column value is a string (not a sequence container)"""
        with pytest.raises(TypeError, match="must be a sequence of values"):
            from_dict({'col': 'string_value'})
    
    def test_type_error_bytes_column_value(self):
        """Test TypeError when column value is bytes (not a sequence container)"""
        with pytest.raises(TypeError, match="must be a sequence of values"):
            from_dict({'col': b'bytes_value'})
    
    def test_type_error_non_sequence_column_value(self):
        """Test TypeError when column value lacks __len__ (e.g., integer)"""
        with pytest.raises(TypeError, match="must be a sequence of values"):
            from_dict({'col': 42})
    
    def test_value_error_nested_dict_column(self):
        """Test ValueError when a column value is a nested dictionary"""
        with pytest.raises(ValueError, match="Nested objects are not supported"):
            from_dict({'col': {'nested': 'dict'}})
    
    def test_value_error_column_length_mismatch(self):
        """Test ValueError when columns have different lengths"""
        with pytest.raises(ValueError, match="column lengths differ"):
            from_dict({'a': [1, 2, 3], 'b': [4, 5]})
