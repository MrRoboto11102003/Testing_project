import pytest
import pandas as pd
import numpy as np
import json
from pathlib import Path

try:
    import arnio as ar
except ImportError:
    pytest.skip("arnio not installed", allow_module_level=True)


class TestFromDictBasicConversions:
    """Test basic successful conversions with from_dict."""
    
    def test_simple_numeric_columns(self):
        """Convert dict with numeric columns to ArFrame."""
        data = {
            'id': [1, 2, 3],
            'value': [10.5, 20.5, 30.5]
        }
        result = ar.convert.from_dict(data)
        assert result is not None
        assert len(result) == 3
    
    def test_mixed_string_numeric_columns(self):
        """Convert dict with mixed string and numeric columns."""
        data = {
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35],
            'salary': [50000.0, 60000.0, 75000.0]
        }
        result = ar.convert.from_dict(data)
        assert len(result) == 3
    
    def test_numpy_array_columns(self):
        """Convert dict with numpy arrays as values."""
        data = {
            'col1': np.array([1, 2, 3]),
            'col2': np.array([4.0, 5.0, 6.0])
        }
        result = ar.convert.from_dict(data)
        assert len(result) == 3
    
    def test_tuple_columns(self):
        """Convert dict with tuples as column values."""
        data = {
            'x': (1, 2, 3, 4),
            'y': (5.0, 6.0, 7.0, 8.0)
        }
        result = ar.convert.from_dict(data)
        assert len(result) == 4
    
    def test_with_null_values(self):
        """Dict with None values in columns."""
        data = {
            'col1': [1, None, 3],
            'col2': ['a', None, 'c']
        }
        result = ar.convert.from_dict(data)
        assert len(result) == 3


class TestFromDictInvalidInputs:
    """Test error handling for invalid inputs."""
    
    def test_non_dict_input_raises_typeerror(self):
        """Non-dict input raises TypeError with descriptive message."""
        with pytest.raises(TypeError) as exc_info:
            ar.convert.from_dict([1, 2, 3])
        assert "Expected dict datatype" in str(exc_info.value)
        assert "list" in str(exc_info.value).lower()
    
    def test_non_string_keys_raise_typeerror(self):
        """Non-string dictionary keys raise TypeError."""
        with pytest.raises(TypeError) as exc_info:
            ar.convert.from_dict({1: [1, 2, 3]})
        assert "All dictionary keys must be strings" in str(exc_info.value)
    
    def test_nested_dict_values_raise_valueerror(self):
        """Nested dictionary values raise ValueError."""
        data = {
            'col1': [1, 2, 3],
            'col2': {'nested': 'dict'}
        }
        with pytest.raises(ValueError) as exc_info:
            ar.convert.from_dict(data)
        assert "Nested objects are not supported" in str(exc_info.value)
        assert "col2" in str(exc_info.value)
    
    def test_string_value_raises_typeerror(self):
        """String column value (not list of strings) raises TypeError."""
        data = {
            'col1': [1, 2, 3],
            'col2': 'single_string'
        }
        with pytest.raises(TypeError) as exc_info:
            ar.convert.from_dict(data)
        assert "must be a sequence of values" in str(exc_info.value)
        assert "col2" in str(exc_info.value)
    
    def test_bytes_value_raises_typeerror(self):
        """Bytes column value raises TypeError."""
        data = {
            'col1': [1, 2, 3],
            'col2': b'bytes'
        }
        with pytest.raises(TypeError) as exc_info:
            ar.convert.from_dict(data)
        assert "must be a sequence of values" in str(exc_info.value)
        assert "bytes" in str(exc_info.value)
    
    def test_scalar_value_raises_typeerror(self):
        """Scalar numeric value raises TypeError."""
        data = {
            'col1': [1, 2, 3],
            'col2': 42
        }
        with pytest.raises(TypeError) as exc_info:
            ar.convert.from_dict(data)
        assert "must be a sequence of values" in str(exc_info.value)
    
    def test_mismatched_column_lengths_raise_valueerror(self):
        """Columns with different lengths raise ValueError with details."""
        data = {
            'col1': [1, 2, 3],
            'col2': [4, 5]
        }
        with pytest.raises(ValueError) as exc_info:
            ar.convert.from_dict(data)
        error_msg = str(exc_info.value)
        assert "column lengths differ" in error_msg
        assert "col1" in error_msg
        assert "col2" in error_msg
        assert "3" in error_msg
        assert "2" in error_msg


class TestFromDictEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_dict(self):
        """Empty dict converts successfully to empty ArFrame."""
        result = ar.convert.from_dict({})
        assert result is not None
        assert len(result) == 0
    
    def test_single_row_multiple_columns(self):
        """Dict with one row and multiple columns."""
        data = {
            'col1': [1],
            'col2': ['a'],
            'col3': [True]
        }
        result = ar.convert.from_dict(data)
        assert len(result) == 1
    
    def test_single_column_many_rows(self):
        """Dict with one column and many rows."""
        data = {'values': list(range(100))}
        result = ar.convert.from_dict(data)
        assert len(result) == 100
        assert len(result.columns) == 1
    
    def test_all_duplicate_values(self):
        """Dict where all values in columns are identical."""
        data = {
            'constant': [42, 42, 42, 42],
            'repeated': ['same', 'same', 'same', 'same']
        }
        result = ar.convert.from_dict(data)
        assert len(result) == 4


class TestFromDictFileIOWorkflow:
    """Test end-to-end workflow with file I/O."""
    
    def test_json_to_dict_to_arframe_to_csv(self, tmp_path):
        """Complete workflow: JSON file → dict → ArFrame → CSV file."""
        # Create JSON source file
        source_data = {
            'id': [1, 2, 3],
            'score': [95.5, 87.3, 92.1],
            'name': ['Alice', 'Bob', 'Charlie']
        }
        json_file = tmp_path / "input.json"
        with open(json_file, 'w') as f:
            json.dump(source_data, f)
        
        # Load from JSON and convert with from_dict
        with open(json_file, 'r') as f:
            loaded_dict = json.load(f)
        
        arframe = ar.convert.from_dict(loaded_dict)
        assert len(arframe) == 3
        
        # Save output using pandas if available
        output_file = tmp_path / "output.csv"
        if hasattr(arframe, 'to_pandas'):
            df = arframe.to_pandas()
            df.to_csv(str(output_file), index=False)
            assert output_file.exists()
            
            # Verify roundtrip
            df_verify = pd.read_csv(str(output_file))
            assert len(df_verify) == 3
            assert 'id' in df_verify.columns
