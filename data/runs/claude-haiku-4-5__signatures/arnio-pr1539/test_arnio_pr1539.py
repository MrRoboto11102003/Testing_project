import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestArFrameDropColumns:
    """Test suite for ArFrame.drop_columns method"""
    
    @pytest.fixture
    def sample_frame(self):
        """Create a sample ArFrame for testing"""
        data = {
            'col_a': [1, 2, 3, 4],
            'col_b': [5, 6, 7, 8],
            'col_c': [9, 10, 11, 12],
            'col_d': ['a', 'b', 'c', 'd']
        }
        df = pd.DataFrame(data)
        return ar.ArFrame(df)
    
    def test_drop_single_column_with_list(self, sample_frame):
        """Test dropping a single column using a list"""
        result = sample_frame.drop_columns(['col_a'])
        assert 'col_a' not in result.columns
        assert len(result.columns) == 3
        assert 'col_b' in result.columns
    
    def test_drop_single_column_with_tuple(self, sample_frame):
        """Test dropping a single column using a tuple"""
        result = sample_frame.drop_columns(('col_b',))
        assert 'col_b' not in result.columns
        assert len(result.columns) == 3
    
    def test_drop_multiple_columns(self, sample_frame):
        """Test dropping multiple columns at once"""
        result = sample_frame.drop_columns(['col_a', 'col_c'])
        assert 'col_a' not in result.columns
        assert 'col_c' not in result.columns
        assert 'col_b' in result.columns
        assert 'col_d' in result.columns
        assert len(result.columns) == 2
    
    def test_drop_all_columns(self, sample_frame):
        """Test dropping all columns results in empty frame"""
        result = sample_frame.drop_columns(['col_a', 'col_b', 'col_c', 'col_d'])
        assert len(result.columns) == 0
        assert len(result) == len(sample_frame)
    
    def test_drop_empty_list(self, sample_frame):
        """Test dropping with empty list returns unchanged frame"""
        result = sample_frame.drop_columns([])
        pd.testing.assert_frame_equal(result.to_pandas(), sample_frame.to_pandas())
    
    def test_drop_empty_tuple(self, sample_frame):
        """Test dropping with empty tuple returns unchanged frame"""
        result = sample_frame.drop_columns(())
        pd.testing.assert_frame_equal(result.to_pandas(), sample_frame.to_pandas())
    
    def test_returns_arframe_instance(self, sample_frame):
        """Test that method returns an ArFrame instance"""
        result = sample_frame.drop_columns(['col_a'])
        assert isinstance(result, ar.ArFrame)
    
    def test_data_integrity_after_drop(self, sample_frame):
        """Test that remaining data is preserved correctly"""
        result = sample_frame.drop_columns(['col_b', 'col_d'])
        assert list(result['col_a']) == [1, 2, 3, 4]
        assert list(result['col_c']) == [9, 10, 11, 12]
    
    def test_original_frame_unchanged(self, sample_frame):
        """Test that original frame is not modified (immutability)"""
        original_cols = set(sample_frame.columns)
        result = sample_frame.drop_columns(['col_a'])
        assert set(sample_frame.columns) == original_cols
        assert 'col_a' in sample_frame.columns
    
    def test_preserves_row_index(self, sample_frame):
        """Test that row count and index are preserved"""
        result = sample_frame.drop_columns(['col_a', 'col_b'])
        assert len(result) == len(sample_frame)
        pd.testing.assert_index_equal(result.index, sample_frame.index)
    
    def test_drop_nonexistent_column_raises_error(self, sample_frame):
        """Test that dropping non-existent column raises KeyError"""
        with pytest.raises(KeyError):
            sample_frame.drop_columns(['nonexistent_col'])
    
    def test_drop_mixed_existent_nonexistent_raises_error(self, sample_frame):
        """Test that dropping mix of existent and non-existent columns raises error"""
        with pytest.raises(KeyError):
            sample_frame.drop_columns(['col_a', 'nonexistent_col'])
    
    def test_drop_columns_with_mixed_dtypes(self):
        """Test dropping columns with various data types"""
        data = {
            'int_col': [1, 2, 3],
            'float_col': [1.5, 2.5, 3.5],
            'str_col': ['a', 'b', 'c'],
            'bool_col': [True, False, True]
        }
        df = pd.DataFrame(data)
        frame = ar.ArFrame(df)
        result = frame.drop_columns(['float_col', 'bool_col'])
        assert 'int_col' in result.columns
        assert 'str_col' in result.columns
        assert 'float_col' not in result.columns
        assert 'bool_col' not in result.columns
    
    def test_column_order_preserved(self, sample_frame):
        """Test that remaining columns maintain their order"""
        result = sample_frame.drop_columns(['col_b'])
        expected_order = ['col_a', 'col_c', 'col_d']
        assert list(result.columns) == expected_order
