import pytest
import pandas as pd
import arnio as ar


@pytest.fixture
def sample_frame():
    """Create a simple ArFrame for testing."""
    df = pd.DataFrame({
        'a': [1, 2, 3],
        'b': [4, 5, 6],
        'c': [7, 8, 9],
        'd': [10, 11, 12]
    })
    return ar.from_pandas(df)


class TestDropColumnsBasic:
    """Test basic functionality of drop_columns."""
    
    def test_drop_single_column(self, sample_frame):
        """Test dropping a single column."""
        result = ar.drop_columns(sample_frame, ['a'])
        assert list(result.columns) == ['b', 'c', 'd']
    
    def test_drop_multiple_columns(self, sample_frame):
        """Test dropping multiple columns."""
        result = ar.drop_columns(sample_frame, ['a', 'c'])
        assert list(result.columns) == ['b', 'd']
    
    def test_drop_all_but_one(self, sample_frame):
        """Test dropping all but one column."""
        result = ar.drop_columns(sample_frame, ['a', 'b', 'c'])
        assert list(result.columns) == ['d']
    
    def test_column_order_preserved(self, sample_frame):
        """Test that remaining column order is preserved."""
        result = ar.drop_columns(sample_frame, ['b'])
        assert list(result.columns) == ['a', 'c', 'd']
    
    def test_data_preserved(self, sample_frame):
        """Test that data in remaining columns is preserved."""
        result = ar.drop_columns(sample_frame, ['a', 'b'])
        result_df = ar.to_pandas(result)
        expected_df = ar.to_pandas(sample_frame)[['c', 'd']]
        pd.testing.assert_frame_equal(result_df, expected_df)


class TestDropColumnsEdgeCases:
    """Test edge cases for drop_columns."""
    
    def test_drop_empty_list(self, sample_frame):
        """Test dropping with empty column list returns all columns."""
        result = ar.drop_columns(sample_frame, [])
        assert list(result.columns) == ['a', 'b', 'c', 'd']
    
    def test_drop_all_columns_raises_error(self, sample_frame):
        """Test that dropping all columns raises ValueError."""
        with pytest.raises(ValueError, match="cannot remove all columns"):
            ar.drop_columns(sample_frame, ['a', 'b', 'c', 'd'])
    
    def test_drop_columns_with_tuple(self, sample_frame):
        """Test that drop_columns works with tuple input."""
        result = ar.drop_columns(sample_frame, ('a', 'b'))
        assert list(result.columns) == ['c', 'd']


class TestDropColumnsErrors:
    """Test error handling in drop_columns."""
    
    def test_missing_column_raises_value_error(self, sample_frame):
        """Test that requesting non-existent column raises ValueError."""
        with pytest.raises(ValueError, match="Columns not found"):
            ar.drop_columns(sample_frame, ['nonexistent'])
    
    def test_string_instead_of_sequence_raises_type_error(self, sample_frame):
        """Test that passing a string instead of sequence raises TypeError."""
        with pytest.raises(TypeError):
            ar.drop_columns(sample_frame, 'a')
    
    def test_bytes_instead_of_sequence_raises_type_error(self, sample_frame):
        """Test that passing bytes instead of sequence raises TypeError."""
        with pytest.raises(TypeError):
            ar.drop_columns(sample_frame, b'a')
    
    def test_non_string_column_names_raise_type_error(self, sample_frame):
        """Test that non-string column names raise TypeError."""
        with pytest.raises(TypeError):
            ar.drop_columns(sample_frame, [1, 2])
    
    def test_mixed_valid_and_invalid_columns_raise_error(self, sample_frame):
        """Test that mixed valid and invalid columns raise ValueError."""
        with pytest.raises(ValueError):
            ar.drop_columns(sample_frame, ['a', 'nonexistent'])


class TestDropColumnsReturnType:
    """Test return type and immutability."""
    
    def test_returns_arframe_type(self, sample_frame):
        """Test that return value is same type as input."""
        result = ar.drop_columns(sample_frame, ['a'])
        assert type(result) == type(sample_frame)
    
    def test_original_frame_unchanged(self, sample_frame):
        """Test that original frame is not modified."""
        original_columns = list(sample_frame.columns)
        ar.drop_columns(sample_frame, ['a'])
        assert list(sample_frame.columns) == original_columns
