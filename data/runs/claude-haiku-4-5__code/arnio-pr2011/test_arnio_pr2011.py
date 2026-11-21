import pytest
import pandas as pd
import arnio as ar


@pytest.fixture
def sample_frame():
    """Create a sample ArFrame for testing."""
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6],
        'C': [7, 8, 9],
        'D': [10, 11, 12]
    })
    return ar.ArFrame(df)


class TestDropColumns:
    """Test cases for ArFrame.drop_columns"""
    
    def test_drop_single_column(self, sample_frame):
        """Drop a single column from the frame"""
        result = sample_frame.drop_columns(['A'])
        assert list(result.columns) == ['B', 'C', 'D']
    
    def test_drop_multiple_columns_list_and_tuple(self, sample_frame):
        """Drop multiple columns using list or tuple"""
        result_list = sample_frame.drop_columns(['A', 'C'])
        result_tuple = sample_frame.drop_columns(('B', 'D'))
        assert list(result_list.columns) == ['B', 'D']
        assert list(result_tuple.columns) == ['A', 'C']
    
    def test_drop_columns_empty_input_returns_copy(self, sample_frame):
        """Empty list returns unchanged copy of the frame"""
        result = sample_frame.drop_columns([])
        assert list(result.columns) == list(sample_frame.columns)
        assert result is not sample_frame
    
    def test_drop_columns_duplicates_silently_ignored(self, sample_frame):
        """Duplicate column names in input are silently ignored"""
        result = sample_frame.drop_columns(['A', 'A', 'B', 'B'])
        assert list(result.columns) == ['C', 'D']
    
    def test_drop_columns_preserves_order(self, sample_frame):
        """Original column order is preserved after dropping"""
        result = sample_frame.drop_columns(['B', 'D'])
        assert list(result.columns) == ['A', 'C']
    
    def test_drop_columns_nonexistent_column_raises_valueerror(self, sample_frame):
        """Raises ValueError when column doesn't exist"""
        with pytest.raises(ValueError, match="Unknown column"):
            sample_frame.drop_columns(['E'])
    
    def test_drop_columns_invalid_type_raises_typeerror(self, sample_frame):
        """Raises TypeError when cols is not list or tuple"""
        with pytest.raises(TypeError, match="cols must be a list or tuple"):
            sample_frame.drop_columns({'A', 'B'})
    
    def test_drop_columns_non_string_element_raises_typeerror(self, sample_frame):
        """Raises TypeError when column names are not strings"""
        with pytest.raises(TypeError, match="All column names in cols must be strings"):
            sample_frame.drop_columns(['A', 1])
    
    def test_drop_all_columns_raises_valueerror(self, sample_frame):
        """Raises ValueError when trying to remove all columns"""
        with pytest.raises(ValueError, match="cannot remove all columns"):
            sample_frame.drop_columns(['A', 'B', 'C', 'D'])


class TestSelectColumns:
    """Test cases for ArFrame.select_columns"""
    
    def test_select_single_column(self, sample_frame):
        """Select a single column from the frame"""
        result = sample_frame.select_columns(['A'])
        assert list(result.columns) == ['A']
    
    def test_select_multiple_columns_with_reorder(self, sample_frame):
        """Select multiple columns and reorder them"""
        result = sample_frame.select_columns(['D', 'B', 'A'])
        assert list(result.columns) == ['D', 'B', 'A']
    
    def test_select_columns_empty_raises_valueerror(self, sample_frame):
        """Raises ValueError when selection is empty"""
        with pytest.raises(ValueError, match="Column selection cannot be empty"):
            sample_frame.select_columns([])
    
    def test_select_columns_duplicates_raises_valueerror(self, sample_frame):
        """Raises ValueError when column names are duplicated"""
        with pytest.raises(ValueError, match="Duplicate column names"):
            sample_frame.select_columns(['A', 'B', 'A'])
    
    def test_select_columns_unknown_raises_valueerror(self, sample_frame):
        """Raises ValueError when column doesn't exist"""
        with pytest.raises(ValueError, match="Unknown columns"):
            sample_frame.select_columns(['A', 'Z'])
    
    def test_select_columns_invalid_input_raises_typeerror(self, sample_frame):
        """Raises TypeError for invalid input types and names"""
        with pytest.raises(TypeError, match="must be a sequence"):
            sample_frame.select_columns('A')
        
        with pytest.raises(TypeError, match="must be a list or tuple"):
            sample_frame.select_columns({'A', 'B'})
        
        with pytest.raises(TypeError, match="All column names must be strings"):
            sample_frame.select_columns(['A', 1])
