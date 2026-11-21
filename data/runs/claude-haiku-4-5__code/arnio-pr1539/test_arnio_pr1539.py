import pytest
import pandas as pd
import arnio as ar


class TestArFrameDropColumns:
    """Test suite for ArFrame.drop_columns method."""
    
    @pytest.fixture
    def sample_frame(self):
        """Create a sample ArFrame for testing."""
        data = {
            'col1': [1, 2, 3],
            'col2': [4, 5, 6],
            'col3': [7, 8, 9],
            'col4': [10, 11, 12]
        }
        df = pd.DataFrame(data)
        return ar.frame.ArFrame(df)
    
    def test_drop_single_column_list(self, sample_frame):
        """Drop a single column with list input."""
        result = sample_frame.drop_columns(['col1'])
        assert 'col1' not in result.columns
        assert list(result.columns) == ['col2', 'col3', 'col4']
    
    def test_drop_single_column_tuple(self, sample_frame):
        """Drop a single column with tuple input."""
        result = sample_frame.drop_columns(('col1',))
        assert 'col1' not in result.columns
        assert list(result.columns) == ['col2', 'col3', 'col4']
    
    def test_drop_multiple_columns(self, sample_frame):
        """Drop multiple columns at once."""
        result = sample_frame.drop_columns(['col1', 'col3'])
        assert list(result.columns) == ['col2', 'col4']
    
    def test_column_order_preserved(self, sample_frame):
        """Original column order is preserved after dropping."""
        result = sample_frame.drop_columns(['col3', 'col1'])
        assert list(result.columns) == ['col2', 'col4']
    
    def test_duplicate_columns_ignored(self, sample_frame):
        """Duplicate column names in cols are silently ignored."""
        result = sample_frame.drop_columns(['col1', 'col1', 'col3', 'col3'])
        assert list(result.columns) == ['col2', 'col4']
    
    def test_empty_input_returns_copy(self, sample_frame):
        """Empty list or tuple returns unchanged copy."""
        result_list = sample_frame.drop_columns([])
        result_tuple = sample_frame.drop_columns(())
        assert list(result_list.columns) == list(sample_frame.columns)
        assert list(result_tuple.columns) == list(sample_frame.columns)
        assert result_list is not sample_frame
        assert result_tuple is not sample_frame
    
    def test_drop_all_but_one(self, sample_frame):
        """Dropping all but one column is allowed."""
        result = sample_frame.drop_columns(['col1', 'col2', 'col3'])
        assert list(result.columns) == ['col4']
    
    def test_returns_arframe(self, sample_frame):
        """Result is an ArFrame instance."""
        result = sample_frame.drop_columns(['col1'])
        assert isinstance(result, ar.frame.ArFrame)
    
    def test_original_frame_unchanged(self, sample_frame):
        """Original frame is not modified."""
        original_cols = list(sample_frame.columns)
        sample_frame.drop_columns(['col1'])
        assert list(sample_frame.columns) == original_cols
    
    def test_error_invalid_type(self, sample_frame):
        """TypeError for non-list/tuple cols argument."""
        with pytest.raises(TypeError, match="cols must be a list or tuple"):
            sample_frame.drop_columns({'col1': None})
        with pytest.raises(TypeError, match="cols must be a list or tuple"):
            sample_frame.drop_columns('col1')
    
    def test_error_non_string_element(self, sample_frame):
        """TypeError when cols contains non-string elements."""
        with pytest.raises(TypeError, match="All column names in cols must be strings"):
            sample_frame.drop_columns(['col1', 123])
        with pytest.raises(TypeError, match="All column names in cols must be strings"):
            sample_frame.drop_columns(['col1', None])
    
    def test_error_unknown_column(self, sample_frame):
        """ValueError when column name doesn't exist."""
        with pytest.raises(ValueError, match="Unknown column"):
            sample_frame.drop_columns(['nonexistent'])
        with pytest.raises(ValueError, match="Unknown column"):
            sample_frame.drop_columns(['col1', 'fake_col'])
    
    def test_error_drop_all_columns(self, sample_frame):
        """ValueError when trying to drop all columns."""
        all_cols = list(sample_frame.columns)
        with pytest.raises(ValueError, match="cannot remove all columns"):
            sample_frame.drop_columns(all_cols)
