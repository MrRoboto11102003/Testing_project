import pytest
import pandas as pd
import arnio as ar
from arnio.frame import ArFrame


class TestArFrameDropColumns:
    """Test suite for ArFrame.drop_columns method."""

    @pytest.fixture
    def sample_frame(self):
        """Create a sample ArFrame with multiple columns."""
        data = {
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c'],
            'col3': [1.1, 2.2, 3.3],
            'col4': [True, False, True],
            'col5': [None, 'x', 'y']
        }
        df = pd.DataFrame(data)
        return ar.from_pandas(df)

    def test_drop_single_column_with_list(self, sample_frame):
        """Test dropping a single column specified as list."""
        result = sample_frame.drop_columns(['col1'])
        assert len(result.columns) == 4
        assert 'col1' not in result.columns
        assert result.shape[0] == 3

    def test_drop_multiple_columns_with_list(self, sample_frame):
        """Test dropping multiple columns specified as list."""
        result = sample_frame.drop_columns(['col1', 'col3', 'col5'])
        assert set(result.columns) == {'col2', 'col4'}
        assert len(result.columns) == 2

    def test_drop_columns_with_tuple(self, sample_frame):
        """Test dropping columns specified as tuple."""
        result = sample_frame.drop_columns(('col1', 'col2'))
        assert set(result.columns) == {'col3', 'col4', 'col5'}

    def test_drop_columns_preserves_original_order(self, sample_frame):
        """Test that dropping maintains original column order."""
        result = sample_frame.drop_columns(['col2', 'col4'])
        assert result.columns == ['col1', 'col3', 'col5']

    def test_empty_list_returns_unchanged_copy(self, sample_frame):
        """Test that empty list returns unchanged copy."""
        result = sample_frame.drop_columns([])
        assert result.columns == sample_frame.columns
        assert result.shape == sample_frame.shape
        assert result is not sample_frame

    def test_duplicate_column_names_silently_ignored(self, sample_frame):
        """Test that duplicate column names are silently ignored."""
        result = sample_frame.drop_columns(['col1', 'col1', 'col3', 'col3'])
        assert set(result.columns) == {'col2', 'col4', 'col5'}

    def test_returns_arframe_instance(self, sample_frame):
        """Test that result is an ArFrame instance."""
        result = sample_frame.drop_columns(['col1'])
        assert isinstance(result, ArFrame)

    def test_original_frame_not_modified(self, sample_frame):
        """Test that original frame remains unchanged."""
        original_cols = sample_frame.columns.copy()
        _ = sample_frame.drop_columns(['col1', 'col2'])
        assert sample_frame.columns == original_cols

    def test_error_when_cols_not_list_or_tuple(self, sample_frame):
        """Test TypeError when cols is not a list or tuple."""
        with pytest.raises(TypeError):
            sample_frame.drop_columns("col1")
        with pytest.raises(TypeError):
            sample_frame.drop_columns({'col1'})
        with pytest.raises(TypeError):
            sample_frame.drop_columns(42)

    def test_error_when_non_string_column_names(self, sample_frame):
        """Test TypeError when cols contains non-string elements."""
        with pytest.raises(TypeError):
            sample_frame.drop_columns([1, 2])
        with pytest.raises(TypeError):
            sample_frame.drop_columns(['col1', None])

    def test_error_when_column_does_not_exist(self, sample_frame):
        """Test ValueError when trying to drop non-existent column."""
        with pytest.raises(ValueError):
            sample_frame.drop_columns(['nonexistent'])
        with pytest.raises(ValueError):
            sample_frame.drop_columns(['col1', 'nonexistent', 'col3'])

    def test_error_when_drop_all_columns(self, sample_frame):
        """Test ValueError when attempting to drop all columns."""
        with pytest.raises(ValueError):
            sample_frame.drop_columns(sample_frame.columns)

    def test_error_for_single_column_frame(self):
        """Test that single-column frame cannot drop its only column."""
        data = {'only_col': [1, 2, 3]}
        df = pd.DataFrame(data)
        frame = ar.from_pandas(df)
        with pytest.raises(ValueError):
            frame.drop_columns(['only_col'])

    def test_column_names_case_sensitive(self, sample_frame):
        """Test that column name matching is case sensitive."""
        with pytest.raises(ValueError):
            sample_frame.drop_columns(['COL1'])

    def test_data_preserved_in_remaining_columns(self, sample_frame):
        """Test that data in remaining columns is preserved correctly."""
        result = sample_frame.drop_columns(['col1', 'col4'])
        df = ar.to_pandas(result)
        assert df['col2'].tolist() == ['a', 'b', 'c']
        assert len(df) == 3
