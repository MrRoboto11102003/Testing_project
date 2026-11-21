import pytest
import pandas as pd
import arnio as ar


class TestDropColumns:
    """Test suite for arnio.cleaning.drop_columns function."""

    @pytest.fixture
    def sample_frame(self):
        """Create a sample ArFrame for testing."""
        data = {
            'col_a': [1, 2, 3, 4, 5],
            'col_b': ['x', 'y', 'z', 'w', 'v'],
            'col_c': [1.1, 2.2, 3.3, 4.4, 5.5],
            'col_d': [True, False, True, False, True]
        }
        df = pd.DataFrame(data)
        return ar.from_pandas(df)

    def test_drop_single_column(self, sample_frame):
        """Test dropping a single column."""
        result = ar.drop_columns(sample_frame, ['col_a'])
        assert result.shape == (5, 3)
        assert 'col_a' not in result.columns
        assert set(result.columns) == {'col_b', 'col_c', 'col_d'}

    def test_drop_multiple_columns(self, sample_frame):
        """Test dropping multiple columns."""
        result = ar.drop_columns(sample_frame, ['col_a', 'col_c'])
        assert result.shape == (5, 2)
        assert set(result.columns) == {'col_b', 'col_d'}

    def test_drop_all_but_one_column(self, sample_frame):
        """Test dropping all but one column (valid operation)."""
        result = ar.drop_columns(sample_frame, ['col_a', 'col_b', 'col_c'])
        assert result.shape == (5, 1)
        assert result.columns == ['col_d']

    def test_drop_no_columns_empty_list(self, sample_frame):
        """Test that passing empty list returns frame with same shape."""
        result = ar.drop_columns(sample_frame, [])
        assert result.shape == sample_frame.shape
        assert result.columns == sample_frame.columns

    def test_drop_all_columns_raises_error(self, sample_frame):
        """Test that attempting to drop all columns raises ValueError."""
        with pytest.raises(ValueError, match="cannot remove all columns"):
            ar.drop_columns(sample_frame, sample_frame.columns)

    def test_drop_nonexistent_column_raises_error(self, sample_frame):
        """Test that dropping non-existent column raises ValueError."""
        with pytest.raises(ValueError, match="Columns not found"):
            ar.drop_columns(sample_frame, ['nonexistent_col'])

    def test_drop_mixed_existent_nonexistent_raises_error(self, sample_frame):
        """Test that mix of existent and non-existent columns raises error."""
        with pytest.raises(ValueError, match="Columns not found"):
            ar.drop_columns(sample_frame, ['col_a', 'nonexistent_col'])

    def test_columns_as_string_raises_typeerror(self, sample_frame):
        """Test that passing columns as a string raises TypeError."""
        with pytest.raises(TypeError):
            ar.drop_columns(sample_frame, 'col_a')

    def test_columns_as_bytes_raises_typeerror(self, sample_frame):
        """Test that passing columns as bytes raises TypeError."""
        with pytest.raises(TypeError):
            ar.drop_columns(sample_frame, b'col_a')

    def test_columns_with_non_string_items_raises_typeerror(self, sample_frame):
        """Test that columns containing non-string items raises TypeError."""
        with pytest.raises(TypeError):
            ar.drop_columns(sample_frame, ['col_a', 123])

    def test_duplicate_columns_in_list(self, sample_frame):
        """Test dropping with duplicate column names in list."""
        result = ar.drop_columns(sample_frame, ['col_a', 'col_a', 'col_b'])
        assert result.shape == (5, 2)
        assert set(result.columns) == {'col_c', 'col_d'}

    def test_data_integrity_after_drop(self, sample_frame):
        """Test that remaining columns retain correct data."""
        result = ar.drop_columns(sample_frame, ['col_b'])
        result_df = ar.to_pandas(result)
        original_df = ar.to_pandas(sample_frame)
        expected_df = original_df[['col_a', 'col_c', 'col_d']]
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_column_order_preserved(self, sample_frame):
        """Test that column order is preserved after dropping."""
        result = ar.drop_columns(sample_frame, ['col_b'])
        assert result.columns == ['col_a', 'col_c', 'col_d']

    def test_drop_from_single_row_frame(self):
        """Test dropping columns from frame with single row."""
        df = pd.DataFrame({'col_a': [1], 'col_b': [2], 'col_c': [3]})
        frame = ar.from_pandas(df)
        result = ar.drop_columns(frame, ['col_a'])
        assert result.shape == (1, 2)
        assert set(result.columns) == {'col_b', 'col_c'}

    def test_drop_returns_new_frame_not_modified(self, sample_frame):
        """Test that drop_columns returns a new frame without modifying original."""
        original_cols = list(sample_frame.columns)
        result = ar.drop_columns(sample_frame, ['col_a'])
        # Original should be unchanged
        assert sample_frame.columns == original_cols
        # Result should be different
        assert result.columns != sample_frame.columns
