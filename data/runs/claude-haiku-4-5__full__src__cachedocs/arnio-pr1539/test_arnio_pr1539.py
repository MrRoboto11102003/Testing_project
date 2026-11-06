"""
Pytest tests for arnio.frame.ArFrame.drop_columns

Tests cover normal usage, edge cases, and error conditions as documented.
"""

import pytest
import pandas as pd
import arnio as ar


class TestArFrameDropColumns:
    """Tests for ArFrame.drop_columns method."""

    @pytest.fixture
    def sample_frame(self):
        """Create a sample ArFrame with 5 columns for testing."""
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "c": [7, 8, 9],
            "d": [10, 11, 12],
            "e": [13, 14, 15],
        })
        return ar.from_pandas(df)

    def test_drop_single_column(self, sample_frame):
        """Test dropping a single column."""
        result = sample_frame.drop_columns(["b"])
        assert result.columns == ["a", "c", "d", "e"]
        assert result.shape == (3, 4)

    def test_drop_multiple_columns_with_list(self, sample_frame):
        """Test dropping multiple columns using a list."""
        result = sample_frame.drop_columns(["b", "d"])
        assert result.columns == ["a", "c", "e"]
        assert result.shape == (3, 3)

    def test_drop_columns_with_tuple(self, sample_frame):
        """Test that tuple input is accepted."""
        result = sample_frame.drop_columns(("b", "d"))
        assert result.columns == ["a", "c", "e"]

    def test_drop_columns_preserves_order(self, sample_frame):
        """Test that remaining columns preserve original order."""
        result = sample_frame.drop_columns(["e", "a", "c"])
        assert result.columns == ["b", "d"]

    def test_drop_columns_with_duplicates_ignored(self, sample_frame):
        """Test that duplicates in cols are silently ignored."""
        result = sample_frame.drop_columns(["b", "b", "d", "d"])
        assert result.columns == ["a", "c", "e"]
        assert result.shape == (3, 3)

    def test_drop_columns_empty_list_unchanged(self, sample_frame):
        """Test that empty list returns a copy unchanged."""
        result = sample_frame.drop_columns([])
        assert result.columns == sample_frame.columns
        assert result.shape == sample_frame.shape
        assert result is not sample_frame  # Verify it's a copy

    def test_drop_columns_data_preserved(self, sample_frame):
        """Test that data is preserved for remaining columns."""
        result = sample_frame.drop_columns(["b", "c"])
        result_df = ar.to_pandas(result)
        expected_df = ar.to_pandas(sample_frame)[["a", "d", "e"]]
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_drop_columns_returns_arframe(self, sample_frame):
        """Test that drop_columns returns an ArFrame instance."""
        result = sample_frame.drop_columns(["b"])
        assert isinstance(result, ar.ArFrame)

    def test_drop_columns_original_not_modified(self, sample_frame):
        """Test that original frame is not modified."""
        original_columns = sample_frame.columns.copy()
        sample_frame.drop_columns(["b"])
        assert sample_frame.columns == original_columns

    def test_drop_columns_type_error_not_list_or_tuple(self, sample_frame):
        """Test TypeError when cols is not a list or tuple."""
        with pytest.raises(TypeError, match="cols must be a list or tuple"):
            sample_frame.drop_columns("b")

        with pytest.raises(TypeError, match="cols must be a list or tuple"):
            sample_frame.drop_columns({"b"})

    def test_drop_columns_type_error_non_string_element(self, sample_frame):
        """Test TypeError when cols contains non-string elements."""
        with pytest.raises(TypeError, match="All column names in cols must be strings"):
            sample_frame.drop_columns(["b", 1])

        with pytest.raises(TypeError, match="All column names in cols must be strings"):
            sample_frame.drop_columns([1, 2])

    def test_drop_columns_value_error_nonexistent_column(self, sample_frame):
        """Test ValueError when column name does not exist."""
        with pytest.raises(ValueError, match="Unknown column"):
            sample_frame.drop_columns(["b", "nonexistent"])

    def test_drop_columns_value_error_all_columns(self, sample_frame):
        """Test ValueError when attempting to drop all columns."""
        with pytest.raises(ValueError, match="cannot remove all columns"):
            sample_frame.drop_columns(["a", "b", "c", "d", "e"])

    def test_drop_columns_single_column_remaining(self, sample_frame):
        """Test dropping all but one column."""
        result = sample_frame.drop_columns(["a", "b", "c", "d"])
        assert result.columns == ["e"]
        assert result.shape == (3, 1)

    def test_drop_columns_multiple_frames_independent(self, sample_frame):
        """Test that operations on multiple frames are independent."""
        result1 = sample_frame.drop_columns(["a"])
        result2 = sample_frame.drop_columns(["b"])
        assert result1.columns == ["b", "c", "d", "e"]
        assert result2.columns == ["a", "c", "d", "e"]
        assert sample_frame.columns == ["a", "b", "c", "d", "e"]
