import pytest
import pandas as pd
import arnio as ar


class TestArFrameDropColumns:
    """Test suite for ArFrame.drop_columns()"""

    @pytest.fixture
    def sample_frame(self):
        """Create a sample ArFrame for testing."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
            "city": ["NYC", "LA", "SF"],
        })
        return ar.from_pandas(df)

    def test_drop_single_column_with_list(self, sample_frame):
        """Drop a single column using list."""
        result = sample_frame.drop_columns(["id"])
        assert "id" not in result.columns
        assert set(result.columns) == {"name", "age", "city"}
        assert result.shape[1] == 3

    def test_drop_single_column_with_tuple(self, sample_frame):
        """Drop a single column using tuple."""
        result = sample_frame.drop_columns(("name",))
        assert "name" not in result.columns
        assert set(result.columns) == {"id", "age", "city"}

    def test_drop_multiple_columns(self, sample_frame):
        """Drop multiple columns at once."""
        result = sample_frame.drop_columns(["id", "age"])
        assert "id" not in result.columns
        assert "age" not in result.columns
        assert set(result.columns) == {"name", "city"}
        assert result.shape[1] == 2

    def test_drop_with_duplicates_silently_ignored(self, sample_frame):
        """Duplicate column names in drop list are silently ignored."""
        result = sample_frame.drop_columns(["id", "id", "name"])
        assert set(result.columns) == {"age", "city"}

    def test_empty_list_returns_copy_unchanged(self, sample_frame):
        """Empty drop list returns a copy with all columns intact."""
        result = sample_frame.drop_columns([])
        assert result.columns == sample_frame.columns
        assert result.shape == sample_frame.shape
        # Verify it's a copy, not the same object
        assert result is not sample_frame

    def test_preserves_original_column_order(self, sample_frame):
        """Remaining columns maintain their original order."""
        result = sample_frame.drop_columns(["age"])
        assert result.columns == ["id", "name", "city"]

    def test_column_order_with_multiple_drops(self, sample_frame):
        """Column order preserved when dropping multiple columns."""
        result = sample_frame.drop_columns(["age", "city"])
        assert list(result.columns) == ["id", "name"]

    def test_does_not_mutate_original_frame(self, sample_frame):
        """Original frame is not modified by drop_columns."""
        original_columns = list(sample_frame.columns)
        sample_frame.drop_columns(["id", "name"])
        assert list(sample_frame.columns) == original_columns

    def test_returns_arframe_type(self, sample_frame):
        """Return type is ArFrame."""
        result = sample_frame.drop_columns(["id"])
        assert isinstance(result, ar.ArFrame)

    def test_data_integrity_after_drop(self, sample_frame):
        """Data in remaining columns is preserved correctly."""
        result = sample_frame.drop_columns(["age"])
        result_df = ar.to_pandas(result)
        original_df = ar.to_pandas(sample_frame)
        
        pd.testing.assert_frame_equal(
            result_df[["id", "name", "city"]],
            original_df[["id", "name", "city"]]
        )

    def test_type_error_when_cols_is_string(self, sample_frame):
        """TypeError raised when cols is a string instead of list/tuple."""
        with pytest.raises(TypeError):
            sample_frame.drop_columns("id")

    def test_type_error_when_cols_is_set(self, sample_frame):
        """TypeError raised when cols is a set."""
        with pytest.raises(TypeError):
            sample_frame.drop_columns({"id", "name"})

    def test_type_error_non_string_elements(self, sample_frame):
        """TypeError raised when cols contains non-string elements."""
        with pytest.raises(TypeError):
            sample_frame.drop_columns([1, 2])

        with pytest.raises(TypeError):
            sample_frame.drop_columns(["id", 42])

    def test_value_error_column_not_found(self, sample_frame):
        """ValueError raised when trying to drop a non-existent column."""
        with pytest.raises(ValueError):
            sample_frame.drop_columns(["nonexistent"])

    def test_value_error_when_any_column_not_found(self, sample_frame):
        """ValueError raised if any column in the list doesn't exist."""
        with pytest.raises(ValueError):
            sample_frame.drop_columns(["id", "nonexistent_col"])

    def test_drop_all_but_one_column(self, sample_frame):
        """Can drop all columns except one."""
        result = sample_frame.drop_columns(["id", "age", "city"])
        assert result.columns == ["name"]
        assert result.shape[1] == 1
