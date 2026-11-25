import pytest
import arnio as ar


class TestArFrameDropColumns:
    """Tests for ArFrame.drop_columns method."""

    @pytest.fixture
    def sample_frame(self):
        """Create a sample ArFrame for testing."""
        data = {
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
            "city": ["NYC", "LA", "Chicago"],
        }
        return ar.from_dict(data)

    def test_drop_single_column(self, sample_frame):
        """Test dropping a single column."""
        result = sample_frame.drop_columns(["name"])
        assert "name" not in result.columns
        assert list(result.columns) == ["id", "age", "city"]

    def test_drop_multiple_columns(self, sample_frame):
        """Test dropping multiple columns."""
        result = sample_frame.drop_columns(["name", "age"])
        assert "name" not in result.columns
        assert "age" not in result.columns
        assert list(result.columns) == ["id", "city"]

    def test_drop_columns_with_tuple_input(self, sample_frame):
        """Test that tuple input works like list."""
        result = sample_frame.drop_columns(("name", "age"))
        assert list(result.columns) == ["id", "city"]

    def test_drop_columns_duplicates_silently_ignored(self, sample_frame):
        """Test that duplicate names in drop list are silently ignored."""
        result = sample_frame.drop_columns(["name", "name", "age"])
        assert list(result.columns) == ["id", "city"]

    def test_drop_columns_empty_list_returns_copy(self, sample_frame):
        """Test that empty list returns a copy with all columns unchanged."""
        result = sample_frame.drop_columns([])
        assert list(result.columns) == list(sample_frame.columns)
        assert result is not sample_frame

    def test_column_order_preserved(self, sample_frame):
        """Test that original column order is preserved."""
        result = sample_frame.drop_columns(["age", "name"])
        assert list(result.columns) == ["id", "city"]

    def test_nonexistent_column_raises_value_error(self, sample_frame):
        """Test that ValueError is raised for non-existent columns."""
        with pytest.raises(ValueError):
            sample_frame.drop_columns(["nonexistent"])

    def test_any_nonexistent_column_raises_value_error(self, sample_frame):
        """Test that ValueError is raised if any column doesn't exist."""
        with pytest.raises(ValueError):
            sample_frame.drop_columns(["name", "nonexistent", "age"])

    def test_wrong_type_for_cols_raises_type_error(self, sample_frame):
        """Test that TypeError is raised for invalid cols type."""
        with pytest.raises(TypeError):
            sample_frame.drop_columns("name")
        
        with pytest.raises(TypeError):
            sample_frame.drop_columns(123)

    def test_non_string_element_raises_type_error(self, sample_frame):
        """Test that TypeError is raised for non-string column names."""
        with pytest.raises(TypeError):
            sample_frame.drop_columns(["name", 123])

    def test_original_frame_not_modified(self, sample_frame):
        """Test that the original frame is not modified."""
        original_columns = list(sample_frame.columns)
        _ = sample_frame.drop_columns(["name"])
        assert list(sample_frame.columns) == original_columns

    def test_data_integrity_after_drop(self, sample_frame):
        """Test that remaining columns have correct data after drop."""
        result = sample_frame.drop_columns(["name", "city"])
        result_pd = ar.to_pandas(result)
        
        assert list(result_pd.columns) == ["id", "age"]
        assert result_pd["id"].tolist() == [1, 2, 3]
        assert result_pd["age"].tolist() == [25, 30, 35]

    def test_returns_arframe_instance(self, sample_frame):
        """Test that drop_columns returns an ArFrame instance."""
        result = sample_frame.drop_columns(["name"])
        assert isinstance(result, ar.ArFrame)
