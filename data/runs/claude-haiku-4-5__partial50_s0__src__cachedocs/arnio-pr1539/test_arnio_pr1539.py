import pytest
import arnio as ar


class TestDropColumns:
    """Test suite for ArFrame.drop_columns()"""

    @pytest.fixture
    def frame(self):
        """Create a sample ArFrame for testing."""
        return ar.from_dict({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "email": ["a@x.com", "b@x.com", "c@x.com"],
            "age": [25, 30, 35],
        })

    def test_drop_single_column(self, frame):
        """Test dropping a single column."""
        result = frame.drop_columns(["id"])
        assert result.columns == ["name", "email", "age"]
        assert result.shape[0] == 3

    def test_drop_multiple_columns(self, frame):
        """Test dropping multiple columns."""
        result = frame.drop_columns(["id", "email"])
        assert result.columns == ["name", "age"]

    def test_drop_columns_with_tuple(self, frame):
        """Test that tuples are accepted."""
        result = frame.drop_columns(("id", "age"))
        assert result.columns == ["name", "email"]

    def test_preserves_column_order(self, frame):
        """Test original column order is preserved."""
        result = frame.drop_columns(["name", "age"])
        assert result.columns == ["id", "email"]

    def test_empty_list_returns_copy(self, frame):
        """Test empty list returns unchanged copy."""
        result = frame.drop_columns([])
        assert result.columns == frame.columns
        assert result is not frame

    def test_duplicate_columns_ignored(self, frame):
        """Test duplicates in input are silently ignored."""
        result = frame.drop_columns(["id", "id", "email", "email"])
        assert result.columns == ["name", "age"]

    def test_type_error_for_string_input(self, frame):
        """Test TypeError when cols is a string."""
        with pytest.raises(TypeError, match="cols must be a list or tuple"):
            frame.drop_columns("id")

    def test_type_error_for_dict_input(self, frame):
        """Test TypeError for invalid input type."""
        with pytest.raises(TypeError):
            frame.drop_columns({"id": None})

    def test_type_error_for_non_string_element(self, frame):
        """Test TypeError when cols contains non-string."""
        with pytest.raises(TypeError, match="All column names in cols must be strings"):
            frame.drop_columns(["id", 1])

    def test_type_error_for_none_in_cols(self, frame):
        """Test TypeError when cols contains None."""
        with pytest.raises(TypeError):
            frame.drop_columns([None, "id"])

    def test_value_error_for_nonexistent_column(self, frame):
        """Test ValueError for non-existent column."""
        with pytest.raises(ValueError, match="Unknown column"):
            frame.drop_columns(["nonexistent"])

    def test_value_error_for_partial_missing(self, frame):
        """Test ValueError when some columns don't exist."""
        with pytest.raises(ValueError, match="Unknown column"):
            frame.drop_columns(["id", "fake"])

    def test_value_error_remove_all_columns(self, frame):
        """Test ValueError when trying to drop all columns."""
        all_cols = frame.columns
        with pytest.raises(ValueError, match="cannot remove all columns"):
            frame.drop_columns(all_cols)

    def test_data_integrity_after_drop(self, frame):
        """Test that remaining data is correct after dropping."""
        result = frame.drop_columns(["email", "age"])
        df = ar.to_pandas(result)
        assert list(df["id"]) == [1, 2, 3]
        assert list(df["name"]) == ["Alice", "Bob", "Charlie"]

    def test_returns_arframe_instance(self, frame):
        """Test that result is an ArFrame."""
        result = frame.drop_columns(["id"])
        assert isinstance(result, ar.ArFrame)
