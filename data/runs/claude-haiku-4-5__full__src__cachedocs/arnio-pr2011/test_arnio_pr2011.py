import pytest
import arnio as ar
import pandas as pd


@pytest.fixture
def sample_frame():
    """Create a sample ArFrame for testing."""
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["alice", "bob", "charlie"],
        "age": [25, 30, 35],
        "city": ["NYC", "LA", "SF"],
    })
    return ar.from_pandas(df)


class TestDropColumns:
    """Tests for ArFrame.drop_columns"""

    def test_drop_single_column(self, sample_frame):
        """Drop a single column."""
        result = sample_frame.drop_columns(["id"])
        assert result.columns == ["name", "age", "city"]

    def test_drop_multiple_columns(self, sample_frame):
        """Drop multiple columns."""
        result = sample_frame.drop_columns(["id", "city"])
        assert result.columns == ["name", "age"]

    def test_drop_with_list_and_tuple_input(self, sample_frame):
        """Drop columns works with both list and tuple input."""
        result_list = sample_frame.drop_columns(["id", "age"])
        result_tuple = sample_frame.drop_columns(("id", "age"))
        assert result_list.columns == result_tuple.columns == ["name", "city"]

    def test_drop_with_duplicate_names_silently_ignored(self, sample_frame):
        """Duplicate column names in drop list are silently ignored."""
        result = sample_frame.drop_columns(["id", "id", "city"])
        assert result.columns == ["name", "age"]

    def test_drop_empty_list_returns_copy(self, sample_frame):
        """Empty list returns unchanged copy."""
        result = sample_frame.drop_columns([])
        assert result.columns == sample_frame.columns
        assert result.shape == sample_frame.shape

    def test_drop_preserves_column_order(self, sample_frame):
        """Dropping non-contiguous columns preserves original order."""
        result = sample_frame.drop_columns(["name", "city"])
        assert result.columns == ["id", "age"]

    def test_drop_type_error_invalid_input_type(self, sample_frame):
        """Raises TypeError if cols is not list or tuple."""
        with pytest.raises(TypeError):
            sample_frame.drop_columns("id")
        with pytest.raises(TypeError):
            sample_frame.drop_columns({"id"})

    def test_drop_type_error_non_string_element(self, sample_frame):
        """Raises TypeError if cols contains non-string elements."""
        with pytest.raises(TypeError):
            sample_frame.drop_columns(["id", 123])

    def test_drop_value_error_unknown_column(self, sample_frame):
        """Raises ValueError if column doesn't exist."""
        with pytest.raises(ValueError, match="Unknown column"):
            sample_frame.drop_columns(["nonexistent"])

    def test_drop_value_error_all_columns(self, sample_frame):
        """Raises ValueError when trying to drop all columns."""
        with pytest.raises(ValueError):
            sample_frame.drop_columns(["id", "name", "age", "city"])


class TestSelectColumns:
    """Tests for ArFrame.select_columns"""

    def test_select_single_column(self, sample_frame):
        """Select a single column."""
        result = sample_frame.select_columns(["id"])
        assert result.columns == ["id"]

    def test_select_multiple_columns_preserves_order(self, sample_frame):
        """Select multiple columns in specified order."""
        result = sample_frame.select_columns(["city", "name"])
        assert result.columns == ["city", "name"]

    def test_select_type_error_string_input(self, sample_frame):
        """Raises TypeError if columns is a string instead of sequence."""
        with pytest.raises(TypeError, match="sequence of column names, not a string"):
            sample_frame.select_columns("id")

    def test_select_value_error_empty_selection(self, sample_frame):
        """Raises ValueError if selection is empty."""
        with pytest.raises(ValueError, match="empty"):
            sample_frame.select_columns([])

    def test_select_value_error_duplicates(self, sample_frame):
        """Raises ValueError if selection contains duplicates."""
        with pytest.raises(ValueError, match="Duplicate"):
            sample_frame.select_columns(["id", "name", "id"])

    def test_select_type_error_non_string_element(self, sample_frame):
        """Raises TypeError if columns contains non-string elements."""
        with pytest.raises(TypeError):
            sample_frame.select_columns(["id", 123])

    def test_select_value_error_unknown_column(self, sample_frame):
        """Raises ValueError if any column doesn't exist."""
        with pytest.raises(ValueError, match="Unknown"):
            sample_frame.select_columns(["nonexistent"])
