import pytest
import pandas as pd
import arnio as ar


@pytest.fixture
def sample_frame():
    """Create a simple ArFrame for testing."""
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["alice", "bob", "charlie"],
        "age": [25, 30, 35],
        "city": ["NYC", "LA", "SF"],
        "salary": [50000, 60000, 70000],
    })
    return ar.from_pandas(df)


class TestDropColumns:
    """Tests for ArFrame.drop_columns()"""

    def test_drop_single_column(self, sample_frame):
        """Drop a single column."""
        result = sample_frame.drop_columns(["name"])
        assert result.columns == ["id", "age", "city", "salary"]

    def test_drop_multiple_columns(self, sample_frame):
        """Drop multiple columns."""
        result = sample_frame.drop_columns(["name", "age"])
        assert result.columns == ["id", "city", "salary"]

    def test_drop_columns_with_tuple(self, sample_frame):
        """Accept tuple of column names."""
        result = sample_frame.drop_columns(("name", "age"))
        assert result.columns == ["id", "city", "salary"]

    def test_drop_columns_empty_list(self, sample_frame):
        """Empty list returns copy with all columns unchanged."""
        result = sample_frame.drop_columns([])
        assert result.columns == sample_frame.columns

    def test_drop_columns_preserves_order(self, sample_frame):
        """Dropping columns preserves original column order."""
        result = sample_frame.drop_columns(["age", "name"])
        assert result.columns == ["id", "city", "salary"]

    def test_drop_columns_ignores_duplicates(self, sample_frame):
        """Duplicate column names in drop list are silently ignored."""
        result = sample_frame.drop_columns(["name", "name", "age"])
        assert result.columns == ["id", "city", "salary"]

    def test_drop_columns_type_error_invalid_element(self, sample_frame):
        """Raises TypeError if cols contains non-string elements."""
        with pytest.raises(TypeError):
            sample_frame.drop_columns(["name", 123])

    def test_drop_columns_value_error_missing_column(self, sample_frame):
        """Raises ValueError if a column name doesn't exist."""
        with pytest.raises(ValueError):
            sample_frame.drop_columns(["nonexistent"])


class TestSelectColumns:
    """Tests for ArFrame.select_columns()"""

    def test_select_single_column(self, sample_frame):
        """Select a single column."""
        result = sample_frame.select_columns(["name"])
        assert result.columns == ["name"]

    def test_select_multiple_columns(self, sample_frame):
        """Select multiple columns."""
        result = sample_frame.select_columns(["id", "name", "age"])
        assert result.columns == ["id", "name", "age"]

    def test_select_columns_preserves_order(self, sample_frame):
        """Selected columns are returned in the specified order."""
        result = sample_frame.select_columns(["city", "id", "name"])
        assert result.columns == ["city", "id", "name"]

    def test_select_columns_value_error_empty(self, sample_frame):
        """Raises ValueError if selection is empty."""
        with pytest.raises(ValueError):
            sample_frame.select_columns([])

    def test_select_columns_value_error_duplicates(self, sample_frame):
        """Raises ValueError if columns list contains duplicates."""
        with pytest.raises(ValueError):
            sample_frame.select_columns(["name", "age", "name"])

    def test_select_columns_value_error_unknown(self, sample_frame):
        """Raises ValueError if an unknown column is specified."""
        with pytest.raises(ValueError):
            sample_frame.select_columns(["name", "nonexistent"])

    def test_select_columns_type_error_not_sequence(self, sample_frame):
        """Raises TypeError if columns is not a valid sequence of strings."""
        with pytest.raises(TypeError):
            sample_frame.select_columns("name")
