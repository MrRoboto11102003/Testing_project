import pytest
import pandas as pd
import arnio as ar


@pytest.fixture
def sample_frame():
    """Create a test frame with mixed dtypes for testing."""
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "score": [95.5, 87.3, 92.1],
        "active": [True, False, True],
    })
    return ar.from_pandas(df)


class TestDropColumns:
    """Tests for ArFrame.drop_columns()"""

    def test_drop_single_and_multiple_columns(self, sample_frame):
        """Test dropping single and multiple columns."""
        result = sample_frame.drop_columns(["id"])
        assert result.columns == ["name", "score", "active"]

        result = sample_frame.drop_columns(["id", "score"])
        assert result.columns == ["name", "active"]

    def test_drop_empty_list_and_duplicates(self, sample_frame):
        """Test that empty list returns unchanged copy and duplicates are ignored."""
        result = sample_frame.drop_columns([])
        assert result.columns == sample_frame.columns

        result = sample_frame.drop_columns(["id", "id", "name"])
        assert result.columns == ["score", "active"]

    def test_drop_with_tuple_input(self, sample_frame):
        """Test that tuple input is accepted."""
        result = sample_frame.drop_columns(("id", "name"))
        assert result.columns == ["score", "active"]

    def test_drop_errors(self, sample_frame):
        """Test error cases: invalid type, non-existent column, all columns."""
        with pytest.raises(TypeError, match="must be a list or tuple"):
            sample_frame.drop_columns("id")

        with pytest.raises(TypeError, match="must be strings"):
            sample_frame.drop_columns(["id", 123])

        with pytest.raises(ValueError, match="Unknown column"):
            sample_frame.drop_columns(["nonexistent"])

        with pytest.raises(ValueError, match="cannot remove all"):
            sample_frame.drop_columns(["id", "name", "score", "active"])


class TestSelectColumns:
    """Tests for ArFrame.select_columns()"""

    def test_select_single_and_multiple_columns(self, sample_frame):
        """Test selecting single and multiple columns."""
        result = sample_frame.select_columns(["id"])
        assert result.columns == ["id"]

        result = sample_frame.select_columns(["name", "score"])
        assert result.columns == ["name", "score"]

    def test_select_preserves_order(self, sample_frame):
        """Test that column order is preserved as requested."""
        result = sample_frame.select_columns(["active", "id", "name"])
        assert result.columns == ["active", "id", "name"]

    def test_select_errors(self, sample_frame):
        """Test error cases: empty list, duplicates, unknown columns, invalid types."""
        with pytest.raises(ValueError, match="empty"):
            sample_frame.select_columns([])

        with pytest.raises(ValueError, match="Duplicate"):
            sample_frame.select_columns(["id", "id"])

        with pytest.raises(ValueError, match="Unknown"):
            sample_frame.select_columns(["nonexistent"])

        with pytest.raises(TypeError, match="not a string"):
            sample_frame.select_columns("id")

        with pytest.raises(TypeError, match="list or tuple"):
            sample_frame.select_columns({"id": 1})


class TestSelectDtypes:
    """Tests for ArFrame.select_dtypes()"""

    def test_include_and_exclude_dtypes(self, sample_frame):
        """Test include/exclude with string and list inputs."""
        result = sample_frame.select_dtypes(include="int64")
        assert result.columns == ["id"]

        result = sample_frame.select_dtypes(include=["int64", "float64"])
        assert set(result.columns) == {"id", "score"}

        result = sample_frame.select_dtypes(exclude="string")
        assert "name" not in result.columns

    def test_select_dtypes_with_tuple(self, sample_frame):
        """Test that tuple input is accepted."""
        result = sample_frame.select_dtypes(include=("int64", "bool"))
        assert set(result.columns) == {"id", "active"}

    def test_select_dtypes_no_match(self, sample_frame):
        """Test that no matching dtypes returns empty column list."""
        result = sample_frame.select_dtypes(include="null")
        assert len(result.columns) == 0

    def test_select_dtypes_errors(self, sample_frame):
        """Test error cases: neither include/exclude, overlap, invalid dtype/type."""
        with pytest.raises(ValueError, match="at least one"):
            sample_frame.select_dtypes()

        with pytest.raises(ValueError, match="overlap"):
            sample_frame.select_dtypes(
                include=["int64", "float64"],
                exclude="int64"
            )

        with pytest.raises(ValueError, match="Unrecognised dtype"):
            sample_frame.select_dtypes(include="invalid")

        with pytest.raises(TypeError, match="must be a string"):
            sample_frame.select_dtypes(include=123)
