import pytest
import arnio as ar
import pandas as pd


class TestDropColumns:
    """Tests for ArFrame.drop_columns()"""

    @pytest.fixture
    def frame(self):
        """Create a sample ArFrame for testing"""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
            "salary": [50000, 60000, 70000],
        })
        return ar.from_pandas(df)

    def test_drop_single_column(self, frame):
        """Test dropping a single column"""
        result = frame.drop_columns(["name"])
        assert "name" not in result.columns
        assert set(result.columns) == {"id", "age", "salary"}
        assert result.shape == (3, 3)

    def test_drop_multiple_columns(self, frame):
        """Test dropping multiple columns"""
        result = frame.drop_columns(["name", "age"])
        assert set(result.columns) == {"id", "salary"}
        assert result.shape == (3, 2)

    def test_drop_columns_accepts_tuple(self, frame):
        """Test that drop_columns accepts tuple input"""
        result = frame.drop_columns(("name", "age"))
        assert set(result.columns) == {"id", "salary"}

    def test_drop_columns_preserves_order(self, frame):
        """Test that original column order is preserved"""
        result = frame.drop_columns(["age"])
        assert result.columns == ["id", "name", "salary"]

    def test_drop_columns_empty_list(self, frame):
        """Test that empty list returns copy with all columns unchanged"""
        result = frame.drop_columns([])
        assert result.columns == frame.columns
        assert result.shape == frame.shape
        assert result is not frame

    def test_drop_columns_silently_ignores_duplicates(self, frame):
        """Test that duplicates in cols are silently ignored"""
        result = frame.drop_columns(["name", "name", "age", "age"])
        assert set(result.columns) == {"id", "salary"}

    def test_drop_columns_nonexistent_raises_value_error(self, frame):
        """Test ValueError when dropping non-existent column"""
        with pytest.raises(ValueError):
            frame.drop_columns(["nonexistent"])

    def test_drop_columns_mixed_valid_invalid_raises_value_error(self, frame):
        """Test ValueError when some columns don't exist"""
        with pytest.raises(ValueError):
            frame.drop_columns(["name", "nonexistent"])

    def test_drop_columns_non_list_raises_type_error(self, frame):
        """Test TypeError when cols is not list or tuple"""
        with pytest.raises(TypeError):
            frame.drop_columns("name")
        with pytest.raises(TypeError):
            frame.drop_columns(123)

    def test_drop_columns_non_string_elements_raises_type_error(self, frame):
        """Test TypeError when cols contains non-string elements"""
        with pytest.raises(TypeError):
            frame.drop_columns([123, "name"])
        with pytest.raises(TypeError):
            frame.drop_columns(["name", None])


class TestSelectColumns:
    """Tests for ArFrame.select_columns()"""

    @pytest.fixture
    def frame(self):
        """Create a sample ArFrame for testing"""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
            "salary": [50000, 60000, 70000],
        })
        return ar.from_pandas(df)

    def test_select_single_column(self, frame):
        """Test selecting a single column"""
        result = frame.select_columns(["name"])
        assert result.columns == ["name"]
        assert result.shape == (3, 1)

    def test_select_multiple_columns(self, frame):
        """Test selecting multiple columns"""
        result = frame.select_columns(["id", "name"])
        assert result.columns == ["id", "name"]
        assert result.shape == (3, 2)

    def test_select_columns_preserves_requested_order(self, frame):
        """Test that requested column order is preserved"""
        result = frame.select_columns(["salary", "id", "name"])
        assert result.columns == ["salary", "id", "name"]

    def test_select_columns_returns_new_frame(self, frame):
        """Test that select_columns returns a new ArFrame"""
        result = frame.select_columns(["name"])
        assert result is not frame
        assert isinstance(result, ar.ArFrame)

    def test_select_columns_nonexistent_raises_value_error(self, frame):
        """Test ValueError when selecting non-existent column"""
        with pytest.raises(ValueError):
            frame.select_columns(["nonexistent"])

    def test_select_columns_mixed_valid_invalid_raises_value_error(self, frame):
        """Test ValueError when some columns don't exist"""
        with pytest.raises(ValueError):
            frame.select_columns(["name", "nonexistent"])

    def test_select_columns_non_string_elements_raises_type_error(self, frame):
        """Test TypeError when columns contains non-string elements"""
        with pytest.raises(TypeError):
            frame.select_columns([123, "name"])
