import pytest
import pandas as pd
import arnio as ar


class TestArFrameDropColumns:
    """Tests for ArFrame.drop_columns()"""

    @pytest.fixture
    def frame(self):
        """Create a test frame with columns a, b, c"""
        return ar.from_pandas(
            pd.DataFrame({
                "a": [1, 2, 3],
                "b": ["x", None, "z"],
                "c": [1.5, 2.5, 3.5],
            })
        )

    def test_drop_single_column(self, frame):
        """Test dropping a single column"""
        result = frame.drop_columns(["b"])
        assert result.columns == ["a", "c"]
        assert result is not frame

    def test_drop_empty_list_returns_copy(self, frame):
        """Test that empty list returns unchanged copy"""
        result = frame.drop_columns([])
        assert result.columns == frame.columns
        assert result is not frame

    def test_drop_duplicates_silently_ignored(self, frame):
        """Test that duplicate column names are silently ignored"""
        result = frame.drop_columns(["b", "b"])
        assert result.columns == ["a", "c"]

    def test_drop_non_list_or_tuple_raises_typeerror(self, frame):
        """Test that non-list/tuple input raises TypeError"""
        with pytest.raises(TypeError, match="cols must be a list or tuple"):
            frame.drop_columns({"a"})

    def test_drop_non_string_element_raises_typeerror(self, frame):
        """Test that non-string column name raises TypeError"""
        with pytest.raises(TypeError, match="All column names in cols must be strings"):
            frame.drop_columns(["a", 1])

    def test_drop_unknown_column_raises_valueerror(self, frame):
        """Test that unknown column raises ValueError"""
        with pytest.raises(ValueError, match="Unknown column"):
            frame.drop_columns(["z"])

    def test_drop_all_columns_raises_valueerror(self, frame):
        """Test that dropping all columns raises ValueError"""
        with pytest.raises(ValueError, match="cannot remove all columns"):
            frame.drop_columns(["a", "b", "c"])

    def test_drop_attrs_deepcopy(self, frame):
        """Test that _attrs is deepcopied"""
        frame._attrs = {"meta": {"k": 1}}
        result = frame.drop_columns(["a"])
        result._attrs["meta"]["k"] = 2
        assert frame._attrs["meta"]["k"] == 1


class TestArFrameSelectColumns:
    """Tests for ArFrame.select_columns()"""

    @pytest.fixture
    def frame(self):
        """Create a test frame with columns a, b, c"""
        return ar.from_pandas(
            pd.DataFrame({
                "a": [1, 2, 3],
                "b": [1.5, 2.5, 3.5],
                "c": ["x", "y", "z"],
            })
        )

    def test_select_single_column(self, frame):
        """Test selecting a single column"""
        result = frame.select_columns(["b"])
        assert result.columns == ["b"]
        assert result is not frame

    def test_select_string_raises_typeerror(self, frame):
        """Test that string input raises TypeError (check 1)"""
        with pytest.raises(TypeError, match="must be a sequence"):
            frame.select_columns("a")

    def test_select_non_list_tuple_raises_typeerror(self, frame):
        """Test that non-list/tuple input raises TypeError (check 2)"""
        with pytest.raises(TypeError, match="must be a list or tuple"):
            frame.select_columns({"a"})

    def test_select_empty_raises_valueerror(self, frame):
        """Test that empty list raises ValueError (check 3)"""
        with pytest.raises(ValueError, match="cannot be empty"):
            frame.select_columns([])

    def test_select_non_string_raises_typeerror(self, frame):
        """Test that non-string element raises TypeError before duplicate check (check 4)"""
        with pytest.raises(TypeError, match="All column names must be strings"):
            frame.select_columns([1, 1])

    def test_select_duplicate_raises_valueerror(self, frame):
        """Test that duplicates raise ValueError before unknown check (check 5)"""
        with pytest.raises(ValueError, match="not allowed"):
            frame.select_columns(["z", "z"])

    def test_select_unknown_column_raises_valueerror(self, frame):
        """Test that unknown column raises ValueError (check 6)"""
        with pytest.raises(ValueError, match="Unknown columns"):
            frame.select_columns(["z"])
