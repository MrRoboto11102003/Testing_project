import pytest
import arnio as ar


class TestArFrameColumnSelection:
    """Test ArFrame.drop_columns, select_columns, and select_dtypes methods."""

    @pytest.fixture
    def mixed_frame(self):
        """Frame with multiple data types for testing."""
        return ar.from_dict({
            "int_col": [1, 2, 3],
            "float_col": [1.5, 2.5, 3.5],
            "str_col": ["a", "b", "c"],
            "bool_col": [True, False, True],
        })

    # drop_columns tests
    def test_drop_columns_single(self, mixed_frame):
        """Drop single column."""
        result = mixed_frame.drop_columns(["int_col"])
        assert result.columns == ["float_col", "str_col", "bool_col"]

    def test_drop_columns_multiple(self, mixed_frame):
        """Drop multiple columns."""
        result = mixed_frame.drop_columns(["int_col", "str_col"])
        assert result.columns == ["float_col", "bool_col"]

    def test_drop_columns_with_tuple(self, mixed_frame):
        """Accept tuple as well as list."""
        result = mixed_frame.drop_columns(("int_col", "bool_col"))
        assert result.columns == ["float_col", "str_col"]

    def test_drop_columns_empty_list(self, mixed_frame):
        """Empty list returns copy unchanged."""
        result = mixed_frame.drop_columns([])
        assert result.columns == mixed_frame.columns

    def test_drop_columns_nonexistent_raises(self, mixed_frame):
        """Raise ValueError for unknown column name."""
        with pytest.raises(ValueError, match="Unknown column"):
            mixed_frame.drop_columns(["nonexistent"])

    def test_drop_columns_all_raises(self, mixed_frame):
        """Raise ValueError when dropping all columns."""
        with pytest.raises(ValueError, match="cannot remove all columns"):
            mixed_frame.drop_columns(["int_col", "float_col", "str_col", "bool_col"])

    def test_drop_columns_invalid_type_raises(self, mixed_frame):
        """Raise TypeError if not list or tuple."""
        with pytest.raises(TypeError, match="cols must be a list or tuple"):
            mixed_frame.drop_columns("int_col")  # type: ignore

    # select_columns tests
    def test_select_columns_single(self, mixed_frame):
        """Select single column."""
        result = mixed_frame.select_columns(["str_col"])
        assert result.columns == ["str_col"]
        assert result.shape == (3, 1)

    def test_select_columns_multiple_preserves_order(self, mixed_frame):
        """Select multiple columns in specified order."""
        result = mixed_frame.select_columns(["bool_col", "int_col"])
        assert result.columns == ["bool_col", "int_col"]

    def test_select_columns_empty_raises(self, mixed_frame):
        """Raise ValueError for empty column selection."""
        with pytest.raises(ValueError, match="cannot be empty"):
            mixed_frame.select_columns([])

    def test_select_columns_duplicates_raises(self, mixed_frame):
        """Raise ValueError for duplicate column names."""
        with pytest.raises(ValueError, match="Duplicate"):
            mixed_frame.select_columns(["int_col", "int_col"])

    def test_select_columns_nonexistent_raises(self, mixed_frame):
        """Raise ValueError for unknown column."""
        with pytest.raises(ValueError, match="Unknown columns"):
            mixed_frame.select_columns(["missing"])

    def test_select_columns_string_raises(self, mixed_frame):
        """Raise TypeError when columns argument is a string."""
        with pytest.raises(TypeError, match="must be a sequence"):
            mixed_frame.select_columns("int_col")  # type: ignore

    # select_dtypes tests
    def test_select_dtypes_include_single(self, mixed_frame):
        """Include single dtype as string."""
        result = mixed_frame.select_dtypes(include="int64")
        assert "int_col" in result.columns
        assert "float_col" not in result.columns

    def test_select_dtypes_include_multiple(self, mixed_frame):
        """Include multiple dtypes as list."""
        result = mixed_frame.select_dtypes(include=["int64", "float64"])
        assert set(result.columns) == {"int_col", "float_col"}

    def test_select_dtypes_exclude(self, mixed_frame):
        """Exclude dtypes."""
        result = mixed_frame.select_dtypes(exclude=["string", "bool"])
        assert set(result.columns) == {"int_col", "float_col"}

    def test_select_dtypes_include_and_exclude(self, mixed_frame):
        """Use both include and exclude together."""
        result = mixed_frame.select_dtypes(
            include=["int64", "float64", "string"],
            exclude=["int64"]
        )
        assert set(result.columns) == {"float_col", "str_col"}

    def test_select_dtypes_neither_raises(self, mixed_frame):
        """Raise ValueError when neither include nor exclude provided."""
        with pytest.raises(ValueError, match="at least one"):
            mixed_frame.select_dtypes()

    def test_select_dtypes_overlap_raises(self, mixed_frame):
        """Raise ValueError when include and exclude overlap."""
        with pytest.raises(ValueError, match="overlap"):
            mixed_frame.select_dtypes(include="int64", exclude="int64")

    def test_select_dtypes_invalid_dtype_raises(self, mixed_frame):
        """Raise ValueError for unrecognized dtype."""
        with pytest.raises(ValueError, match="Unrecognised dtype"):
            mixed_frame.select_dtypes(include="invalid_type")

    def test_select_dtypes_invalid_type_raises(self, mixed_frame):
        """Raise TypeError when include/exclude is wrong type."""
        with pytest.raises(TypeError, match="must be a string, list, or tuple"):
            mixed_frame.select_dtypes(include=123)  # type: ignore
