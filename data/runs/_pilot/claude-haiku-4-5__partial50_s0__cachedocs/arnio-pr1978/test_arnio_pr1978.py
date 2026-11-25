import pytest
import pandas as pd
import arnio as ar


class TestSelectDtypes:
    """Comprehensive tests for ArFrame.select_dtypes method."""

    @pytest.fixture
    def mixed_frame(self):
        """Create an ArFrame with multiple data types."""
        df = pd.DataFrame({
            "int_col": [1, 2, 3],
            "float_col": [1.5, 2.5, 3.5],
            "string_col": ["alice", "bob", "charlie"],
            "bool_col": [True, False, True],
        })
        return ar.from_pandas(df)

    @pytest.fixture
    def numeric_only_frame(self):
        """Create an ArFrame with only numeric columns."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "score": [10.5, 20.5, 30.5],
            "count": [100, 200, 300],
        })
        return ar.from_pandas(df)

    def test_include_single_dtype_as_string(self, mixed_frame):
        """Test include with a single dtype as string."""
        result = mixed_frame.select_dtypes(include="int64")
        assert result.shape == (3, 1)
        assert result.columns == ["int_col"]
        assert isinstance(result, ar.ArFrame)

    def test_include_multiple_dtypes_as_list(self, mixed_frame):
        """Test include with multiple dtypes as list."""
        result = mixed_frame.select_dtypes(include=["int64", "float64"])
        assert result.shape == (3, 2)
        assert set(result.columns) == {"int_col", "float_col"}

    def test_include_multiple_dtypes_as_tuple(self, mixed_frame):
        """Test include with multiple dtypes as tuple."""
        result = mixed_frame.select_dtypes(include=("string", "bool"))
        assert result.shape == (3, 2)
        assert set(result.columns) == {"string_col", "bool_col"}

    def test_exclude_single_dtype_as_string(self, mixed_frame):
        """Test exclude with a single dtype as string."""
        result = mixed_frame.select_dtypes(exclude="string")
        assert result.shape == (3, 3)
        assert "string_col" not in result.columns
        assert set(result.columns) == {"int_col", "float_col", "bool_col"}

    def test_exclude_multiple_dtypes_as_list(self, mixed_frame):
        """Test exclude with multiple dtypes as list."""
        result = mixed_frame.select_dtypes(exclude=["int64", "float64"])
        assert result.shape == (3, 2)
        assert set(result.columns) == {"string_col", "bool_col"}

    def test_exclude_multiple_dtypes_as_tuple(self, mixed_frame):
        """Test exclude with multiple dtypes as tuple."""
        result = mixed_frame.select_dtypes(exclude=("int64", "bool"))
        assert result.shape == (3, 2)
        assert set(result.columns) == {"float_col", "string_col"}

    def test_neither_include_nor_exclude_raises_valueerror(self, mixed_frame):
        """Test that calling without include or exclude raises ValueError."""
        with pytest.raises(ValueError):
            mixed_frame.select_dtypes()

    def test_overlapping_include_and_exclude_raises_valueerror(self, mixed_frame):
        """Test that overlapping include and exclude raises ValueError."""
        with pytest.raises(ValueError):
            mixed_frame.select_dtypes(include=["int64", "float64"], exclude="int64")

    def test_unknown_dtype_raises_valueerror(self, mixed_frame):
        """Test that unknown dtype strings raise ValueError."""
        with pytest.raises(ValueError):
            mixed_frame.select_dtypes(include="unknown_dtype")

    def test_no_columns_match_raises_valueerror(self, numeric_only_frame):
        """Test that when no columns match, ValueError is raised."""
        with pytest.raises(ValueError):
            numeric_only_frame.select_dtypes(include="string")

    def test_exclude_all_columns_raises_valueerror(self, mixed_frame):
        """Test that excluding all columns raises ValueError."""
        with pytest.raises(ValueError):
            mixed_frame.select_dtypes(exclude=["int64", "float64", "string", "bool"])

    def test_column_order_preserved(self, mixed_frame):
        """Test that column order in result matches original frame."""
        result = mixed_frame.select_dtypes(include=["bool", "int64", "float64"])
        # Columns should appear in original frame order
        assert list(result.columns) == ["int_col", "float_col", "bool_col"]

    def test_include_and_exclude_non_overlapping(self, mixed_frame):
        """Test that include and exclude work together if non-overlapping."""
        result = mixed_frame.select_dtypes(
            include=["int64", "float64"],
            exclude="float64"
        )
        assert result.shape == (3, 1)
        assert result.columns == ["int_col"]

    def test_returns_new_arframe_instance(self, mixed_frame):
        """Test that result is a new ArFrame instance, not the original."""
        result = mixed_frame.select_dtypes(include="string")
        assert isinstance(result, ar.ArFrame)
        assert result is not mixed_frame
