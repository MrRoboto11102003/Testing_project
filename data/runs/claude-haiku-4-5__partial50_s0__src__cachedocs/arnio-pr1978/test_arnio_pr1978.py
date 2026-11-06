import pytest
import pandas as pd
import arnio as ar


class TestArFrameSelectDtypes:
    """Test suite for ArFrame.select_dtypes()"""

    @pytest.fixture
    def mixed_frame(self):
        """Create a test frame with various supported dtypes"""
        data = {
            "id": [1, 2, 3],
            "score": [95.5, 87.3, 92.1],
            "name": ["Alice", "Bob", "Charlie"],
            "is_active": [True, False, True],
        }
        df = pd.DataFrame(data)
        return ar.from_pandas(df)

    @pytest.fixture
    def numeric_only_frame(self):
        """Create a test frame with only numeric columns"""
        data = {
            "int_col": [1, 2, 3],
            "float_col": [1.1, 2.2, 3.3],
        }
        df = pd.DataFrame(data)
        return ar.from_pandas(df)

    def test_include_single_dtype_string(self, mixed_frame):
        """Test include with single dtype as string"""
        result = mixed_frame.select_dtypes(include="int64")
        assert result.columns == ["id"]
        assert result.shape[0] == mixed_frame.shape[0]

    def test_include_multiple_dtypes_list(self, mixed_frame):
        """Test include with multiple dtypes as list"""
        result = mixed_frame.select_dtypes(include=["int64", "float64"])
        assert set(result.columns) == {"id", "score"}

    def test_include_multiple_dtypes_tuple(self, mixed_frame):
        """Test include with multiple dtypes as tuple"""
        result = mixed_frame.select_dtypes(include=("int64", "float64"))
        assert set(result.columns) == {"id", "score"}

    def test_exclude_single_dtype(self, mixed_frame):
        """Test exclude with single dtype"""
        result = mixed_frame.select_dtypes(exclude="string")
        assert "name" not in result.columns
        assert set(result.columns) == {"id", "score", "is_active"}

    def test_exclude_multiple_dtypes(self, mixed_frame):
        """Test exclude with multiple dtypes"""
        result = mixed_frame.select_dtypes(exclude=["string", "bool"])
        assert set(result.columns) == {"id", "score"}

    def test_column_order_preserved(self, mixed_frame):
        """Test that original column order is preserved in result"""
        result = mixed_frame.select_dtypes(include=["bool", "float64", "int64"])
        # Should preserve original frame's column order, not filter order
        assert result.columns == ["id", "score", "is_active"]

    def test_no_args_raises_valueerror(self, mixed_frame):
        """Test ValueError when neither include nor exclude provided"""
        with pytest.raises(ValueError, match="requires at least one"):
            mixed_frame.select_dtypes()

    def test_overlapping_include_exclude_raises_valueerror(self, mixed_frame):
        """Test ValueError when include and exclude have overlapping dtypes"""
        with pytest.raises(ValueError, match="overlap"):
            mixed_frame.select_dtypes(include=["int64", "float64"], exclude="int64")

    def test_invalid_dtype_raises_valueerror(self, mixed_frame):
        """Test ValueError for unrecognized dtype"""
        with pytest.raises(ValueError, match="Unrecognised dtype"):
            mixed_frame.select_dtypes(include="not_a_dtype")

    def test_no_matching_columns(self, numeric_only_frame):
        """Test empty frame returned when no columns match filter"""
        result = numeric_only_frame.select_dtypes(include="string")
        assert result.shape[1] == 0

    def test_all_columns_match(self, mixed_frame):
        """Test when all columns match the filter"""
        result = mixed_frame.select_dtypes(
            include=["int64", "float64", "string", "bool"]
        )
        assert result.shape == mixed_frame.shape
        assert len(result.columns) == len(mixed_frame.columns)

    def test_wrong_type_for_include(self, mixed_frame):
        """Test TypeError when include is not string/list/tuple"""
        with pytest.raises(TypeError):
            mixed_frame.select_dtypes(include=123)

    def test_non_string_in_include_list(self, mixed_frame):
        """Test TypeError when include list contains non-string element"""
        with pytest.raises(TypeError, match="must contain only strings"):
            mixed_frame.select_dtypes(include=["int64", 123])

    def test_wrong_type_for_exclude(self, mixed_frame):
        """Test TypeError when exclude is not string/list/tuple"""
        with pytest.raises(TypeError):
            mixed_frame.select_dtypes(exclude=False)
