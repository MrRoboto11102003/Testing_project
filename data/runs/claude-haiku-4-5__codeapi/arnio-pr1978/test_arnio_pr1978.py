import pytest
import pandas as pd
import arnio as ar


class TestArFrameSelectDtypes:
    """Tests for ArFrame.select_dtypes method."""
    
    @pytest.fixture
    def mixed_frame(self):
        """Create an ArFrame with multiple dtypes."""
        data = {
            "int1": [1, 2, 3],
            "int2": [10, 20, 30],
            "float1": [1.5, 2.5, 3.5],
            "float2": [0.1, 0.2, 0.3],
            "str1": ["a", "b", "c"],
            "str2": ["x", "y", "z"],
            "bool1": [True, False, True],
        }
        df = pd.DataFrame(data)
        return ar.from_pandas(df)
    
    def test_include_single_dtype_string(self, mixed_frame):
        """Test include with a single dtype as string."""
        result = mixed_frame.select_dtypes(include="int64")
        assert set(result.columns) == {"int1", "int2"}
        assert result.shape[0] == mixed_frame.shape[0]
    
    def test_include_multiple_dtypes_list(self, mixed_frame):
        """Test include with multiple dtypes as list."""
        result = mixed_frame.select_dtypes(include=["int64", "float64"])
        assert set(result.columns) == {"int1", "int2", "float1", "float2"}
    
    def test_include_multiple_dtypes_tuple(self, mixed_frame):
        """Test include with multiple dtypes as tuple."""
        result = mixed_frame.select_dtypes(include=("string", "bool"))
        assert set(result.columns) == {"str1", "str2", "bool1"}
    
    def test_exclude_single_dtype(self, mixed_frame):
        """Test exclude with a single dtype."""
        result = mixed_frame.select_dtypes(exclude="string")
        assert "str1" not in result.columns
        assert "str2" not in result.columns
        assert set(result.columns) == {"int1", "int2", "float1", "float2", "bool1"}
    
    def test_exclude_multiple_dtypes(self, mixed_frame):
        """Test exclude with multiple dtypes."""
        result = mixed_frame.select_dtypes(exclude=["int64", "bool"])
        assert set(result.columns) == {"float1", "float2", "str1", "str2"}
    
    def test_include_and_exclude_non_overlapping(self, mixed_frame):
        """Test with both include and exclude that don't overlap."""
        result = mixed_frame.select_dtypes(include=["int64", "float64"], exclude="float64")
        assert set(result.columns) == {"int1", "int2"}
    
    def test_preserve_column_order(self, mixed_frame):
        """Test that original column order is preserved."""
        result = mixed_frame.select_dtypes(include=["int64", "string", "float64"])
        expected = ["int1", "int2", "float1", "float2", "str1", "str2"]
        assert result.columns == expected
    
    def test_error_neither_include_nor_exclude(self, mixed_frame):
        """Test ValueError when neither include nor exclude provided."""
        with pytest.raises(ValueError, match="requires at least one"):
            mixed_frame.select_dtypes()
    
    def test_error_overlapping_include_exclude(self, mixed_frame):
        """Test ValueError when include and exclude overlap."""
        with pytest.raises(ValueError, match="overlap"):
            mixed_frame.select_dtypes(include="int64", exclude="int64")
    
    def test_error_overlapping_multiple_dtypes(self, mixed_frame):
        """Test ValueError when multiple dtypes overlap between include and exclude."""
        with pytest.raises(ValueError, match="overlap"):
            mixed_frame.select_dtypes(
                include=["int64", "float64"],
                exclude=["float64", "string"]
            )
    
    def test_error_invalid_dtype_in_include(self, mixed_frame):
        """Test ValueError for unrecognised dtype in include."""
        with pytest.raises(ValueError, match="Unrecognised dtype"):
            mixed_frame.select_dtypes(include="invalid_dtype")
    
    def test_error_invalid_dtype_in_exclude(self, mixed_frame):
        """Test ValueError for unrecognised dtype in exclude."""
        with pytest.raises(ValueError, match="Unrecognised dtype"):
            mixed_frame.select_dtypes(exclude=["string", "unknown"])
    
    def test_error_invalid_type_for_include(self, mixed_frame):
        """Test TypeError when include is not string, list, or tuple."""
        with pytest.raises(TypeError, match="must be a string, list, or tuple"):
            mixed_frame.select_dtypes(include=123)
    
    def test_error_invalid_type_for_exclude(self, mixed_frame):
        """Test TypeError when exclude is not string, list, or tuple."""
        with pytest.raises(TypeError, match="must be a string, list, or tuple"):
            mixed_frame.select_dtypes(exclude={"dtype": "int64"})
    
    def test_error_non_string_in_include_list(self, mixed_frame):
        """Test TypeError when include list contains non-strings."""
        with pytest.raises(TypeError, match="must contain only strings"):
            mixed_frame.select_dtypes(include=["int64", 123])
    
    def test_error_non_string_in_exclude_list(self, mixed_frame):
        """Test TypeError when exclude list contains non-strings."""
        with pytest.raises(TypeError, match="must contain only strings"):
            mixed_frame.select_dtypes(exclude=["string", None, "int64"])
    
    def test_empty_result_no_matching_columns(self, mixed_frame):
        """Test when no columns match the filter."""
        result = mixed_frame.select_dtypes(include="null")
        assert result.shape[1] == 0
        assert result.shape[0] == mixed_frame.shape[0]
    
    def test_all_columns_match(self, mixed_frame):
        """Test when all columns match the filter."""
        result = mixed_frame.select_dtypes(
            include=["int64", "float64", "string", "bool"]
        )
        assert result.shape == mixed_frame.shape
        assert result.columns == mixed_frame.columns
    
    def test_single_column_selected(self, mixed_frame):
        """Test selecting a single column."""
        result = mixed_frame.select_dtypes(include="bool")
        assert result.columns == ["bool1"]
        assert result.shape == (3, 1)
    
    def test_exclude_all_dtypes_except_one(self, mixed_frame):
        """Test exclude pattern that leaves only one dtype."""
        result = mixed_frame.select_dtypes(
            exclude=["int64", "float64", "string"]
        )
        assert set(result.columns) == {"bool1"}
