import pytest
import pandas as pd
import arnio as ar
import re


class TestSelectDtypes:
    """Test suite for ArFrame.select_dtypes()"""
    
    @pytest.fixture
    def frame(self):
        """Create a frame with int64, float64, string, and bool columns."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "score": [1.5, 2.5, 3.5],
            "name": ["alice", "bob", "charlie"],
            "active": [True, False, True],
        })
        return ar.from_pandas(df)
    
    # Include tests
    def test_include_single_dtype(self, frame):
        """Include a single dtype as string."""
        result = frame.select_dtypes(include="int64")
        assert result.columns == ["id"]
    
    def test_include_multiple_dtypes(self, frame):
        """Include multiple dtypes via list and tuple."""
        result_list = frame.select_dtypes(include=["int64", "float64"])
        assert result_list.columns == ["id", "score"]
        
        result_tuple = frame.select_dtypes(include=("string", "int64"))
        assert result_tuple.columns == ["id", "name"]
    
    # Exclude tests
    def test_exclude_dtypes(self, frame):
        """Exclude single or multiple dtypes."""
        result_single = frame.select_dtypes(exclude="string")
        assert result_single.columns == ["id", "score", "active"]
        
        result_multi = frame.select_dtypes(exclude=["string", "bool"])
        assert result_multi.columns == ["id", "score"]
    
    # Combined include/exclude
    def test_include_and_exclude_combined(self, frame):
        """Combine include and exclude filters (non-overlapping)."""
        result = frame.select_dtypes(
            include=["int64", "float64", "string"],
            exclude=["bool"]
        )
        assert result.columns == ["id", "score", "name"]
    
    # Error: No arguments
    def test_no_arguments_raises_valueerror(self, frame):
        """Raise ValueError when both include and exclude are None."""
        with pytest.raises(ValueError, match=re.escape(
            "select_dtypes() requires at least one of 'include' or 'exclude'."
        )):
            frame.select_dtypes()
    
    # Error: Non-string in list
    def test_non_string_in_list_raises_typeerror(self, frame):
        """Raise TypeError for non-string elements in include/exclude."""
        with pytest.raises(TypeError, match="'include' must contain only strings"):
            frame.select_dtypes(include=["int64", 1])
        
        with pytest.raises(TypeError, match="'exclude' must contain only strings"):
            frame.select_dtypes(exclude=[True])
    
    # Error: Wrong container type
    def test_wrong_container_type_raises_typeerror(self, frame):
        """Raise TypeError for wrong container types (set, bytes, dict)."""
        with pytest.raises(TypeError, match="'include' must be a string, list, or tuple"):
            frame.select_dtypes(include={"int64"})
        
        with pytest.raises(TypeError, match="'exclude' must be a string, list, or tuple"):
            frame.select_dtypes(exclude=b"int64")
    
    # Error: Unknown dtype
    def test_unknown_dtype_raises_valueerror(self, frame):
        """Raise ValueError for unrecognized dtype strings."""
        with pytest.raises(ValueError, match="Unrecognised dtype\\(s\\) in 'include'"):
            frame.select_dtypes(include="object")
        
        with pytest.raises(ValueError, match="Unrecognised dtype\\(s\\) in 'exclude'"):
            frame.select_dtypes(exclude="datetime64")
    
    # Error: Overlap
    def test_include_exclude_overlap_raises_valueerror(self, frame):
        """Raise ValueError when include and exclude sets overlap."""
        with pytest.raises(ValueError, match="'include' and 'exclude' overlap"):
            frame.select_dtypes(include=["int64", "string"], exclude=["string"])
    
    # Edge: Empty lists
    def test_empty_include_returns_no_columns(self, frame):
        """Empty include list matches no columns."""
        result = frame.select_dtypes(include=[])
        assert result.columns == []
    
    def test_empty_exclude_returns_all_columns(self, frame):
        """Empty exclude list excludes nothing."""
        result = frame.select_dtypes(exclude=[])
        assert result.columns == frame.columns
    
    # Edge: No matching dtype
    def test_nonexistent_dtype_returns_empty_frame(self, frame):
        """Including a dtype not present in frame returns empty result."""
        result = frame.select_dtypes(include="null")
        assert result.columns == []
    
    # Property: Order preservation
    def test_column_order_preserved_from_frame(self, frame):
        """Returned columns follow original frame order, not argument order."""
        result = frame.select_dtypes(include=["bool", "string", "int64"])
        assert result.columns == ["id", "name", "active"]
    
    # Property: Returns new object
    def test_returns_new_frame_object(self, frame):
        """Result is a new ArFrame instance with same row count."""
        result = frame.select_dtypes(include="int64")
        assert result is not frame
        assert isinstance(result, ar.ArFrame)
        assert result.shape[0] == frame.shape[0]
