import pytest
import re
import pandas as pd
import arnio as ar


class TestSelectDtypes:
    """Test suite for ArFrame.select_dtypes method."""
    
    @pytest.fixture
    def frame(self):
        """Create a test frame with int64, float64, string, and bool columns."""
        df = pd.DataFrame({
            "a": [1, 2],
            "b": [1.5, 2.5],
            "c": ["x", "y"],
            "d": [True, False]
        })
        return ar.from_pandas(df)
    
    def test_include_single_string(self, frame):
        """Test include with single dtype as string."""
        result = frame.select_dtypes(include="int64")
        assert result.columns == ["a"]
    
    def test_include_multiple_as_list(self, frame):
        """Test include with multiple dtypes as list."""
        result = frame.select_dtypes(include=["string", "int64"])
        assert result.columns == ["a", "c"]
    
    def test_include_multiple_as_tuple(self, frame):
        """Test include with multiple dtypes as tuple."""
        result = frame.select_dtypes(include=("float64", "int64"))
        assert result.columns == ["a", "b"]
    
    def test_exclude_single_string(self, frame):
        """Test exclude with single dtype as string."""
        result = frame.select_dtypes(exclude="string")
        assert result.columns == ["a", "b", "d"]
    
    def test_exclude_multiple_as_list(self, frame):
        """Test exclude with multiple dtypes as list."""
        result = frame.select_dtypes(exclude=["int64", "float64"])
        assert result.columns == ["c", "d"]
    
    def test_include_and_exclude_disjoint(self, frame):
        """Test include and exclude with disjoint sets."""
        result = frame.select_dtypes(
            include=["int64", "float64", "string"],
            exclude=["bool"]
        )
        assert result.columns == ["a", "b", "c"]
    
    def test_include_empty_list(self, frame):
        """Test include with empty list returns no columns (no-match path)."""
        result = frame.select_dtypes(include=[])
        assert result.columns == []
        assert result.shape[1] == 0
    
    def test_exclude_empty_list(self, frame):
        """Test exclude with empty list excludes nothing."""
        result = frame.select_dtypes(exclude=[])
        assert result.columns == ["a", "b", "c", "d"]
    
    def test_include_missing_dtype(self, frame):
        """Test include with dtype not in frame (no-match path)."""
        result = frame.select_dtypes(include="null")
        assert result.columns == []
        assert result.shape[1] == 0
    
    def test_neither_include_nor_exclude(self, frame):
        """Test that neither argument raises ValueError."""
        with pytest.raises(ValueError, match=re.escape("select_dtypes() requires at least one of 'include' or 'exclude'.")):
            frame.select_dtypes()
    
    def test_non_string_in_include_list(self, frame):
        """Test that non-string in include list raises TypeError."""
        with pytest.raises(TypeError, match="'include' must contain only strings"):
            frame.select_dtypes(include=["int64", 1])
    
    def test_wrong_container_type(self, frame):
        """Test that wrong container type for include raises TypeError."""
        with pytest.raises(TypeError, match=re.escape("'include' must be a string, list, or tuple of strings, got 'set'.")):
            frame.select_dtypes(include={"int64"})
    
    def test_unknown_dtype_in_include(self, frame):
        """Test that unknown dtype raises ValueError."""
        with pytest.raises(ValueError, match=re.escape("Unrecognised dtype(s) in 'include':")):
            frame.select_dtypes(include="object")
    
    def test_overlapping_include_exclude(self, frame):
        """Test that overlapping include and exclude raises ValueError."""
        with pytest.raises(ValueError, match=re.escape("'include' and 'exclude' overlap")):
            frame.select_dtypes(include=["int64"], exclude=["int64"])
    
    def test_attrs_behavior(self, frame):
        """Test attrs deep copy for match, shallow copy for no-match."""
        frame._attrs = {"k": {"nested": 1}}
        # Match case: deep copy
        result_match = frame.select_dtypes(include="int64")
        assert result_match._attrs == frame._attrs
        assert result_match._attrs["k"] is not frame._attrs["k"]
        # No-match case: shallow copy
        result_no_match = frame.select_dtypes(include="null")
        assert result_no_match._attrs is not frame._attrs
        assert result_no_match._attrs["k"] is frame._attrs["k"]
