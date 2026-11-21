import pytest
import re
import pandas as pd
import arnio as ar


class TestArFrameDropColumns:
    """Test suite for ArFrame.drop_columns()"""
    
    @pytest.fixture
    def frame(self):
        """Create a test ArFrame with columns a, b, c and mixed types"""
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": ["x", None, "z"],
            "c": [1.5, 2.5, 3.5]
        })
        return ar.from_pandas(df)
    
    # --- Type validation for cols parameter ---
    
    def test_cols_string_raises_typeerror(self, frame):
        """Passing a string instead of list should raise TypeError"""
        with pytest.raises(TypeError, match=re.escape("cols must be a list or tuple of column names, got 'str'")):
            frame.drop_columns("a")
    
    def test_cols_set_raises_typeerror(self, frame):
        """Passing a set should raise TypeError"""
        with pytest.raises(TypeError, match=re.escape("cols must be a list or tuple of column names, got 'set'")):
            frame.drop_columns({"a"})
    
    def test_cols_none_raises_typeerror(self, frame):
        """Passing None should raise TypeError"""
        with pytest.raises(TypeError, match=re.escape("cols must be a list or tuple of column names, got 'NoneType'")):
            frame.drop_columns(None)
    
    # --- Element type validation ---
    
    def test_non_string_element_raises_typeerror(self, frame):
        """Non-string element in list should raise TypeError"""
        with pytest.raises(TypeError, match="All column names in cols must be strings"):
            frame.drop_columns(["a", 1])
    
    def test_type_check_before_existence_check(self, frame):
        """Type check runs before existence check: [unknown_col, int] raises TypeError"""
        with pytest.raises(TypeError, match="All column names in cols must be strings"):
            frame.drop_columns(["zz", 1])
    
    # --- Unknown columns and all-drop validation ---
    
    def test_unknown_column_raises_valueerror(self, frame):
        """Unknown column name should raise ValueError"""
        with pytest.raises(ValueError, match=re.escape("Unknown column(s): ['z']")):
            frame.drop_columns(["z"])
    
    def test_drop_all_columns_raises_valueerror(self, frame):
        """Trying to drop all columns should raise ValueError"""
        with pytest.raises(ValueError, match="drop_columns cannot remove all columns from the frame"):
            frame.drop_columns(["a", "b", "c"])
    
    # --- Normal operation: basic drops ---
    
    def test_drop_single_column(self, frame):
        """Drop a single column successfully"""
        result = frame.drop_columns(["b"])
        assert result.columns == ["a", "c"]
        assert result.shape[0] == 3
        assert result.shape[1] == 2
    
    def test_drop_multiple_preserves_order(self, frame):
        """Drop multiple columns, keeping original order of remaining columns"""
        result = frame.drop_columns(["c", "a"])  # Order of input doesn't matter
        assert result.columns == ["b"]
    
    def test_drop_with_tuple_input(self, frame):
        """Accept tuple as well as list"""
        result = frame.drop_columns(("a", "c"))
        assert result.columns == ["b"]
    
    # --- Duplicates and empty input ---
    
    def test_duplicates_silently_ignored(self, frame):
        """Duplicate column names should be silently ignored"""
        result = frame.drop_columns(["b", "b", "b"])
        assert result.columns == ["a", "c"]
    
    def test_empty_list_returns_unchanged_copy(self, frame):
        """Empty list should return unchanged copy (different object)"""
        result = frame.drop_columns([])
        assert result.columns == frame.columns
        assert result is not frame
        assert result.shape == frame.shape
    
    # --- Data and state preservation ---
    
    def test_data_and_nulls_preserved(self, frame):
        """Values and nulls should be preserved in kept columns"""
        result = frame.drop_columns(["a"])
        # Check b column still has None
        b_vals = result["b"]
        assert None in b_vals
        # Check c values preserved
        assert result["c"] == frame["c"]
    
    def test_original_frame_unchanged(self, frame):
        """Original frame should not be modified"""
        original_cols = frame.columns[:]
        original_shape = frame.shape
        frame.drop_columns(["a"])
        assert frame.columns == original_cols
        assert frame.shape == original_shape
    
    def test_attrs_deepcopied(self, frame):
        """Attributes should be deepcopied, not shared"""
        frame_with_attrs = ar.ArFrame(
            frame._frame,
            attrs={"meta": {"key": 1}}
        )
        result = frame_with_attrs.drop_columns(["b"])
        # Attrs are equal but different objects
        assert result._attrs == frame_with_attrs._attrs
        assert result._attrs is not frame_with_attrs._attrs
        # Nested dict also copied
        assert result._attrs["meta"] is not frame_with_attrs._attrs["meta"]
        # Mutation of result doesn't affect original
        result._attrs["meta"]["key"] = 999
        assert frame_with_attrs._attrs["meta"]["key"] == 1
