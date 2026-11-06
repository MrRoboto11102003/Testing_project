import pytest
import arnio as ar


class TestDropColumns:
    """Tests for ArFrame.drop_columns method."""
    
    def test_drop_columns_with_list(self):
        """Drop columns using a list."""
        frame = ar.from_dict({
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "c": [7, 8, 9],
        })
        result = frame.drop_columns(["a"])
        assert result.columns == ["b", "c"]
        assert result.shape == (3, 2)
    
    def test_drop_columns_with_tuple(self):
        """Drop columns using a tuple."""
        frame = ar.from_dict({
            "a": [1, 2],
            "b": [3, 4],
            "c": [5, 6],
        })
        result = frame.drop_columns(("a", "c"))
        assert result.columns == ["b"]
    
    def test_drop_columns_with_duplicates(self):
        """Duplicates in drop list are silently ignored."""
        frame = ar.from_dict({
            "a": [1, 2],
            "b": [3, 4],
            "c": [5, 6],
        })
        result = frame.drop_columns(["a", "a", "b"])
        assert result.columns == ["c"]
    
    def test_drop_columns_nonexistent_raises_valueerror(self):
        """ValueError when column doesn't exist."""
        frame = ar.from_dict({"a": [1, 2], "b": [3, 4]})
        with pytest.raises(ValueError):
            frame.drop_columns(["nonexistent"])
    
    def test_drop_columns_non_list_raises_typeerror(self):
        """TypeError when cols is not list or tuple."""
        frame = ar.from_dict({"a": [1, 2]})
        with pytest.raises(TypeError):
            frame.drop_columns("a")
    
    def test_drop_columns_non_string_element_raises_typeerror(self):
        """TypeError when list contains non-string elements."""
        frame = ar.from_dict({"a": [1, 2], "b": [3, 4]})
        with pytest.raises(TypeError):
            frame.drop_columns(["a", 1])


class TestSelectColumns:
    """Tests for ArFrame.select_columns method."""
    
    def test_select_columns_basic(self):
        """Select specific columns preserving order."""
        frame = ar.from_dict({
            "a": [1, 2],
            "b": [3, 4],
            "c": [5, 6],
        })
        result = frame.select_columns(["a", "c"])
        assert result.columns == ["a", "c"]
        assert result.shape == (2, 2)
    
    def test_select_columns_reordered(self):
        """Select columns in different order than original."""
        frame = ar.from_dict({
            "a": [1, 2],
            "b": [3, 4],
            "c": [5, 6],
        })
        result = frame.select_columns(["c", "a", "b"])
        assert result.columns == ["c", "a", "b"]
    
    def test_select_columns_nonexistent_raises_valueerror(self):
        """ValueError when column doesn't exist."""
        frame = ar.from_dict({"a": [1, 2], "b": [3, 4]})
        with pytest.raises(ValueError):
            frame.select_columns(["nonexistent"])
    
    def test_select_columns_non_string_raises_typeerror(self):
        """TypeError when column list contains non-strings."""
        frame = ar.from_dict({"a": [1, 2], "b": [3, 4]})
        with pytest.raises(TypeError):
            frame.select_columns(["a", 1])


class TestSelectDtypes:
    """Tests for ArFrame.select_dtypes method."""
    
    def test_select_dtypes_include_single(self):
        """Include a single dtype as string."""
        frame = ar.from_dict({
            "int_col": [1, 2, 3],
            "float_col": [1.0, 2.0, 3.0],
            "str_col": ["a", "b", "c"],
        })
        result = frame.select_dtypes(include="int64")
        assert result.columns == ["int_col"]
    
    def test_select_dtypes_include_multiple(self):
        """Include multiple dtypes."""
        frame = ar.from_dict({
            "int_col": [1, 2, 3],
            "float_col": [1.0, 2.0, 3.0],
            "str_col": ["a", "b", "c"],
        })
        result = frame.select_dtypes(include=["int64", "float64"])
        assert set(result.columns) == {"int_col", "float_col"}
    
    def test_select_dtypes_exclude(self):
        """Exclude columns by dtype."""
        frame = ar.from_dict({
            "int_col": [1, 2, 3],
            "float_col": [1.0, 2.0, 3.0],
            "str_col": ["a", "b", "c"],
        })
        result = frame.select_dtypes(exclude="string")
        assert set(result.columns) == {"int_col", "float_col"}
    
    def test_select_dtypes_include_and_exclude(self):
        """Use both include and exclude together."""
        frame = ar.from_dict({
            "int_col": [1, 2, 3],
            "float_col": [1.0, 2.0, 3.0],
            "str_col": ["a", "b", "c"],
        })
        result = frame.select_dtypes(
            include=["int64", "float64", "string"],
            exclude="int64"
        )
        assert set(result.columns) == {"float_col", "str_col"}
    
    def test_select_dtypes_no_parameters_raises_valueerror(self):
        """ValueError when neither include nor exclude provided."""
        frame = ar.from_dict({"a": [1, 2]})
        with pytest.raises(ValueError):
            frame.select_dtypes()
    
    def test_select_dtypes_overlapping_raises_valueerror(self):
        """ValueError when include and exclude overlap."""
        frame = ar.from_dict({
            "int_col": [1, 2],
            "str_col": ["a", "b"],
        })
        with pytest.raises(ValueError):
            frame.select_dtypes(include="int64", exclude="int64")
    
    def test_select_dtypes_unknown_dtype_raises_valueerror(self):
        """ValueError for unknown dtype string."""
        frame = ar.from_dict({"a": [1, 2]})
        with pytest.raises(ValueError):
            frame.select_dtypes(include="unknown_type")
    
    def test_select_dtypes_no_match_raises_valueerror(self):
        """ValueError when no columns match the criteria."""
        frame = ar.from_dict({
            "int_col": [1, 2, 3],
            "str_col": ["a", "b", "c"],
        })
        with pytest.raises(ValueError):
            frame.select_dtypes(include="float64")
