"""
Pytest tests for arnio.frame.ArFrame column selection methods.
Tests drop_columns, select_columns, and select_dtypes.
"""

import re
import copy

import pytest
import pandas as pd
import numpy as np

import arnio as ar


@pytest.fixture
def sample_frame():
    """Create a test frame with mixed dtypes."""
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [1.5, 2.5, 3.5],
        "c": ["x", "y", "z"],
        "d": [True, False, True],
    })
    return ar.from_pandas(df)


class TestDropColumns:
    """Tests for ArFrame.drop_columns()"""

    def test_drop_columns_basic(self, sample_frame):
        """Test basic column dropping."""
        result = sample_frame.drop_columns(["a"])
        assert result.columns == ["b", "c", "d"]
        assert result.shape == (3, 3)
        assert sample_frame.columns == ["a", "b", "c", "d"]  # unchanged

    def test_drop_columns_multiple(self, sample_frame):
        """Test dropping multiple columns."""
        result = sample_frame.drop_columns(["a", "c"])
        assert result.columns == ["b", "d"]
        assert result.shape == (3, 2)

    def test_drop_columns_tuple_input(self, sample_frame):
        """Test tuple input works like list."""
        result = sample_frame.drop_columns(("b", "d"))
        assert result.columns == ["a", "c"]

    def test_drop_columns_empty_list(self, sample_frame):
        """Test empty list returns unchanged copy."""
        result = sample_frame.drop_columns([])
        assert result.columns == ["a", "b", "c", "d"]
        assert result.shape == (3, 4)
        assert result is not sample_frame

    def test_drop_columns_duplicates_ignored(self, sample_frame):
        """Test duplicates are silently ignored."""
        result = sample_frame.drop_columns(["a", "a", "b", "b"])
        assert result.columns == ["c", "d"]

    def test_drop_columns_drop_all_raises(self, sample_frame):
        """Test dropping all columns raises ValueError."""
        with pytest.raises(ValueError, match="cannot remove all columns"):
            sample_frame.drop_columns(["a", "b", "c", "d"])

    def test_drop_columns_unknown_raises(self, sample_frame):
        """Test unknown column names raise ValueError."""
        with pytest.raises(ValueError, match=re.escape("Unknown column(s): ['z']")):
            sample_frame.drop_columns(["z"])

    def test_drop_columns_mixed_unknown_raises(self, sample_frame):
        """Test ValueError lists all unknown columns."""
        with pytest.raises(ValueError, match=r"Unknown column\(s\): \['x', 'z'\]"):
            sample_frame.drop_columns(["x", "a", "z"])

    def test_drop_columns_not_list_raises(self, sample_frame):
        """Test non-list/tuple input raises TypeError."""
        with pytest.raises(TypeError, match="cols must be a list or tuple"):
            sample_frame.drop_columns("a")

    def test_drop_columns_set_raises(self, sample_frame):
        """Test set input raises TypeError."""
        with pytest.raises(TypeError, match="cols must be a list or tuple"):
            sample_frame.drop_columns({"a"})

    def test_drop_columns_non_string_element_raises(self, sample_frame):
        """Test non-string elements raise TypeError."""
        with pytest.raises(TypeError, match="All column names in cols must be strings"):
            sample_frame.drop_columns(["a", 1])

    def test_drop_columns_attrs_deep_copied(self, sample_frame):
        """Test that attrs are deep-copied."""
        frame_with_attrs = ar.ArFrame(
            sample_frame._frame,
            attrs={"meta": {"key": [1, 2, 3]}}
        )
        result = frame_with_attrs.drop_columns(["a"])
        # Attrs should be equal but not the same object
        assert result._attrs == {"meta": {"key": [1, 2, 3]}}
        assert result._attrs is not frame_with_attrs._attrs
        assert result._attrs["meta"] is not frame_with_attrs._attrs["meta"]
        # Mutation should not affect original
        result._attrs["meta"]["key"].append(4)
        assert frame_with_attrs._attrs["meta"]["key"] == [1, 2, 3]


class TestSelectColumns:
    """Tests for ArFrame.select_columns()"""

    def test_select_columns_basic(self, sample_frame):
        """Test basic column selection."""
        result = sample_frame.select_columns(["a", "c"])
        assert result.columns == ["a", "c"]
        assert result.shape == (3, 2)
        assert sample_frame.columns == ["a", "b", "c", "d"]  # unchanged

    def test_select_columns_single(self, sample_frame):
        """Test selecting a single column."""
        result = sample_frame.select_columns(["c"])
        assert result.columns == ["c"]
        assert result.shape == (3, 1)

    def test_select_columns_tuple_input(self, sample_frame):
        """Test tuple input works like list."""
        result = sample_frame.select_columns(("b", "d"))
        assert result.columns == ["b", "d"]

    def test_select_columns_all_columns(self, sample_frame):
        """Test selecting all columns returns new frame."""
        result = sample_frame.select_columns(["a", "b", "c", "d"])
        assert result.columns == ["a", "b", "c", "d"]
        assert result is not sample_frame
        assert result.shape == sample_frame.shape

    def test_select_columns_string_raises(self, sample_frame):
        """Test passing a string raises TypeError."""
        with pytest.raises(TypeError, match="must be a sequence of column names"):
            sample_frame.select_columns("a")

    def test_select_columns_empty_raises(self, sample_frame):
        """Test empty list raises ValueError."""
        with pytest.raises(ValueError, match="Column selection cannot be empty"):
            sample_frame.select_columns([])

    def test_select_columns_non_list_raises(self, sample_frame):
        """Test non-list/tuple input raises TypeError."""
        with pytest.raises(TypeError, match="must be a list or tuple of column names"):
            sample_frame.select_columns({"a"})

    def test_select_columns_non_string_element_raises(self, sample_frame):
        """Test non-string elements raise TypeError."""
        with pytest.raises(TypeError, match="All column names must be strings"):
            sample_frame.select_columns(["a", 1])

    def test_select_columns_duplicates_raises(self, sample_frame):
        """Test duplicate column names raise ValueError."""
        with pytest.raises(ValueError, match="Duplicate column names"):
            sample_frame.select_columns(["a", "a"])

    def test_select_columns_unknown_raises(self, sample_frame):
        """Test unknown column names raise ValueError."""
        with pytest.raises(ValueError, match=r"Unknown columns: \['z'\]"):
            sample_frame.select_columns(["z"])

    def test_select_columns_mixed_unknown_raises(self, sample_frame):
        """Test ValueError lists all unknown columns."""
        with pytest.raises(ValueError, match=r"Unknown columns: \['x', 'z'\]"):
            sample_frame.select_columns(["x", "a", "z"])

    def test_select_columns_attrs_deep_copied(self, sample_frame):
        """Test that attrs are deep-copied."""
        frame_with_attrs = ar.ArFrame(
            sample_frame._frame,
            attrs={"meta": {"key": [1, 2, 3]}}
        )
        result = frame_with_attrs.select_columns(["a", "b"])
        assert result._attrs == {"meta": {"key": [1, 2, 3]}}
        assert result._attrs is not frame_with_attrs._attrs
        assert result._attrs["meta"] is not frame_with_attrs._attrs["meta"]


class TestSelectDtypes:
    """Tests for ArFrame.select_dtypes()"""

    def test_select_dtypes_include_int(self, sample_frame):
        """Test filtering by include int64."""
        result = sample_frame.select_dtypes(include="int64")
        assert result.columns == ["a"]

    def test_select_dtypes_include_multiple(self, sample_frame):
        """Test include multiple dtypes."""
        result = sample_frame.select_dtypes(include=["int64", "float64"])
        assert set(result.columns) == {"a", "b"}
        assert result.shape == (3, 2)

    def test_select_dtypes_exclude_string(self, sample_frame):
        """Test exclude string dtype."""
        result = sample_frame.select_dtypes(exclude="string")
        assert "c" not in result.columns
        assert result.shape == (3, 3)

    def test_select_dtypes_include_and_exclude(self, sample_frame):
        """Test both include and exclude."""
        result = sample_frame.select_dtypes(
            include=["int64", "float64", "string"],
            exclude=["string"]
        )
        assert set(result.columns) == {"a", "b"}

    def test_select_dtypes_no_args_raises(self, sample_frame):
        """Test missing both args raises ValueError."""
        with pytest.raises(ValueError, match="requires at least one"):
            sample_frame.select_dtypes()

    def test_select_dtypes_non_string_element_raises(self, sample_frame):
        """Test non-string in list raises TypeError."""
        with pytest.raises(TypeError, match="must contain only strings"):
            sample_frame.select_dtypes(include=["int64", 1])

    def test_select_dtypes_wrong_container_raises(self, sample_frame):
        """Test wrong container type raises TypeError."""
        with pytest.raises(TypeError, match="must be a string, list, or tuple"):
            sample_frame.select_dtypes(include={"int64"})

    def test_select_dtypes_unknown_dtype_raises(self, sample_frame):
        """Test unknown dtype raises ValueError."""
        with pytest.raises(ValueError, match="Unrecognised dtype"):
            sample_frame.select_dtypes(include="object")

    def test_select_dtypes_overlap_raises(self, sample_frame):
        """Test overlap between include and exclude raises ValueError."""
        with pytest.raises(ValueError, match="overlap"):
            sample_frame.select_dtypes(
                include=["int64", "string"],
                exclude=["string"]
            )

    def test_select_dtypes_no_match_empty_result(self, sample_frame):
        """Test no-match case returns empty frame."""
        result = sample_frame.select_dtypes(include="null")
        assert result.columns == []
        assert result.shape[1] == 0

    def test_select_dtypes_empty_include_no_match(self, sample_frame):
        """Test empty include list gives no columns."""
        result = sample_frame.select_dtypes(include=[])
        assert result.columns == []

    def test_select_dtypes_empty_exclude_all_match(self, sample_frame):
        """Test empty exclude list excludes nothing."""
        result = sample_frame.select_dtypes(exclude=[])
        assert result.columns == ["a", "b", "c", "d"]

    def test_select_dtypes_tuple_input(self, sample_frame):
        """Test tuple input works like list."""
        result = sample_frame.select_dtypes(include=("int64", "float64"))
        assert set(result.columns) == {"a", "b"}
