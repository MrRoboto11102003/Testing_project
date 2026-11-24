import re
import pytest
import pandas as pd
import arnio as ar


@pytest.fixture
def test_frame():
    """Create a test frame with multiple data types."""
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [1.5, 2.5, 3.5],
        "c": ["x", "y", "z"],
        "d": [True, False, True]
    })
    return ar.from_pandas(df)


@pytest.fixture
def test_frame_with_attrs():
    """Create a test frame with custom attributes."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    frame = ar.from_pandas(df)
    frame._attrs = {"meta": {"key": "value", "nested": {"level": 1}}}
    return frame


# ============================================================================
# drop_columns tests
# ============================================================================

def test_drop_columns_basic(test_frame):
    """Test normal column dropping and that result is new object."""
    result = test_frame.drop_columns(["a", "c"])
    assert result.columns == ["b", "d"]
    assert len(result) == 3
    assert result is not test_frame
    assert test_frame.columns == ["a", "b", "c", "d"]  # original unchanged


def test_drop_columns_empty_and_tuple(test_frame):
    """Test empty list returns copy and tuple input works."""
    # Empty list
    result = test_frame.drop_columns([])
    assert result.columns == ["a", "b", "c", "d"]
    assert result is not test_frame
    # Tuple input
    result = test_frame.drop_columns(("a", "b"))
    assert result.columns == ["c", "d"]


def test_drop_columns_attrs_deepcopy(test_frame_with_attrs):
    """Test attrs are deep-copied and mutations don't affect original."""
    result = test_frame_with_attrs.drop_columns(["a"])
    assert result._attrs["meta"] is not test_frame_with_attrs._attrs["meta"]
    result._attrs["meta"]["nested"]["level"] = 2
    assert test_frame_with_attrs._attrs["meta"]["nested"]["level"] == 1


def test_drop_columns_errors(test_frame):
    """Test drop_columns error cases."""
    # Drop all columns
    with pytest.raises(ValueError, match="cannot remove all columns"):
        test_frame.drop_columns(["a", "b", "c", "d"])
    # Unknown columns
    with pytest.raises(ValueError, match="Unknown column"):
        test_frame.drop_columns(["x"])
    # Non-list/tuple
    with pytest.raises(TypeError, match="must be a list or tuple"):
        test_frame.drop_columns("a")
    # Non-string element
    with pytest.raises(TypeError, match="must be strings"):
        test_frame.drop_columns(["a", 1])


def test_drop_columns_duplicates_dedup(test_frame):
    """Test duplicates are dedup'd and original order preserved."""
    result = test_frame.drop_columns(["b", "b", "a", "a"])
    assert result.columns == ["c", "d"]


# ============================================================================
# select_columns tests
# ============================================================================

def test_select_columns_basic(test_frame):
    """Test normal column selection."""
    result = test_frame.select_columns(["a", "c"])
    assert set(result.columns) == {"a", "c"}
    assert len(result) == 3
    assert result is not test_frame


def test_select_columns_tuple_input(test_frame):
    """Test tuple input and single column."""
    result = test_frame.select_columns(("b",))
    assert result.columns == ["b"]


def test_select_columns_attrs_deepcopy(test_frame_with_attrs):
    """Test attrs are deep-copied and mutations don't affect original."""
    result = test_frame_with_attrs.select_columns(["a"])
    assert result._attrs["meta"] is not test_frame_with_attrs._attrs["meta"]
    result._attrs["meta"]["nested"]["level"] = 2
    assert test_frame_with_attrs._attrs["meta"]["nested"]["level"] == 1


def test_select_columns_errors(test_frame):
    """Test select_columns error cases."""
    # String input
    with pytest.raises(TypeError, match="not a string"):
        test_frame.select_columns("a")
    # Empty selection
    with pytest.raises(ValueError, match="cannot be empty"):
        test_frame.select_columns([])
    # Non-list/tuple
    with pytest.raises(TypeError, match="must be a list or tuple"):
        test_frame.select_columns({"a"})
    # Non-string element
    with pytest.raises(TypeError, match="must be strings"):
        test_frame.select_columns(["a", 1])
    # Duplicates
    with pytest.raises(ValueError, match="Duplicate"):
        test_frame.select_columns(["a", "a"])
    # Unknown
    with pytest.raises(ValueError, match="Unknown columns"):
        test_frame.select_columns(["x"])


# ============================================================================
# select_dtypes tests
# ============================================================================

def test_select_dtypes_include_exclude(test_frame):
    """Test include and exclude dtype filters."""
    # Include single dtype
    result = test_frame.select_dtypes(include="int64")
    assert len(result.columns) == 1
    # Exclude single dtype
    result = test_frame.select_dtypes(exclude="string")
    assert "c" not in result.columns
    # Both include and exclude
    result = test_frame.select_dtypes(include=["int64", "float64"], exclude="int64")
    assert "a" not in result.columns
    assert "b" in result.columns


def test_select_dtypes_no_match_and_empty(test_frame):
    """Test no match and empty include/exclude filters."""
    # No match (empty include)
    result = test_frame.select_dtypes(include=[])
    assert result.columns == []
    # Empty exclude does nothing
    result = test_frame.select_dtypes(exclude=[])
    assert len(result.columns) == 4


def test_select_dtypes_tuple_and_multiple(test_frame):
    """Test tuple input and multiple include dtypes."""
    result = test_frame.select_dtypes(include=("int64", "float64"))
    assert len(result.columns) == 2


def test_select_dtypes_errors(test_frame):
    """Test select_dtypes error cases."""
    # Both None
    with pytest.raises(ValueError, match="requires at least one"):
        test_frame.select_dtypes()
    # Invalid dtype
    with pytest.raises(ValueError, match="Unrecognised dtype"):
        test_frame.select_dtypes(include="object")
    # Invalid container
    with pytest.raises(TypeError, match="must be a string, list, or tuple"):
        test_frame.select_dtypes(include={"int64"})
    # Non-string in list
    with pytest.raises(TypeError, match="must contain only strings"):
        test_frame.select_dtypes(include=["int64", 1])
    # Overlap between include and exclude
    with pytest.raises(ValueError, match="overlap"):
        test_frame.select_dtypes(include="int64", exclude="int64")
