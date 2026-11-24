import pytest
import pandas as pd
import arnio as ar
import re


@pytest.fixture
def simple_frame():
    """Test frame with columns a, b, c."""
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": ["x", None, "z"],
        "c": [1.5, 2.5, 3.5]
    })
    return ar.from_pandas(df)


@pytest.fixture
def frame_with_attrs():
    """Frame with nested attributes."""
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [4, 5, 6],
        "c": [7, 8, 9]
    })
    frame = ar.from_pandas(df)
    frame._attrs = {"meta": {"k": 1, "nested": {"v": 2}}}
    return frame


def test_drop_single_column(simple_frame):
    """Drop a single column."""
    result = simple_frame.drop_columns(["b"])
    assert result.columns == ["a", "c"]
    assert len(result) == 3


def test_drop_multiple_columns(simple_frame):
    """Drop multiple columns."""
    result = simple_frame.drop_columns(["a", "c"])
    assert result.columns == ["b"]
    assert len(result) == 3


def test_drop_tuple_input(simple_frame):
    """Tuple input works."""
    result = simple_frame.drop_columns(("b", "c"))
    assert result.columns == ["a"]


def test_empty_list_returns_copy(simple_frame):
    """Empty list returns unchanged copy."""
    result = simple_frame.drop_columns([])
    assert result.columns == simple_frame.columns
    assert result is not simple_frame


def test_duplicates_silently_ignored(simple_frame):
    """Duplicates in cols are silently ignored."""
    result = simple_frame.drop_columns(["b", "b", "b"])
    assert result.columns == ["a", "c"]


def test_preserves_column_order(simple_frame):
    """Original column order is preserved."""
    result = simple_frame.drop_columns(["c", "a"])
    assert result.columns == ["b"]


def test_returns_new_object(simple_frame):
    """Always returns a new ArFrame object."""
    result = simple_frame.drop_columns(["b"])
    assert result is not simple_frame


def test_deep_copies_attributes(frame_with_attrs):
    """Attributes are deep copied."""
    result = frame_with_attrs.drop_columns(["a"])
    assert result._attrs == frame_with_attrs._attrs
    assert result._attrs is not frame_with_attrs._attrs
    assert result._attrs["meta"] is not frame_with_attrs._attrs["meta"]
    
    result._attrs["meta"]["k"] = 999
    assert frame_with_attrs._attrs["meta"]["k"] == 1


def test_type_error_cols_not_list_or_tuple():
    """TypeError if cols is not list or tuple."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    frame = ar.from_pandas(df)
    
    with pytest.raises(TypeError, match=re.escape("cols must be a list or tuple of column names, got 'str'")):
        frame.drop_columns("a")
    
    with pytest.raises(TypeError, match=re.escape("got 'set'")):
        frame.drop_columns({"a"})
    
    with pytest.raises(TypeError, match=re.escape("got 'NoneType'")):
        frame.drop_columns(None)


def test_type_error_non_string_element():
    """TypeError if any element is not a string."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    frame = ar.from_pandas(df)
    
    with pytest.raises(TypeError, match=re.escape("All column names in cols must be strings.")):
        frame.drop_columns(["a", 1])


def test_value_error_unknown_column():
    """ValueError if column doesn't exist."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    frame = ar.from_pandas(df)
    
    with pytest.raises(ValueError, match=re.escape("Unknown column(s): ['z']")):
        frame.drop_columns(["z"])


def test_value_error_cannot_remove_all():
    """ValueError if dropping all columns."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    frame = ar.from_pandas(df)
    
    with pytest.raises(ValueError, match=re.escape("drop_columns cannot remove all columns from the frame")):
        frame.drop_columns(["a", "b"])


def test_error_order_non_string_before_unknown():
    """Non-string error checked before unknown column error."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    frame = ar.from_pandas(df)
    
    with pytest.raises(TypeError, match=re.escape("All column names in cols must be strings.")):
        frame.drop_columns(["zz", 1])


def test_error_order_unknown_before_remove_all():
    """Unknown column error checked before remove-all error."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    frame = ar.from_pandas(df)
    
    with pytest.raises(ValueError, match=re.escape("Unknown column(s): ['zz']")):
        frame.drop_columns(["a", "b", "zz"])


def test_original_frame_unchanged(simple_frame):
    """Original frame is unchanged after drop_columns."""
    original_cols = simple_frame.columns
    original_shape = simple_frame.shape
    
    simple_frame.drop_columns(["b"])
    
    assert simple_frame.columns == original_cols
    assert simple_frame.shape == original_shape
