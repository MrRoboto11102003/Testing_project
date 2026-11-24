import pytest
import pandas as pd
import numpy as np
import arnio as ar
import re


def test_simple_dict_conversion():
    """Test basic dict to ArFrame conversion."""
    data = {"a": [1, 2, 3], "b": [4.5, 5.5, 6.5]}
    result = ar.from_dict(data)
    df = ar.to_pandas(result)
    assert df.shape == (3, 2)
    assert list(df.columns) == ["a", "b"]


def test_empty_dict():
    """Empty dict returns empty ArFrame."""
    result = ar.from_dict({})
    df = ar.to_pandas(result)
    assert df.shape == (0, 0)


def test_non_dict_input_raises_type_error():
    """Non-dict input raises TypeError with type name."""
    with pytest.raises(TypeError, match="Expected dict datatype but instead got list"):
        ar.from_dict([1, 2, 3])


def test_non_string_key_raises():
    """Non-string keys raise TypeError."""
    with pytest.raises(TypeError, match="All dictionary keys must be strings"):
        ar.from_dict({1: [1, 2, 3]})


def test_string_value_not_sequence():
    """String values raise TypeError (not sequences)."""
    with pytest.raises(TypeError, match=re.escape("Column 'col' must be a sequence of values, not str")):
        ar.from_dict({"col": "abc"})


def test_nested_dict_value_raises():
    """Nested dict values raise ValueError."""
    with pytest.raises(ValueError, match=re.escape("Nested objects are not supported in column 'col'")):
        ar.from_dict({"col": {"x": 1}})


def test_unequal_column_lengths():
    """Columns with different lengths raise ValueError with details in insertion order."""
    with pytest.raises(ValueError, match=re.escape("from_dict() column lengths differ: a=2, b=1")):
        ar.from_dict({"a": [1, 2], "b": [1]})


def test_datetime64_unsupported():
    """datetime64 dtype raises ValueError."""
    with pytest.raises(ValueError, match="unsupported dtype 'datetime64"):
        ar.from_dict({"d": pd.to_datetime(["2020-01-01"])})


def test_timedelta_unsupported():
    """timedelta dtype raises ValueError."""
    with pytest.raises(ValueError, match="unsupported dtype 'timedelta64"):
        ar.from_dict({"t": pd.to_timedelta(["1s"])})


def test_category_unsupported():
    """Category dtype raises ValueError."""
    with pytest.raises(ValueError, match="unsupported dtype 'category'"):
        ar.from_dict({"c": pd.Categorical(["a"])})


def test_sequence_types_accepted():
    """Various sequence types (range, tuple, ndarray) are accepted."""
    # Test range
    result = ar.from_dict({"a": range(3)})
    assert ar.to_pandas(result).shape == (3, 1)
    
    # Test tuple
    result = ar.from_dict({"b": (1, 2, 3)})
    assert ar.to_pandas(result).shape == (3, 1)
    
    # Test numpy array
    result = ar.from_dict({"c": np.array([1, 2, 3])})
    assert ar.to_pandas(result).shape == (3, 1)


def test_column_order_preserved():
    """Column order is preserved from dict insertion order."""
    data = {"z": [1], "a": [2], "m": [3]}
    result = ar.from_dict(data)
    df = ar.to_pandas(result)
    assert list(df.columns) == ["z", "a", "m"]


def test_nested_list_in_values():
    """Nested lists in column values raise TypeError."""
    with pytest.raises(TypeError, match="unsupported nested value"):
        ar.from_dict({"a": [[1, 2]]})


def test_infinity_raises():
    """Infinity in numeric column raises ValueError."""
    with pytest.raises(ValueError, match="Invalid financial value"):
        ar.from_dict({"a": [1.0, float("inf")]})


def test_complex_unsupported():
    """Complex numbers raise ValueError."""
    with pytest.raises(ValueError, match="unsupported dtype 'complex"):
        ar.from_dict({"z": [1+2j]})


def test_bytes_value_raises():
    """Bytes values raise TypeError (not sequences)."""
    with pytest.raises(TypeError, match=re.escape("Column 'col' must be a sequence of values, not bytes")):
        ar.from_dict({"col": b"abc"})


def test_none_value_raises():
    """None values raise TypeError (no __len__)."""
    with pytest.raises(TypeError, match=re.escape("Column 'col' must be a sequence of values, not NoneType")):
        ar.from_dict({"col": None})
