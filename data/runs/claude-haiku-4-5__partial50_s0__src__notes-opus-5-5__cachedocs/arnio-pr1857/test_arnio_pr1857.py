import pytest
import pandas as pd
import numpy as np
import re
import arnio as ar
from decimal import Decimal


def test_from_dict_simple_conversion():
    """Test basic dict to ArFrame conversion."""
    data = {"name": ["Alice", "Bob"], "age": [25, 30]}
    frame = ar.from_dict(data)
    assert frame.shape == (2, 2)
    assert frame.columns == ["name", "age"]


def test_from_dict_empty_dict():
    """Test that empty dict returns empty frame."""
    frame = ar.from_dict({})
    assert frame.shape == (0, 0)
    assert frame.columns == []


def test_from_dict_rejects_non_dict_input():
    """Test that non-dict inputs raise TypeError with correct message."""
    with pytest.raises(TypeError, match="Expected dict datatype but instead got list"):
        ar.from_dict([1, 2, 3])
    
    with pytest.raises(TypeError, match="Expected dict datatype but instead got str"):
        ar.from_dict("not a dict")
    
    with pytest.raises(TypeError, match="Expected dict datatype but instead got tuple"):
        ar.from_dict((1, 2))


def test_from_dict_rejects_non_string_keys():
    """Test that non-string keys raise TypeError."""
    with pytest.raises(TypeError, match="All dictionary keys must be strings"):
        ar.from_dict({1: [1, 2]})
    
    with pytest.raises(TypeError, match="All dictionary keys must be strings"):
        ar.from_dict({("a", "b"): [1, 2]})


def test_from_dict_rejects_nested_dicts():
    """Test that nested dict values raise ValueError."""
    with pytest.raises(ValueError, match="Nested objects are not supported in column 'a'"):
        ar.from_dict({"a": {"x": 1}})


def test_from_dict_rejects_non_sequences():
    """Test that non-sequence values raise TypeError."""
    with pytest.raises(TypeError, match="Column 'a' must be a sequence of values, not str"):
        ar.from_dict({"a": "hello"})
    
    with pytest.raises(TypeError, match="Column 'a' must be a sequence of values, not bytes"):
        ar.from_dict({"a": b"bytes"})
    
    with pytest.raises(TypeError, match="Column 'a' must be a sequence of values, not int"):
        ar.from_dict({"a": 5})
    
    with pytest.raises(TypeError, match="Column 'a' must be a sequence of values, not NoneType"):
        ar.from_dict({"a": None})


def test_from_dict_requires_matching_lengths():
    """Test that mismatched column lengths raise ValueError with details."""
    with pytest.raises(ValueError, match=re.escape("from_dict() column lengths differ: a=2, b=1")):
        ar.from_dict({"a": [1, 2], "b": [1]})
    
    with pytest.raises(ValueError, match=re.escape("from_dict() column lengths differ: a=1, b=2, c=1")):
        ar.from_dict({"a": [1], "b": [1, 2], "c": [1]})


def test_from_dict_accepts_various_sequence_types():
    """Test that various sequence types work: list, tuple, range, ndarray, Series."""
    frame_list = ar.from_dict({"a": [1, 2, 3]})
    frame_tuple = ar.from_dict({"a": (1, 2, 3)})
    frame_range = ar.from_dict({"a": range(3)})
    frame_ndarray = ar.from_dict({"a": np.array([1, 2, 3])})
    frame_series = ar.from_dict({"a": pd.Series([1, 2, 3])})
    
    assert all(f.shape == (3, 1) for f in [frame_list, frame_tuple, frame_range, frame_ndarray, frame_series])


def test_from_dict_preserves_column_order():
    """Test that dict insertion order is preserved."""
    frame = ar.from_dict({"z": [1], "a": [2], "m": [3]})
    assert frame.columns == ["z", "a", "m"]


def test_from_dict_converts_to_correct_dtypes():
    """Test that values are converted to correct pandas dtypes."""
    frame_int = ar.from_dict({"a": [1, 2, 3]})
    frame_float = ar.from_dict({"a": [1.5, 2.5]})
    frame_str = ar.from_dict({"a": ["x", "y", "z"]})
    frame_bool = ar.from_dict({"a": [True, False, True]})
    
    df_int = ar.to_pandas(frame_int)
    df_float = ar.to_pandas(frame_float)
    df_str = ar.to_pandas(frame_str)
    df_bool = ar.to_pandas(frame_bool)
    
    assert df_int["a"].dtype.name in ("int64", "Int64")
    assert df_float["a"].dtype == np.float64
    assert df_str["a"].dtype.name == "string"
    assert df_bool["a"].dtype.name == "boolean"


def test_from_dict_handles_nulls():
    """Test null handling with mixed types."""
    frame_int_none = ar.from_dict({"a": [1, None, 3]})
    frame_str_none = ar.from_dict({"a": ["x", None, "z"]})
    
    df_int = ar.to_pandas(frame_int_none)
    df_str = ar.to_pandas(frame_str_none)
    
    # Int with None becomes float64 with NaN
    assert df_int["a"].dtype == np.float64
    assert pd.isna(df_int["a"].iloc[1])
    
    # String with None should have pd.NA
    assert pd.isna(df_str["a"].iloc[1])


def test_from_dict_rejects_unsupported_pandas_dtypes():
    """Test that unsupported pandas dtypes raise TypeError."""
    with pytest.raises(TypeError, match="unsupported dtype 'datetime64"):
        ar.from_dict({"d": pd.to_datetime(["2020-01-01"])})
    
    with pytest.raises(TypeError, match="unsupported dtype 'timedelta64"):
        ar.from_dict({"t": pd.to_timedelta(["1s"])})
    
    with pytest.raises(TypeError, match="unsupported dtype 'category'"):
        ar.from_dict({"c": pd.Categorical(["a", "b"])})
    
    with pytest.raises(TypeError, match="unsupported dtype 'complex128'"):
        ar.from_dict({"z": [1+2j]})


def test_from_dict_rejects_nested_structures():
    """Test that nested structures in list values raise TypeError."""
    with pytest.raises(TypeError, match="unsupported nested value of type 'list'"):
        ar.from_dict({"a": [[1, 2]]})
    
    with pytest.raises(TypeError, match="unsupported nested value of type 'dict'"):
        ar.from_dict({"a": [{"x": 1}]})


def test_from_dict_rejects_infinity():
    """Test that infinity in numeric columns raises ValueError."""
    with pytest.raises(ValueError, match="Invalid financial value"):
        ar.from_dict({"a": [1.0, float("inf")]})


def test_from_dict_rejects_integer_overflow():
    """Test that integers outside int64 range raise ValueError."""
    with pytest.raises(ValueError, match="out of bounds for signed 64-bit integer"):
        ar.from_dict({"a": [2**63]})


def test_from_dict_coerces_mixed_types():
    """Test that mixed types are properly coerced."""
    frame_mixed_int_float = ar.from_dict({"a": [1, 2.5, 3]})
    frame_mixed_str_int = ar.from_dict({"a": [1, "x", 3]})
    
    df_mixed_num = ar.to_pandas(frame_mixed_int_float)
    df_mixed_str = ar.to_pandas(frame_mixed_str_int)
    
    # Mixed int and float becomes float64
    assert df_mixed_num["a"].dtype == np.float64
    
    # Mixed string and int coerces to string
    assert df_mixed_str["a"].dtype.name == "string"
    assert list(df_mixed_str["a"]) == ["1", "x", "3"]
