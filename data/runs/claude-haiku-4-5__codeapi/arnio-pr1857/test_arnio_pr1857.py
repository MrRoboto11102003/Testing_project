import pytest
import pandas as pd
import numpy as np
import arnio as ar


def test_simple_dict_conversion():
    """Convert a simple dictionary to ArFrame."""
    data = {"A": [1, 2, 3], "B": [4, 5, 6]}
    frame = ar.convert.from_dict(data)
    assert frame.shape == (3, 2)
    assert set(frame.columns) == {"A", "B"}


def test_data_values_preserved():
    """Verify that data values are correctly preserved."""
    data = {"A": [1, 2, 3], "B": ["x", "y", "z"]}
    frame = ar.convert.from_dict(data)
    df = ar.convert.to_pandas(frame)
    pd.testing.assert_frame_equal(df, pd.DataFrame(data))


def test_single_column_dict():
    """Convert dict with a single column."""
    data = {"values": [1, 2, 3, 4]}
    frame = ar.convert.from_dict(data)
    assert frame.shape == (4, 1)
    assert "values" in frame.columns


def test_empty_dict_conversion():
    """Convert an empty dictionary."""
    frame = ar.convert.from_dict({})
    assert frame.shape == (0, 0)


def test_multiple_data_types():
    """Handle columns with different Python data types."""
    data = {
        "ints": [1, 2],
        "floats": [1.1, 2.2],
        "strings": ["a", "b"],
        "bools": [True, False]
    }
    frame = ar.convert.from_dict(data)
    assert frame.shape == (2, 4)


def test_none_and_null_values():
    """Handle None values in columns."""
    data = {"A": [1, None, 3], "B": [None, "y", None]}
    frame = ar.convert.from_dict(data)
    df = ar.convert.to_pandas(frame)
    assert frame.shape == (3, 2)
    assert df.isna().sum().sum() > 0


def test_non_dict_input_error():
    """Raise TypeError when input is not a dict."""
    with pytest.raises(TypeError, match="Expected dict"):
        ar.convert.from_dict([1, 2, 3])
    
    with pytest.raises(TypeError, match="Expected dict"):
        ar.convert.from_dict(42)


def test_non_string_keys_error():
    """Raise TypeError when dictionary keys are not strings."""
    with pytest.raises(TypeError, match="must be strings"):
        ar.convert.from_dict({1: [1, 2], "B": [3, 4]})


def test_string_column_value_error():
    """Raise TypeError when column value is a string instead of sequence."""
    with pytest.raises(TypeError, match="sequence"):
        ar.convert.from_dict({"col": "not_a_sequence"})


def test_nested_dict_error():
    """Raise ValueError when column contains nested dictionaries."""
    with pytest.raises(ValueError, match="Nested"):
        ar.convert.from_dict({"col": [{"key": "value"}]})


def test_mismatched_column_lengths_error():
    """Raise ValueError when columns have different lengths."""
    with pytest.raises(ValueError, match="differ"):
        ar.convert.from_dict({"A": [1, 2, 3], "B": [4, 5]})


def test_bytes_column_error():
    """Raise TypeError when column value is bytes."""
    with pytest.raises(TypeError, match="sequence"):
        ar.convert.from_dict({"col": b"bytes_sequence"})


def test_tuple_sequence_accepted():
    """Accept tuple as column sequence."""
    data = {"A": (1, 2, 3), "B": (4, 5, 6)}
    frame = ar.convert.from_dict(data)
    assert frame.shape == (3, 2)


def test_numpy_array_accepted():
    """Accept numpy arrays as column sequences."""
    data = {"A": np.array([1, 2, 3]), "B": np.array([4, 5, 6])}
    frame = ar.convert.from_dict(data)
    assert frame.shape == (3, 2)


def test_roundtrip_conversion():
    """Dict -> ArFrame -> pandas -> ArFrame roundtrip."""
    data = {"A": [1, 2, 3], "B": [4.0, 5.0, 6.0], "C": ["x", "y", "z"]}
    frame1 = ar.convert.from_dict(data)
    df = ar.convert.to_pandas(frame1)
    frame2 = ar.convert.from_pandas(df)
    assert frame1.shape == frame2.shape
    assert set(frame1.columns) == set(frame2.columns)
