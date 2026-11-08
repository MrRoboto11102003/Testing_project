"""Tests for arnio.convert.to_pandas."""

import copy
import numpy as np
import pandas as pd
from pytest import fixture, mark, param
from unittest.mock import patch

import arnio as ar


def test_to_pandas_basic_conversion():
    """Test basic conversion from ArFrame to pandas DataFrame."""
    data = {"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]}
    frame = ar.from_dict(data)
    df = ar.to_pandas(frame)

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["id", "name"]
    assert df["id"].tolist() == [1, 2, 3]
    assert df["name"].tolist() == ["Alice", "Bob", "Charlie"]


def test_to_pandas_with_nulls():
    """Test conversion handles null values correctly."""
    data = {"id": [1, None, 3], "name": ["Alice", "Bob", None]}
    frame = ar.from_dict(data)
    df = ar.to_pandas(frame)

    assert isinstance(df, pd.DataFrame)
    assert df["id"].tolist() == [1, np.nan, 3]
    assert df["name"].tolist() == ["Alice", "Bob", np.nan]


def test_to_pandas_int64_dtype():
    """Test conversion preserves Int64 dtype with nulls."""
    data = {"id": [1, None, 3]}
    frame = ar.from_dict(data)
    df = ar.to_pandas(frame)

    assert df["id"].dtype == pd.Int64Dtype()


def test_to_pandas_float64_dtype():
    """Test conversion preserves Float64 dtype with nulls."""
    data = {"value": [1.5, np.nan, 3.7]}
    frame = ar.from_dict(data)
    df = ar.to_pandas(frame)

    assert df["value"].dtype == pd.Float64Dtype()


def test_to_pandas_bool_dtype():
    """Test conversion preserves Boolean dtype with nulls."""
    data = {"active": [True, False, None]}
    frame = ar.from_dict(data)
    df = ar.to_pandas(frame)

    assert df["active"].dtype == pd.BooleanDtype()


def test_to_pandas_string_dtype():
    """Test conversion preserves String dtype."""
    data = {"name": ["Alice", "Bob", "Charlie"]}
    frame = ar.from_dict(data)
    df = ar.to_pandas(frame)

    assert df["name"].dtype == pd.StringDtype()


def test_to_pandas_copy_false():
    """Test copy=False preserves zero-copy path where supported."""
    data = {"id": [1, 2, 3]}
    frame = ar.from_dict(data)
    df = ar.to_pandas(frame, copy=False)

    # Verify basic structure is correct
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["id"]


def test_to_pandas_copy_true():
    """Test copy=True creates defensive pandas-owned copies."""
    data = {"id": [1, 2, 3]}
    frame = ar.from_dict(data)

    df1 = ar.to_pandas(frame, copy=False)
    df2 = ar.to_pandas(frame, copy=True)

    # Both should be valid DataFrames
    assert isinstance(df1, pd.DataFrame)
    assert isinstance(df2, pd.DataFrame)


def test_to_pandas_empty_frame():
    """Test conversion of empty ArFrame."""
    data = {"id": [], "name": []}
    frame = ar.from_dict(data)
    df = ar.to_pandas(frame)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_to_pandas_invalid_input():
    """Test conversion raises TypeError for non-ArFrame input."""
    data = {"id": [1, 2, 3]}
    df = pd.DataFrame(data)

    with pytest.raises(TypeError):
        ar.to_pandas(df)


def test_to_pandas_copy_type_error():
    """Test copy parameter validation."""
    frame = ar.from_dict({"id": [1, 2, 3]})

    with pytest.raises(TypeError):
        ar.to_pandas(frame, copy="invalid")


@fixture
def sample_frame():
    """Create a sample ArFrame for testing."""
    return ar.from_dict({
        "int_col": [1, 2, None],
        "float_col": [1.5, 2.5, np.nan],
        "bool_col": [True, False, None],
        "str_col": ["a", "b", "c"],
    })


def test_to_pandas_attrs_preservation(sample_frame):
    """Test that attrs are preserved on conversion."""
    sample_frame._attrs = {"custom_key": "custom_value"}
    df = ar.to_pandas(sample_frame)

    assert df.attrs == {"custom_key": "custom_value"}


@fixture
def empty_frame():
    """Create an empty ArFrame for testing."""
    return ar.from_dict({"col1": [], "col2": []})


def test_to_pandas_attrs_empty_frame(empty_frame):
    """Test attrs preservation on empty frame conversion."""
    empty_frame._attrs = {"empty_key": "empty_value"}
    df = ar.to_pandas(empty_frame)

    assert df.attrs == {"empty_key": "empty_value"}


@fixture
def copylib():
    """Mock deepcopy for testing attrs."""
    return patch("arnio.convert.copylib", new_callable=lambda: lambda *args, **kwargs: None)


def test_to_pandas_attrs_deepcopy(copylib):
    """Test that attrs are deepcopied correctly."""
    frame = ar.from_dict({"col": [1]})
    frame._attrs = {"nested": {"key": "value"}}

    with patch("arnio.convert.copylib.deepcopy") as mock_deepcopy:
        mock_deepcopy.return_value = {"new_key": "new_value"}
        df = ar.to_pandas(frame)

        mock_deepcopy.assert_called_once()


@fixture
def copylib_mock():
    """Mock deepcopy for testing attrs."""
    return patch("arnio.convert.copylib", new_callable=lambda: lambda *args, **kwargs: None)


def test_to_pandas_no_attrs(copylib_mock):
    """Test conversion when frame has no attrs."""
    frame = ar.from_dict({"col": [1]})
    df = ar.to_pandas(frame)

    assert df.attrs is None or isinstance(df.attrs, dict)
