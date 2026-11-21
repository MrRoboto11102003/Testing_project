import pytest
import pandas as pd
import numpy as np
import arnio as ar


def test_to_pandas_int64_basic():
    """Test basic INT64 column conversion."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    frame = ar.from_pandas(df)
    result = ar.to_pandas(frame)
    assert result["a"].dtype == pd.Int64Dtype()
    assert len(result) == 3


def test_to_pandas_float64_basic():
    """Test basic FLOAT64 column conversion."""
    df = pd.DataFrame({"x": [1.5, 2.5, 3.5]})
    frame = ar.from_pandas(df)
    result = ar.to_pandas(frame)
    assert result["x"].dtype == np.float64
    np.testing.assert_array_equal(result["x"].values, [1.5, 2.5, 3.5])


def test_to_pandas_bool_basic():
    """Test basic BOOL column conversion."""
    df = pd.DataFrame({"flag": [True, False, True]})
    frame = ar.from_pandas(df)
    result = ar.to_pandas(frame)
    assert result["flag"].dtype == pd.BooleanDtype()


def test_to_pandas_string_basic():
    """Test basic STRING column conversion."""
    df = pd.DataFrame({"name": ["alice", "bob"]})
    frame = ar.from_pandas(df)
    result = ar.to_pandas(frame)
    assert result["name"].dtype == pd.StringDtype()
    assert list(result["name"]) == ["alice", "bob"]


def test_to_pandas_mixed_types():
    """Test conversion with multiple columns of different types."""
    df = pd.DataFrame({
        "int_col": [1, 2],
        "float_col": [1.1, 2.2],
        "bool_col": [True, False],
        "str_col": ["a", "b"]
    })
    frame = ar.from_pandas(df)
    result = ar.to_pandas(frame)
    assert len(result.columns) == 4
    assert result["int_col"].dtype == pd.Int64Dtype()
    assert result["float_col"].dtype == np.float64
    assert result["bool_col"].dtype == pd.BooleanDtype()
    assert result["str_col"].dtype == pd.StringDtype()


def test_to_pandas_null_int64():
    """Test handling of null values in INT64 column."""
    df = pd.DataFrame({"a": pd.array([1, None, 3], dtype=pd.Int64Dtype())})
    frame = ar.from_pandas(df)
    result = ar.to_pandas(frame)
    assert pd.isna(result["a"].iloc[1])
    assert result["a"].iloc[0] == 1
    assert result["a"].iloc[2] == 3


def test_to_pandas_null_float64():
    """Test handling of null values in FLOAT64 column using np.nan."""
    df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
    frame = ar.from_pandas(df)
    result = ar.to_pandas(frame)
    assert np.isnan(result["x"].iloc[1])
    assert result["x"].iloc[0] == 1.0
    assert result["x"].iloc[2] == 3.0


def test_to_pandas_null_bool():
    """Test handling of null values in BOOL column."""
    df = pd.DataFrame({"flag": pd.array([True, None, False], dtype=pd.BooleanDtype())})
    frame = ar.from_pandas(df)
    result = ar.to_pandas(frame)
    assert pd.isna(result["flag"].iloc[1])
    assert result["flag"].iloc[0] is True
    assert result["flag"].iloc[2] is False


def test_to_pandas_null_string():
    """Test handling of null values in STRING column."""
    df = pd.DataFrame({"name": pd.array(["alice", None, "bob"], dtype=pd.StringDtype())})
    frame = ar.from_pandas(df)
    result = ar.to_pandas(frame)
    assert pd.isna(result["name"].iloc[1])
    assert result["name"].iloc[0] == "alice"
    assert result["name"].iloc[2] == "bob"


def test_to_pandas_copy_false_default():
    """Test that copy=False is the default zero-copy behavior."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    frame = ar.from_pandas(df)
    result = ar.to_pandas(frame)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3


def test_to_pandas_copy_true_defensive():
    """Test that copy=True creates defensive pandas-owned copies."""
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    frame = ar.from_pandas(df)
    result = ar.to_pandas(frame, copy=True)
    np.testing.assert_array_equal(result["x"].values, [1.0, 2.0, 3.0])


def test_to_pandas_attrs_restoration():
    """Test that attrs metadata is restored from original DataFrame."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    df.attrs = {"description": "test_data", "version": 1}
    frame = ar.from_pandas(df)
    result = ar.to_pandas(frame)
    assert result.attrs.get("description") == "test_data"
    assert result.attrs.get("version") == 1


def test_to_pandas_invalid_copy_type():
    """Test that non-bool copy parameter raises TypeError."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    frame = ar.from_pandas(df)
    with pytest.raises(TypeError, match="copy must be a bool"):
        ar.to_pandas(frame, copy="true")
    with pytest.raises(TypeError, match="copy must be a bool"):
        ar.to_pandas(frame, copy=1)


def test_to_pandas_invalid_frame_type():
    """Test that non-ArFrame input raises TypeError with helpful message."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(TypeError, match="expects an ArFrame"):
        ar.to_pandas(df)
    with pytest.raises(TypeError, match="expects an ArFrame"):
        ar.to_pandas([1, 2, 3])


def test_to_pandas_empty_frame():
    """Test conversion of empty DataFrame with no columns."""
    df = pd.DataFrame()
    frame = ar.from_pandas(df)
    result = ar.to_pandas(frame)
    assert isinstance(result, pd.DataFrame)
    assert len(result.columns) == 0
