import pytest
import pandas as pd
import arnio as ar


def test_drop_columns_single_column():
    """Test removing a single column."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    frame = ar.from_pandas(df)
    result = ar.drop_columns(frame, ["b"])
    result_df = ar.to_pandas(result)
    assert list(result_df.columns) == ["a", "c"]


def test_drop_columns_multiple_columns():
    """Test removing multiple columns at once."""
    df = pd.DataFrame({"a": [1], "b": [2], "c": [3], "d": [4]})
    frame = ar.from_pandas(df)
    result = ar.drop_columns(frame, ["b", "d"])
    result_df = ar.to_pandas(result)
    assert list(result_df.columns) == ["a", "c"]


def test_drop_columns_preserves_column_order():
    """Test that remaining columns preserve original order."""
    df = pd.DataFrame({"z": [1], "y": [2], "x": [3], "w": [4]})
    frame = ar.from_pandas(df)
    result = ar.drop_columns(frame, ["y", "w"])
    result_df = ar.to_pandas(result)
    assert list(result_df.columns) == ["z", "x"]


def test_drop_columns_preserves_data():
    """Test that data values are preserved after dropping columns."""
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [10, 20, 30],
        "c": [100, 200, 300]
    })
    frame = ar.from_pandas(df)
    result = ar.drop_columns(frame, ["b"])
    result_df = ar.to_pandas(result)
    assert result_df["a"].tolist() == [1, 2, 3]
    assert result_df["c"].tolist() == [100, 200, 300]


def test_drop_columns_empty_list():
    """Test that dropping zero columns returns an equivalent frame."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    frame = ar.from_pandas(df)
    result = ar.drop_columns(frame, [])
    result_df = ar.to_pandas(result)
    assert list(result_df.columns) == ["a", "b"]
    assert result_df.shape == (2, 2)


def test_drop_columns_does_not_mutate_original():
    """Test that the original frame is not modified."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    frame = ar.from_pandas(df)
    original_cols = frame.columns.copy()
    ar.drop_columns(frame, ["b"])
    assert frame.columns == original_cols


def test_drop_columns_missing_column_raises_value_error():
    """Test that removing a non-existent column raises ValueError."""
    df = pd.DataFrame({"a": [1], "b": [2]})
    frame = ar.from_pandas(df)
    with pytest.raises(ValueError, match="not found"):
        ar.drop_columns(frame, ["nonexistent"])


def test_drop_columns_partial_missing_raises_value_error():
    """Test that removing with one missing column raises ValueError."""
    df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    frame = ar.from_pandas(df)
    with pytest.raises(ValueError, match="not found"):
        ar.drop_columns(frame, ["b", "nonexistent"])


def test_drop_columns_remove_all_raises_value_error():
    """Test that removing all columns raises ValueError."""
    df = pd.DataFrame({"a": [1], "b": [2]})
    frame = ar.from_pandas(df)
    with pytest.raises(ValueError, match="cannot remove all columns"):
        ar.drop_columns(frame, ["a", "b"])


def test_drop_columns_single_column_frame_raises_error():
    """Test that removing the only column raises ValueError."""
    df = pd.DataFrame({"only_col": [1, 2, 3]})
    frame = ar.from_pandas(df)
    with pytest.raises(ValueError, match="cannot remove all columns"):
        ar.drop_columns(frame, ["only_col"])


def test_drop_columns_string_argument_raises_type_error():
    """Test that passing a string instead of sequence raises TypeError."""
    df = pd.DataFrame({"a": [1], "b": [2]})
    frame = ar.from_pandas(df)
    with pytest.raises(TypeError):
        ar.drop_columns(frame, "a")


def test_drop_columns_bytes_argument_raises_type_error():
    """Test that passing bytes raises TypeError."""
    df = pd.DataFrame({"a": [1], "b": [2]})
    frame = ar.from_pandas(df)
    with pytest.raises(TypeError):
        ar.drop_columns(frame, b"a")


def test_drop_columns_non_string_item_raises_type_error():
    """Test that non-string column names in sequence raise TypeError."""
    df = pd.DataFrame({"a": [1], "b": [2]})
    frame = ar.from_pandas(df)
    with pytest.raises(TypeError):
        ar.drop_columns(frame, ["a", 1])


def test_drop_columns_with_null_values():
    """Test removing columns from frame with null values."""
    df = pd.DataFrame({
        "a": [1, None, 3],
        "b": [4, 5, 6],
        "c": [None, 8, 9]
    })
    frame = ar.from_pandas(df)
    result = ar.drop_columns(frame, ["b"])
    result_df = ar.to_pandas(result)
    assert list(result_df.columns) == ["a", "c"]
    assert pd.isna(result_df["a"].iloc[1])
    assert pd.isna(result_df["c"].iloc[0])


def test_drop_columns_mixed_data_types():
    """Test removing columns with different data types."""
    df = pd.DataFrame({
        "int_col": [1, 2],
        "float_col": [1.5, 2.5],
        "str_col": ["x", "y"]
    })
    frame = ar.from_pandas(df)
    result = ar.drop_columns(frame, ["float_col"])
    result_df = ar.to_pandas(result)
    assert list(result_df.columns) == ["int_col", "str_col"]
    assert len(result_df) == 2
