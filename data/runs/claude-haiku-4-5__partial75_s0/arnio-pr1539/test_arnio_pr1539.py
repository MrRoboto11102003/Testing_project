import pytest
import arnio as ar
import pandas as pd


def test_drop_single_column():
    """Drop a single column from frame."""
    frame = ar.from_pandas(pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["A", "B", "C"],
        "age": [25, 30, 35],
    }))
    result = frame.drop_columns(["id"])
    assert result.columns == ["name", "age"]


def test_drop_multiple_columns():
    """Drop multiple columns from frame."""
    frame = ar.from_pandas(pd.DataFrame({
        "a": [1], "b": [2], "c": [3], "d": [4]
    }))
    result = frame.drop_columns(["a", "d"])
    assert result.columns == ["b", "c"]


def test_drop_preserves_column_order():
    """Remaining columns preserve original order."""
    frame = ar.from_pandas(pd.DataFrame({
        "a": [1], "b": [2], "c": [3], "d": [4]
    }))
    result = frame.drop_columns(["b", "d"])
    assert result.columns == ["a", "c"]


def test_drop_empty_list():
    """Empty list leaves frame unchanged."""
    frame = ar.from_pandas(pd.DataFrame({"a": [1], "b": [2]}))
    result = frame.drop_columns([])
    assert result.columns == ["a", "b"]


def test_drop_duplicate_names_silently_ignored():
    """Duplicate column names in drop list are silently ignored."""
    frame = ar.from_pandas(pd.DataFrame({"a": [1], "b": [2], "c": [3]}))
    result = frame.drop_columns(["a", "a", "b", "b"])
    assert result.columns == ["c"]


def test_drop_all_columns():
    """Dropping all columns results in empty frame."""
    frame = ar.from_pandas(pd.DataFrame({"a": [1], "b": [2]}))
    result = frame.drop_columns(["a", "b"])
    assert result.columns == []
    assert result.shape[0] == 1


def test_drop_tuple_input():
    """Tuple input works like list input."""
    frame = ar.from_pandas(pd.DataFrame({"a": [1], "b": [2], "c": [3]}))
    result = frame.drop_columns(("a", "c"))
    assert result.columns == ["b"]


def test_drop_nonexistent_column_raises_valueerror():
    """ValueError raised for non-existent column."""
    frame = ar.from_pandas(pd.DataFrame({"a": [1], "b": [2]}))
    with pytest.raises(ValueError):
        frame.drop_columns(["nonexistent"])


def test_drop_does_not_mutate_original():
    """Original frame is not modified."""
    frame = ar.from_pandas(pd.DataFrame({"a": [1], "b": [2], "c": [3]}))
    original_cols = frame.columns.copy()
    frame.drop_columns(["b"])
    assert frame.columns == original_cols


def test_drop_returns_arframe():
    """Result is an ArFrame instance with expected properties."""
    frame = ar.from_pandas(pd.DataFrame({"a": [1], "b": [2]}))
    result = frame.drop_columns(["a"])
    assert hasattr(result, 'columns')
    assert hasattr(result, 'shape')
    assert hasattr(result, 'dtypes')


def test_drop_string_not_list_raises_typeerror():
    """String input raises TypeError."""
    frame = ar.from_pandas(pd.DataFrame({"a": [1], "b": [2]}))
    with pytest.raises(TypeError):
        frame.drop_columns("a")


def test_drop_non_string_elements_raises_typeerror():
    """Non-string elements raise TypeError."""
    frame = ar.from_pandas(pd.DataFrame({"a": [1], "b": [2]}))
    with pytest.raises(TypeError):
        frame.drop_columns([1, 2])


def test_drop_case_sensitive():
    """Column name matching is case-sensitive."""
    frame = ar.from_pandas(pd.DataFrame({"id": [1], "name": ["A"]}))
    with pytest.raises(ValueError):
        frame.drop_columns(["ID"])


def test_drop_preserves_data():
    """Data in remaining columns is preserved correctly."""
    frame = ar.from_pandas(pd.DataFrame({
        "id": [1, 2],
        "value": [10, 20],
        "temp": [99, 99]
    }))
    result = frame.drop_columns(["temp"])
    df = ar.to_pandas(result)
    assert list(df["id"]) == [1, 2]
    assert list(df["value"]) == [10, 20]


def test_drop_with_nulls():
    """Works correctly with null values in data."""
    frame = ar.from_pandas(pd.DataFrame({
        "a": [1, None, 3],
        "b": ["x", "y", None],
        "c": [True, False, True]
    }))
    result = frame.drop_columns(["b"])
    assert result.columns == ["a", "c"]
