import pytest
import pandas as pd
import arnio as ar
import re


def test_drop_columns_matching_basic():
    """Drop columns matching a simple prefix pattern."""
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "temp_col1": [4, 5, 6],
        "temp_col2": [7, 8, 9],
        "name": ["a", "b", "c"]
    })
    result = ar.drop_columns_matching(df, "^temp_")
    assert list(result.columns) == ["id", "name"]


def test_drop_columns_matching_arframe_input():
    """Function accepts and returns ArFrame."""
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "debug_info": [4, 5, 6],
        "name": ["a", "b", "c"]
    })
    frame = ar.from_pandas(df)
    result = ar.drop_columns_matching(frame, "^debug_")
    assert isinstance(result, ar.ArFrame)
    assert list(ar.to_pandas(result).columns) == ["id", "name"]


def test_drop_columns_matching_preserves_return_type_dataframe():
    """Returns DataFrame when input is DataFrame."""
    df = pd.DataFrame({"col_a": [1], "col_b": [2]})
    result = ar.drop_columns_matching(df, "col_a")
    assert isinstance(result, pd.DataFrame)
    assert not isinstance(result, ar.ArFrame)


def test_drop_columns_matching_no_matches():
    """Pattern that matches nothing leaves all columns."""
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["a", "b", "c"]
    })
    result = ar.drop_columns_matching(df, "^temp_")
    assert list(result.columns) == ["id", "name"]


def test_drop_columns_matching_all_columns_raises():
    """ValueError when pattern matches all columns."""
    df = pd.DataFrame({
        "col_a": [1, 2, 3],
        "col_b": [4, 5, 6]
    })
    with pytest.raises(ValueError, match="At least one column must remain"):
        ar.drop_columns_matching(df, "col_")


def test_drop_columns_matching_non_string_pattern():
    """TypeError when pattern is not a string."""
    df = pd.DataFrame({"col_a": [1, 2, 3]})
    with pytest.raises(TypeError, match="pattern must be a string"):
        ar.drop_columns_matching(df, 123)


def test_drop_columns_matching_invalid_regex():
    """re.error when pattern is not valid regex."""
    df = pd.DataFrame({"col_a": [1, 2, 3]})
    with pytest.raises(re.error):
        ar.drop_columns_matching(df, "[invalid(regex")


def test_drop_columns_matching_complex_regex():
    """More complex regex patterns work correctly."""
    df = pd.DataFrame({
        "user_id": [1, 2, 3],
        "user_name": ["a", "b", "c"],
        "temp_123": [4, 5, 6],
        "temp_456": [7, 8, 9],
        "other": [10, 11, 12]
    })
    result = ar.drop_columns_matching(df, r"\d+$")
    assert list(result.columns) == ["user_id", "user_name", "other"]


def test_drop_columns_matching_case_sensitive():
    """Pattern matching is case-sensitive by default."""
    df = pd.DataFrame({
        "Temp_col": [1, 2, 3],
        "temp_col": [4, 5, 6],
        "name": ["a", "b", "c"]
    })
    result = ar.drop_columns_matching(df, "^temp_")
    assert list(result.columns) == ["Temp_col", "name"]


def test_drop_columns_matching_special_chars():
    """Regex works with special characters in column names."""
    df = pd.DataFrame({
        "col-1": [1, 2, 3],
        "col-2": [4, 5, 6],
        "col_3": [7, 8, 9]
    })
    result = ar.drop_columns_matching(df, r"col-")
    assert list(result.columns) == ["col_3"]


def test_drop_columns_matching_preserves_data():
    """Data in remaining columns is preserved correctly."""
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "temp_x": [10, 20, 30],
        "name": ["a", "b", "c"],
        "temp_y": [100, 200, 300]
    })
    result = ar.drop_columns_matching(df, "^temp_")
    assert result["id"].tolist() == [1, 2, 3]
    assert result["name"].tolist() == ["a", "b", "c"]


def test_drop_columns_matching_empty_dataframe():
    """Works with empty DataFrame (no rows)."""
    df = pd.DataFrame({
        "col_a": pd.Series([], dtype=int),
        "col_b": pd.Series([], dtype=int),
        "name": pd.Series([], dtype=str)
    })
    result = ar.drop_columns_matching(df, "^col_")
    assert list(result.columns) == ["name"]


def test_drop_columns_matching_single_column_remaining():
    """Can leave just one column."""
    df = pd.DataFrame({
        "temp_1": [1, 2, 3],
        "temp_2": [4, 5, 6],
        "keep_me": [7, 8, 9]
    })
    result = ar.drop_columns_matching(df, "^temp_")
    assert list(result.columns) == ["keep_me"]


def test_drop_columns_matching_end_anchor():
    """Dollar sign anchor matches end of column name."""
    df = pd.DataFrame({
        "col_temp": [1, 2, 3],
        "temp": [4, 5, 6],
        "data": [7, 8, 9]
    })
    result = ar.drop_columns_matching(df, "temp$")
    assert list(result.columns) == ["col_temp", "data"]


def test_drop_columns_matching_or_pattern():
    """Regex OR operator works."""
    df = pd.DataFrame({
        "debug": [1, 2, 3],
        "test_col": [4, 5, 6],
        "data": [7, 8, 9]
    })
    result = ar.drop_columns_matching(df, "debug|test_")
    assert list(result.columns) == ["data"]
