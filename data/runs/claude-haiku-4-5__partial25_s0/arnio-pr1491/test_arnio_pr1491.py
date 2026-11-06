import pytest
import pandas as pd
import arnio as ar


def test_drop_columns_matching_prefix_pattern():
    """Drop columns starting with 'temp_' using prefix pattern"""
    frame = ar.from_records([
        {"id": 1, "name": "Alice", "temp_debug": 10, "temp_cache": 20},
        {"id": 2, "name": "Bob", "temp_debug": 30, "temp_cache": 40},
    ])
    result = ar.drop_columns_matching(frame, pattern="^temp_")
    result_df = ar.to_pandas(result)
    
    assert "id" in result_df.columns
    assert "name" in result_df.columns
    assert "temp_debug" not in result_df.columns
    assert "temp_cache" not in result_df.columns
    assert result_df.shape[0] == 2


def test_drop_columns_matching_suffix_pattern():
    """Drop columns ending with '_id' using suffix pattern"""
    frame = ar.from_records([
        {"user_id": 1, "name": "Alice", "order_id": 100},
        {"user_id": 2, "name": "Bob", "order_id": 200},
    ])
    result = ar.drop_columns_matching(frame, pattern="_id$")
    result_df = ar.to_pandas(result)
    
    assert "user_id" not in result_df.columns
    assert "order_id" not in result_df.columns
    assert "name" in result_df.columns
    assert len(result_df.columns) == 1


def test_drop_columns_matching_contains_pattern():
    """Drop columns containing substring using wildcard pattern"""
    frame = ar.from_records([
        {"id": 1, "debug_mode": True, "debug_level": 5, "value": 100},
    ])
    result = ar.drop_columns_matching(frame, pattern="debug")
    result_df = ar.to_pandas(result)
    
    assert "debug_mode" not in result_df.columns
    assert "debug_level" not in result_df.columns
    assert "id" in result_df.columns
    assert "value" in result_df.columns


def test_drop_columns_matching_no_matches():
    """When no columns match pattern, all columns preserved"""
    frame = ar.from_records([
        {"id": 1, "name": "Alice", "age": 30},
        {"id": 2, "name": "Bob", "age": 25},
    ])
    result = ar.drop_columns_matching(frame, pattern="^temp_")
    result_df = ar.to_pandas(result)
    
    assert result_df.shape == (2, 3)
    assert list(result_df.columns) == ["id", "name", "age"]


def test_drop_columns_matching_all_columns_match():
    """When all columns match pattern, returns frame with no columns"""
    frame = ar.from_records([
        {"temp_a": 1, "temp_b": 2},
        {"temp_a": 3, "temp_b": 4},
    ])
    result = ar.drop_columns_matching(frame, pattern="^temp_")
    result_df = ar.to_pandas(result)
    
    assert len(result_df.columns) == 0
    assert result_df.shape[0] == 2


def test_drop_columns_matching_preserves_order():
    """Remaining columns maintain original order"""
    frame = ar.from_records([
        {"a": 1, "temp_x": 10, "b": 2, "temp_y": 20, "c": 3},
    ])
    result = ar.drop_columns_matching(frame, pattern="^temp_")
    result_df = ar.to_pandas(result)
    
    assert list(result_df.columns) == ["a", "b", "c"]


def test_drop_columns_matching_case_sensitive():
    """Pattern matching is case-sensitive"""
    frame = ar.from_records([
        {"Temp_Col": 1, "temp_col": 2, "data": 3},
    ])
    result = ar.drop_columns_matching(frame, pattern="^temp_")
    result_df = ar.to_pandas(result)
    
    assert "Temp_Col" in result_df.columns
    assert "temp_col" not in result_df.columns
    assert "data" in result_df.columns


def test_drop_columns_matching_alternation():
    """Use regex alternation to match multiple patterns"""
    frame = ar.from_records([
        {"debug_info": 1, "test_data": 2, "cache_value": 3, "id": 4},
    ])
    result = ar.drop_columns_matching(frame, pattern="^(debug|cache)_")
    result_df = ar.to_pandas(result)
    
    assert "debug_info" not in result_df.columns
    assert "cache_value" not in result_df.columns
    assert "test_data" in result_df.columns
    assert "id" in result_df.columns


def test_drop_columns_matching_character_class():
    """Use character classes in regex pattern"""
    frame = ar.from_records([
        {"col1": "a", "col2": "b", "colA": "c", "data": "d"},
    ])
    result = ar.drop_columns_matching(frame, pattern="^col[0-9]$")
    result_df = ar.to_pandas(result)
    
    assert "col1" not in result_df.columns
    assert "col2" not in result_df.columns
    assert "colA" in result_df.columns
    assert "data" in result_df.columns


def test_drop_columns_matching_single_column_matches():
    """Frame with single matching column"""
    frame = ar.from_records([
        {"debug": 1},
        {"debug": 2},
    ])
    result = ar.drop_columns_matching(frame, pattern="debug")
    result_df = ar.to_pandas(result)
    
    assert len(result_df.columns) == 0
    assert result_df.shape[0] == 2


def test_drop_columns_matching_single_column_no_match():
    """Frame with single non-matching column"""
    frame = ar.from_records([
        {"data": 1},
        {"data": 2},
    ])
    result = ar.drop_columns_matching(frame, pattern="temp")
    result_df = ar.to_pandas(result)
    
    assert list(result_df.columns) == ["data"]
    assert result_df.shape[0] == 2


def test_drop_columns_matching_various_dtypes():
    """Works with columns of different data types"""
    df = pd.DataFrame({
        "id": [1, 2],
        "name": ["Alice", "Bob"],
        "score": [95.5, 87.3],
        "temp_int": [10, 20],
        "temp_str": ["x", "y"],
    })
    frame = ar.from_pandas(df)
    result = ar.drop_columns_matching(frame, pattern="^temp_")
    result_df = ar.to_pandas(result)
    
    assert list(result_df.columns) == ["id", "name", "score"]
    assert result_df.shape[0] == 2


def test_drop_columns_matching_returns_new_frame():
    """Result is a new frame, original unchanged"""
    frame = ar.from_records([
        {"id": 1, "temp_col": 2, "data": 3},
    ])
    original_cols = set(frame.columns)
    result = ar.drop_columns_matching(frame, pattern="temp")
    
    assert set(frame.columns) == original_cols
    assert "temp_col" not in result.columns


def test_drop_columns_matching_wildcard_pattern():
    """Use .* to match any substring"""
    frame = ar.from_records([
        {"_temp_a": 1, "temp_b": 2, "temporary": 3, "data": 4},
    ])
    result = ar.drop_columns_matching(frame, pattern=".*temp.*")
    result_df = ar.to_pandas(result)
    
    assert "_temp_a" not in result_df.columns
    assert "temp_b" not in result_df.columns
    assert "temporary" not in result_df.columns
    assert "data" in result_df.columns


def test_drop_columns_matching_multiple_matches():
    """All matching columns are dropped, non-matching preserved"""
    frame = ar.from_records([
        {"debug_1": 1, "debug_2": 2, "debug_3": 3, "value": 4, "result": 5},
    ])
    result = ar.drop_columns_matching(frame, pattern="^debug_")
    result_df = ar.to_pandas(result)
    
    assert "debug_1" not in result_df.columns
    assert "debug_2" not in result_df.columns
    assert "debug_3" not in result_df.columns
    assert "value" in result_df.columns
    assert "result" in result_df.columns


def test_drop_columns_matching_invalid_regex_raises():
    """Invalid regex pattern raises an exception"""
    frame = ar.from_records([
        {"id": 1, "name": "Alice"},
    ])
    with pytest.raises(Exception):
        ar.drop_columns_matching(frame, pattern="[invalid")
