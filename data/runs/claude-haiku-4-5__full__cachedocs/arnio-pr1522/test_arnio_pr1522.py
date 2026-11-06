import pytest
import pandas as pd
import numpy as np
import arnio as ar


def test_collapse_multiple_internal_spaces():
    """Test collapsing multiple internal spaces to single space"""
    df = pd.DataFrame({"text": ["hello   world", "foo    bar    baz"]})
    frame = ar.from_pandas(df)
    result = ar.normalize_whitespace(frame)
    result_df = ar.to_pandas(result)
    
    assert result_df["text"].iloc[0] == "hello world"
    assert result_df["text"].iloc[1] == "foo bar baz"


def test_strip_leading_and_trailing_whitespace():
    """Test stripping leading and trailing whitespace"""
    df = pd.DataFrame({"text": ["  hello world  ", "\t  test  "]})
    frame = ar.from_pandas(df)
    result = ar.normalize_whitespace(frame)
    result_df = ar.to_pandas(result)
    
    assert result_df["text"].iloc[0] == "hello world"
    assert result_df["text"].iloc[1] == "test"


def test_default_processes_all_string_columns():
    """Test that default (columns=None) processes all string columns"""
    df = pd.DataFrame({
        "text1": ["  hello   world  "],
        "text2": ["\t  multiple  \t  spaces"],
        "number": [42]
    })
    frame = ar.from_pandas(df)
    result = ar.normalize_whitespace(frame)
    result_df = ar.to_pandas(result)
    
    assert result_df["text1"].iloc[0] == "hello world"
    assert "multiple" in result_df["text2"].iloc[0]
    assert result_df["number"].iloc[0] == 42


def test_specific_columns_parameter():
    """Test processing only specified columns"""
    df = pd.DataFrame({
        "process": ["  hello   world  "],
        "ignore": ["  keep   spaces  "]
    })
    frame = ar.from_pandas(df)
    result = ar.normalize_whitespace(frame, columns=["process"])
    result_df = ar.to_pandas(result)
    
    assert result_df["process"].iloc[0] == "hello world"
    assert "   " in result_df["ignore"].iloc[0]


def test_preserves_null_values():
    """Test that null/None values are preserved"""
    df = pd.DataFrame({"text": ["  hello  ", None, "  world  "]})
    frame = ar.from_pandas(df)
    result = ar.normalize_whitespace(frame)
    result_df = ar.to_pandas(result)
    
    assert result_df["text"].iloc[0] == "hello"
    assert pd.isna(result_df["text"].iloc[1])
    assert result_df["text"].iloc[2] == "world"


def test_numeric_columns_unaffected():
    """Test that numeric columns are not modified"""
    df = pd.DataFrame({
        "text": ["  hello  "],
        "int_col": [42],
        "float_col": [3.14]
    })
    frame = ar.from_pandas(df)
    result = ar.normalize_whitespace(frame)
    result_df = ar.to_pandas(result)
    
    assert result_df["int_col"].iloc[0] == 42
    assert result_df["float_col"].iloc[0] == 3.14


def test_empty_dataframe():
    """Test handling of empty dataframe"""
    df = pd.DataFrame({"text": pd.Series([], dtype="object")})
    frame = ar.from_pandas(df)
    result = ar.normalize_whitespace(frame)
    result_df = ar.to_pandas(result)
    
    assert len(result_df) == 0
    assert "text" in result_df.columns


def test_various_whitespace_characters():
    """Test collapsing tabs, newlines, and mixed whitespace"""
    df = pd.DataFrame({
        "text": ["hello\t\t\tworld", "foo\n\nbar", "baz   \t  qux"]
    })
    frame = ar.from_pandas(df)
    result = ar.normalize_whitespace(frame)
    result_df = ar.to_pandas(result)
    
    # Should collapse all whitespace runs to single space
    assert "hello" in result_df["text"].iloc[0]
    assert "world" in result_df["text"].iloc[0]


def test_already_clean_data_unchanged():
    """Test that already clean data is not affected"""
    df = pd.DataFrame({"text": ["hello world", "foo bar baz"]})
    frame = ar.from_pandas(df)
    result = ar.normalize_whitespace(frame)
    result_df = ar.to_pandas(result)
    
    assert result_df["text"].iloc[0] == "hello world"
    assert result_df["text"].iloc[1] == "foo bar baz"


def test_in_pipeline():
    """Test using normalize_whitespace in a pipeline"""
    df = pd.DataFrame({"text": ["  hello   world  ", "  test   data  "]})
    frame = ar.from_pandas(df)
    result = ar.pipeline(frame, [("normalize_whitespace",)])
    result_df = ar.to_pandas(result)
    
    assert result_df["text"].iloc[0] == "hello world"
    assert result_df["text"].iloc[1] == "test data"


def test_multiple_specified_columns():
    """Test processing multiple specified columns"""
    df = pd.DataFrame({
        "col1": ["  a   b  "],
        "col2": ["  x   y  "],
        "col3": ["  p   q  "]
    })
    frame = ar.from_pandas(df)
    result = ar.normalize_whitespace(frame, columns=["col1", "col3"])
    result_df = ar.to_pandas(result)
    
    assert result_df["col1"].iloc[0] == "a b"
    assert result_df["col3"].iloc[0] == "p q"
    assert "   " in result_df["col2"].iloc[0]


def test_whitespace_only_becomes_empty_string():
    """Test that cells containing only whitespace become empty strings"""
    df = pd.DataFrame({"text": ["   ", "\t\t", "  \n  "]})
    frame = ar.from_pandas(df)
    result = ar.normalize_whitespace(frame)
    result_df = ar.to_pandas(result)
    
    assert result_df["text"].iloc[0] == ""
    assert result_df["text"].iloc[1] == ""
    assert result_df["text"].iloc[2] == ""


def test_mixed_content_with_numbers_and_special_chars():
    """Test normalization with mixed alphanumeric and special characters"""
    df = pd.DataFrame({"text": ["  hello 123   world!  ", "test-data   with   dashes"]})
    frame = ar.from_pandas(df)
    result = ar.normalize_whitespace(frame)
    result_df = ar.to_pandas(result)
    
    assert result_df["text"].iloc[0] == "hello 123 world!"
    assert result_df["text"].iloc[1] == "test-data with dashes"


def test_single_word_with_surrounding_whitespace():
    """Test single word surrounded by significant whitespace"""
    df = pd.DataFrame({"text": ["   hello   "]})
    frame = ar.from_pandas(df)
    result = ar.normalize_whitespace(frame)
    result_df = ar.to_pandas(result)
    
    assert result_df["text"].iloc[0] == "hello"


def test_nonexistent_column_raises_error():
    """Test that specifying a non-existent column raises an error"""
    df = pd.DataFrame({"text": ["  hello  "]})
    frame = ar.from_pandas(df)
    
    with pytest.raises((KeyError, ValueError)):
        ar.normalize_whitespace(frame, columns=["nonexistent_column"])


def test_empty_columns_list_no_processing():
    """Test that empty columns list skips processing"""
    df = pd.DataFrame({"text": ["  hello   world  "]})
    frame = ar.from_pandas(df)
    result = ar.normalize_whitespace(frame, columns=[])
    result_df = ar.to_pandas(result)
    
    # With empty columns list, no columns should be processed
    assert result_df["text"].iloc[0] == "  hello   world  "
