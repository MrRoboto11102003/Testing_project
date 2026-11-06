import pytest
import pandas as pd
import arnio as ar


def test_collapse_multiple_spaces():
    """Multiple internal spaces should collapse to single space"""
    frame = ar.from_dict({"text": ["hello   world", "foo  bar  baz"]})
    result = ar.normalize_whitespace(frame)
    df = ar.to_pandas(result)
    assert df["text"].iloc[0] == "hello world"
    assert df["text"].iloc[1] == "foo bar baz"


def test_strip_leading_whitespace():
    """Leading whitespace should be removed"""
    frame = ar.from_dict({"text": ["   hello", "\t\thello"]})
    result = ar.normalize_whitespace(frame)
    df = ar.to_pandas(result)
    assert df["text"].iloc[0] == "hello"
    assert df["text"].iloc[1] == "hello"


def test_strip_trailing_whitespace():
    """Trailing whitespace should be removed"""
    frame = ar.from_dict({"text": ["hello   ", "hello\t\t"]})
    result = ar.normalize_whitespace(frame)
    df = ar.to_pandas(result)
    assert df["text"].iloc[0] == "hello"
    assert df["text"].iloc[1] == "hello"


def test_strip_both_leading_and_trailing():
    """Both leading and trailing whitespace should be removed"""
    frame = ar.from_dict({"text": ["   hello   ", "\t\thello\t\t"]})
    result = ar.normalize_whitespace(frame)
    df = ar.to_pandas(result)
    assert df["text"].iloc[0] == "hello"
    assert df["text"].iloc[1] == "hello"


def test_mixed_whitespace_types():
    """Tabs, newlines, and spaces should all be normalized"""
    frame = ar.from_dict({
        "text": ["hello\t\tworld", "foo\n\nbar", "a  \t  \nb"]
    })
    result = ar.normalize_whitespace(frame)
    df = ar.to_pandas(result)
    assert df["text"].iloc[0] == "hello world"
    assert df["text"].iloc[1] == "foo bar"
    assert df["text"].iloc[2] == "a b"


def test_only_whitespace_becomes_empty():
    """Strings containing only whitespace should become empty"""
    frame = ar.from_dict({"text": ["   ", "\t\t", "\n\n"]})
    result = ar.normalize_whitespace(frame)
    df = ar.to_pandas(result)
    assert df["text"].iloc[0] == ""
    assert df["text"].iloc[1] == ""
    assert df["text"].iloc[2] == ""


def test_empty_string_unchanged():
    """Empty strings should remain empty"""
    frame = ar.from_dict({"text": ["", "hello"]})
    result = ar.normalize_whitespace(frame)
    df = ar.to_pandas(result)
    assert df["text"].iloc[0] == ""
    assert df["text"].iloc[1] == "hello"


def test_single_spaces_preserved():
    """Single spaces between words should be preserved"""
    frame = ar.from_dict({"text": ["hello world", "one two three"]})
    result = ar.normalize_whitespace(frame)
    df = ar.to_pandas(result)
    assert df["text"].iloc[0] == "hello world"
    assert df["text"].iloc[1] == "one two three"


def test_columns_parameter_single_column():
    """Only specified column should be processed"""
    frame = ar.from_dict({
        "col1": ["hello   world"],
        "col2": ["foo  bar"]
    })
    result = ar.normalize_whitespace(frame, columns=["col1"])
    df = ar.to_pandas(result)
    assert df["col1"].iloc[0] == "hello world"
    assert df["col2"].iloc[0] == "foo  bar"  # Unchanged


def test_columns_parameter_multiple_columns():
    """Multiple specified columns should be processed"""
    frame = ar.from_dict({
        "col1": ["hello   world"],
        "col2": ["foo  bar"],
        "col3": ["baz   qux"]
    })
    result = ar.normalize_whitespace(frame, columns=["col1", "col3"])
    df = ar.to_pandas(result)
    assert df["col1"].iloc[0] == "hello world"
    assert df["col2"].iloc[0] == "foo  bar"  # Unchanged
    assert df["col3"].iloc[0] == "baz qux"


def test_columns_none_processes_all_string_columns():
    """With columns=None, all string columns should be processed"""
    frame = ar.from_dict({
        "str1": ["hello   world"],
        "str2": ["foo  bar"],
        "num": [42]
    })
    result = ar.normalize_whitespace(frame, columns=None)
    df = ar.to_pandas(result)
    assert df["str1"].iloc[0] == "hello world"
    assert df["str2"].iloc[0] == "foo bar"
    assert df["num"].iloc[0] == 42


def test_default_processes_all_string_columns():
    """Default behavior should process all string columns"""
    frame = ar.from_dict({
        "str1": ["hello   world"],
        "str2": ["foo  bar"],
        "num": [42]
    })
    result = ar.normalize_whitespace(frame)
    df = ar.to_pandas(result)
    assert df["str1"].iloc[0] == "hello world"
    assert df["str2"].iloc[0] == "foo bar"
    assert df["num"].iloc[0] == 42


def test_numeric_columns_unaffected():
    """Non-string columns should not be affected"""
    frame = ar.from_dict({
        "text": ["hello   world"],
        "int": [123],
        "float": [1.23]
    })
    result = ar.normalize_whitespace(frame)
    df = ar.to_pandas(result)
    assert df["text"].iloc[0] == "hello world"
    assert df["int"].iloc[0] == 123
    assert df["float"].iloc[0] == 1.23


def test_multiple_rows():
    """Should handle multiple rows correctly"""
    frame = ar.from_dict({
        "text": ["hello   world", "foo  bar", "  baz  qux  "]
    })
    result = ar.normalize_whitespace(frame)
    df = ar.to_pandas(result)
    assert list(df["text"]) == ["hello world", "foo bar", "baz qux"]
