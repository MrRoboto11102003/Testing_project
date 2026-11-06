import pytest
import pandas as pd
import arnio as ar


def test_parse_bool_strings_basic():
    """Test basic boolean string parsing with defaults."""
    df = pd.DataFrame({"flag": ["true", "false", "yes", "no"]})
    frame = ar.from_pandas(df)
    result = ar.parse_bool_strings(frame)
    assert isinstance(result, ar.ArFrame)
    result_df = ar.to_pandas(result)
    assert len(result_df) == 4


def test_parse_bool_strings_numeric_strings():
    """Test parsing of numeric boolean strings like '1' and '0'."""
    df = pd.DataFrame({"is_active": ["1", "0", "1", "0"]})
    frame = ar.from_pandas(df)
    result = ar.parse_bool_strings(frame)
    result_df = ar.to_pandas(result)
    assert len(result_df) == 4


def test_parse_bool_strings_single_letters():
    """Test parsing of single-letter boolean representations (y, n, Y, N)."""
    df = pd.DataFrame({"flag": ["y", "n", "Y", "N"]})
    frame = ar.from_pandas(df)
    result = ar.parse_bool_strings(frame)
    result_df = ar.to_pandas(result)
    assert len(result_df) == 4


def test_parse_bool_strings_subset_single_column():
    """Test subset parameter restricts conversion to specified columns."""
    df = pd.DataFrame({
        "is_active": ["true", "false"],
        "description": ["yes", "no"]
    })
    frame = ar.from_pandas(df)
    result = ar.parse_bool_strings(frame, subset=["is_active"])
    result_df = ar.to_pandas(result)
    assert len(result_df) == 2
    assert set(result_df.columns) == {"is_active", "description"}


def test_parse_bool_strings_subset_multiple_columns():
    """Test subset parameter with multiple column names."""
    df = pd.DataFrame({
        "col1": ["true", "false"],
        "col2": ["yes", "no"],
        "col3": ["maybe", "unknown"]
    })
    frame = ar.from_pandas(df)
    result = ar.parse_bool_strings(frame, subset=["col1", "col2"])
    result_df = ar.to_pandas(result)
    assert len(result_df) == 2
    assert "col3" in result_df.columns


def test_parse_bool_strings_custom_true_values():
    """Test parsing with custom true_values parameter."""
    df = pd.DataFrame({"status": ["ok", "fail", "ok"]})
    frame = ar.from_pandas(df)
    result = ar.parse_bool_strings(frame, true_values={"ok"})
    result_df = ar.to_pandas(result)
    assert len(result_df) == 3


def test_parse_bool_strings_custom_false_values():
    """Test parsing with custom false_values parameter."""
    df = pd.DataFrame({"status": ["pass", "fail", "pass"]})
    frame = ar.from_pandas(df)
    result = ar.parse_bool_strings(frame, false_values={"fail"})
    result_df = ar.to_pandas(result)
    assert len(result_df) == 3


def test_parse_bool_strings_custom_true_and_false():
    """Test with both custom true_values and false_values."""
    df = pd.DataFrame({"state": ["on", "off", "on", "off"]})
    frame = ar.from_pandas(df)
    result = ar.parse_bool_strings(
        frame,
        true_values={"on"},
        false_values={"off"}
    )
    result_df = ar.to_pandas(result)
    assert len(result_df) == 4


def test_parse_bool_strings_unsupported_values_preserved():
    """Test that unsupported string values are preserved unchanged."""
    df = pd.DataFrame({"flag": ["true", "false", "maybe", "unknown"]})
    frame = ar.from_pandas(df)
    result = ar.parse_bool_strings(frame)
    result_df = ar.to_pandas(result)
    assert len(result_df) == 4


def test_parse_bool_strings_all_string_columns_by_default():
    """Test that all string/object columns are processed when subset is None."""
    df = pd.DataFrame({
        "flag1": ["true", "false"],
        "flag2": ["yes", "no"],
        "count": [1, 2]  # numeric column should remain unchanged
    })
    frame = ar.from_pandas(df)
    result = ar.parse_bool_strings(frame, subset=None)
    result_df = ar.to_pandas(result)
    assert result_df["count"].tolist() == [1, 2]


def test_parse_bool_strings_empty_frame():
    """Test behavior with an empty frame."""
    df = pd.DataFrame({"flag": pd.Series([], dtype=object)})
    frame = ar.from_pandas(df)
    result = ar.parse_bool_strings(frame)
    result_df = ar.to_pandas(result)
    assert len(result_df) == 0
    assert "flag" in result_df.columns


def test_parse_bool_strings_with_nulls():
    """Test handling of null/NA values in the data."""
    df = pd.DataFrame({"flag": ["true", None, "false"]})
    frame = ar.from_pandas(df)
    result = ar.parse_bool_strings(frame)
    result_df = ar.to_pandas(result)
    assert len(result_df) == 3


def test_parse_bool_strings_returns_arframe():
    """Verify that the function returns an ArFrame instance."""
    df = pd.DataFrame({"flag": ["true", "false"]})
    frame = ar.from_pandas(df)
    result = ar.parse_bool_strings(frame)
    assert isinstance(result, ar.ArFrame)


def test_parse_bool_strings_mixed_recognized_and_unsupported():
    """Test column with mix of recognized booleans and unsupported strings."""
    df = pd.DataFrame({"status": ["yes", "no", "maybe", "yes"]})
    frame = ar.from_pandas(df)
    result = ar.parse_bool_strings(frame)
    result_df = ar.to_pandas(result)
    assert len(result_df) == 4
