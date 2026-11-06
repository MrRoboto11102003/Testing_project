import pytest
import pandas as pd
import numpy as np
import arnio as ar


def test_default_tokens_all_columns():
    """Default tokens should be replaced with NaN across all columns."""
    df = pd.DataFrame({
        "col1": ["value", "N/A", "NA", "none"],
        "col2": ["null", "-", "?", "data"]
    })
    frame = ar.from_pandas(df)
    result = ar.standardize_missing_tokens(frame)
    result_df = ar.to_pandas(result)
    
    assert result_df.loc[0, "col1"] == "value"
    assert pd.isna(result_df.loc[1, "col1"])
    assert pd.isna(result_df.loc[2, "col1"])
    assert pd.isna(result_df.loc[3, "col1"])
    assert pd.isna(result_df.loc[0, "col2"])
    assert pd.isna(result_df.loc[1, "col2"])
    assert pd.isna(result_df.loc[2, "col2"])
    assert result_df.loc[3, "col2"] == "data"


def test_case_insensitive_matching():
    """Token matching should be case-insensitive."""
    df = pd.DataFrame({
        "col": ["NaN", "NAN", "None", "NONE", "nil", "NIL", "n/a"]
    })
    frame = ar.from_pandas(df)
    result = ar.standardize_missing_tokens(frame)
    result_df = ar.to_pandas(result)
    
    for i in range(7):
        assert pd.isna(result_df.loc[i, "col"])


def test_whitespace_trimming():
    """Whitespace around tokens should be trimmed before matching."""
    df = pd.DataFrame({
        "col": ["  N/A  ", "\tNA\t", " none ", "\n-\n", "   ?   "]
    })
    frame = ar.from_pandas(df)
    result = ar.standardize_missing_tokens(frame)
    result_df = ar.to_pandas(result)
    
    for i in range(5):
        assert pd.isna(result_df.loc[i, "col"])


def test_custom_tokens_replaces_defaults():
    """Custom tokens should replace defaults."""
    df = pd.DataFrame({
        "col": ["MISSING", "UNKNOWN", "value", "N/A", "none"]
    })
    frame = ar.from_pandas(df)
    result = ar.standardize_missing_tokens(frame, tokens=["MISSING", "UNKNOWN"])
    result_df = ar.to_pandas(result)
    
    assert pd.isna(result_df.loc[0, "col"])
    assert pd.isna(result_df.loc[1, "col"])
    assert result_df.loc[2, "col"] == "value"
    assert result_df.loc[3, "col"] == "N/A"
    assert result_df.loc[4, "col"] == "none"


def test_subset_single_column():
    """subset parameter should limit replacement to specified columns."""
    df = pd.DataFrame({
        "col1": ["N/A", "value"],
        "col2": ["N/A", "other"]
    })
    frame = ar.from_pandas(df)
    result = ar.standardize_missing_tokens(frame, subset=["col1"])
    result_df = ar.to_pandas(result)
    
    assert pd.isna(result_df.loc[0, "col1"])
    assert result_df.loc[1, "col1"] == "value"
    assert result_df.loc[0, "col2"] == "N/A"


def test_subset_multiple_columns():
    """subset should work with multiple columns."""
    df = pd.DataFrame({
        "col1": ["N/A", "value"],
        "col2": ["none", "data"],
        "col3": ["null", "info"]
    })
    frame = ar.from_pandas(df)
    result = ar.standardize_missing_tokens(frame, subset=["col1", "col2"])
    result_df = ar.to_pandas(result)
    
    assert pd.isna(result_df.loc[0, "col1"])
    assert pd.isna(result_df.loc[0, "col2"])
    assert result_df.loc[0, "col3"] == "null"


def test_non_string_values_preserved():
    """Non-string values should not be affected."""
    df = pd.DataFrame({
        "col": [1, 2.5, True, "N/A"]
    })
    frame = ar.from_pandas(df)
    result = ar.standardize_missing_tokens(frame)
    result_df = ar.to_pandas(result)
    
    assert result_df.loc[0, "col"] == 1
    assert result_df.loc[1, "col"] == 2.5
    assert result_df.loc[2, "col"] is True
    assert pd.isna(result_df.loc[3, "col"])


def test_empty_string_is_default_token():
    """Empty string should be treated as a token and replaced."""
    df = pd.DataFrame({
        "col": ["", "value", "  "]
    })
    frame = ar.from_pandas(df)
    result = ar.standardize_missing_tokens(frame)
    result_df = ar.to_pandas(result)
    
    assert pd.isna(result_df.loc[0, "col"])
    assert result_df.loc[1, "col"] == "value"
    assert result_df.loc[2, "col"] == "  "


def test_preserves_whitespace_non_matching():
    """Non-matching values should preserve original whitespace."""
    df = pd.DataFrame({
        "col": ["  data  ", "\tvalue\t", " regular "]
    })
    frame = ar.from_pandas(df)
    result = ar.standardize_missing_tokens(frame)
    result_df = ar.to_pandas(result)
    
    assert result_df.loc[0, "col"] == "  data  "
    assert result_df.loc[1, "col"] == "\tvalue\t"
    assert result_df.loc[2, "col"] == " regular "


def test_subset_string_raises_type_error():
    """subset as string raises TypeError with helpful message."""
    df = pd.DataFrame({"col": ["N/A"]})
    frame = ar.from_pandas(df)
    
    with pytest.raises(TypeError, match="subset must be a list"):
        ar.standardize_missing_tokens(frame, subset="col")


def test_tokens_string_raises_type_error():
    """tokens as string raises TypeError."""
    df = pd.DataFrame({"col": ["value"]})
    frame = ar.from_pandas(df)
    
    with pytest.raises(TypeError, match="tokens must be a list of strings"):
        ar.standardize_missing_tokens(frame, tokens="N/A")


def test_tokens_not_list_raises_type_error():
    """tokens as non-list raises TypeError."""
    df = pd.DataFrame({"col": ["value"]})
    frame = ar.from_pandas(df)
    
    with pytest.raises(TypeError, match="tokens must be a list"):
        ar.standardize_missing_tokens(frame, tokens=("N/A", "NA"))


def test_tokens_non_string_item_raises_type_error():
    """tokens with non-string items raises TypeError."""
    df = pd.DataFrame({"col": ["value"]})
    frame = ar.from_pandas(df)
    
    with pytest.raises(TypeError, match="tokens must be a list of strings"):
        ar.standardize_missing_tokens(frame, tokens=["N/A", 123])


def test_unknown_subset_column_raises_value_error():
    """Unknown column in subset raises ValueError."""
    df = pd.DataFrame({"col1": ["value"]})
    frame = ar.from_pandas(df)
    
    with pytest.raises(ValueError, match="Unknown columns in subset"):
        ar.standardize_missing_tokens(frame, subset=["nonexistent"])


def test_works_with_pandas_dataframe():
    """Function works directly with pandas DataFrames."""
    df = pd.DataFrame({"col": ["N/A", "value"]})
    result = ar.standardize_missing_tokens(df)
    
    assert isinstance(result, pd.DataFrame)
    assert pd.isna(result.loc[0, "col"])
    assert result.loc[1, "col"] == "value"
