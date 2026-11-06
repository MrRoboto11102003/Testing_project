import pytest
import pandas as pd
import arnio as ar
import numpy as np


def test_parse_bool_strings_defaults():
    """Test default true/false values conversion."""
    df = pd.DataFrame({
        "status": ["true", "false", "yes", "no", "y", "n", "1", "0"],
    })
    frame = ar.from_pandas(df)
    result = ar.parse_bool_strings(frame)
    result_df = ar.to_pandas(result)
    
    assert result_df["status"].tolist() == [True, False, True, False, True, False, True, False]


def test_parse_bool_strings_custom_values():
    """Test with custom true and false values."""
    df = pd.DataFrame({
        "active": ["enabled", "disabled", "other"],
    })
    frame = ar.from_pandas(df)
    result = ar.parse_bool_strings(
        frame,
        true_values={"enabled"},
        false_values={"disabled"},
    )
    result_df = ar.to_pandas(result)
    
    assert result_df["active"][0] is True
    assert result_df["active"][1] is False
    assert result_df["active"][2] == "other"


def test_parse_bool_strings_case_insensitive():
    """Test that conversion is case-insensitive."""
    df = pd.DataFrame({
        "flag": ["TRUE", "FALSE", "Yes", "NO", "Y", "n"],
    })
    frame = ar.from_pandas(df)
    result = ar.parse_bool_strings(frame)
    result_df = ar.to_pandas(result)
    
    assert all(isinstance(v, (bool, np.bool_)) for v in result_df["flag"])
    assert result_df["flag"].tolist() == [True, False, True, False, True, False]


def test_parse_bool_strings_whitespace_handling():
    """Test that whitespace is stripped from values."""
    df = pd.DataFrame({
        "value": [" true ", "  false  ", "\ttrue\n", " yes "],
    })
    frame = ar.from_pandas(df)
    result = ar.parse_bool_strings(frame)
    result_df = ar.to_pandas(result)
    
    assert result_df["value"].tolist() == [True, False, True, True]


def test_parse_bool_strings_subset():
    """Test subset parameter converts only specified columns."""
    df = pd.DataFrame({
        "bool_col": ["true", "false"],
        "other_col": ["true", "false"],
    })
    frame = ar.from_pandas(df)
    result = ar.parse_bool_strings(frame, subset=["bool_col"])
    result_df = ar.to_pandas(result)
    
    assert result_df["bool_col"].tolist() == [True, False]
    assert result_df["other_col"].tolist() == ["true", "false"]


def test_parse_bool_strings_overlapping_values_raises():
    """Test that overlapping true_values and false_values raise ValueError."""
    df = pd.DataFrame({"col": ["value"]})
    frame = ar.from_pandas(df)
    
    with pytest.raises(ValueError, match="true_values and false_values overlap"):
        ar.parse_bool_strings(
            frame,
            true_values={"yes", "true"},
            false_values={"no", "true"},
        )


def test_parse_bool_strings_true_values_string_raises():
    """Test that true_values as bare string raises TypeError."""
    df = pd.DataFrame({"col": ["true"]})
    frame = ar.from_pandas(df)
    
    with pytest.raises(TypeError, match="true_values must be a set"):
        ar.parse_bool_strings(frame, true_values="true")


def test_parse_bool_strings_false_values_string_raises():
    """Test that false_values as bare string raises TypeError."""
    df = pd.DataFrame({"col": ["false"]})
    frame = ar.from_pandas(df)
    
    with pytest.raises(TypeError, match="false_values must be a set"):
        ar.parse_bool_strings(frame, false_values="false")


def test_parse_bool_strings_true_values_invalid_item_type():
    """Test that non-string items in true_values raise TypeError."""
    df = pd.DataFrame({"col": ["value"]})
    frame = ar.from_pandas(df)
    
    with pytest.raises(TypeError, match="true_values must contain only strings"):
        ar.parse_bool_strings(frame, true_values={"true", 1})


def test_parse_bool_strings_false_values_invalid_item_type():
    """Test that non-string items in false_values raise TypeError."""
    df = pd.DataFrame({"col": ["value"]})
    frame = ar.from_pandas(df)
    
    with pytest.raises(TypeError, match="false_values must contain only strings"):
        ar.parse_bool_strings(frame, false_values={"false", 1})


def test_parse_bool_strings_preserves_unsupported():
    """Test that unsupported values are preserved unchanged."""
    df = pd.DataFrame({
        "mixed": ["true", "false", "maybe", "unknown", None],
    })
    frame = ar.from_pandas(df)
    result = ar.parse_bool_strings(frame)
    result_df = ar.to_pandas(result)
    
    assert result_df["mixed"][0] is True
    assert result_df["mixed"][1] is False
    assert result_df["mixed"][2] == "maybe"
    assert result_df["mixed"][3] == "unknown"
    assert pd.isna(result_df["mixed"][4])


def test_parse_bool_strings_all_string_columns_default():
    """Test that all string columns are converted when subset is None."""
    df = pd.DataFrame({
        "col1": ["yes", "no"],
        "col2": ["true", "false"],
        "col3": [1, 2],
    })
    frame = ar.from_pandas(df)
    result = ar.parse_bool_strings(frame)
    result_df = ar.to_pandas(result)
    
    assert result_df["col1"].tolist() == [True, False]
    assert result_df["col2"].tolist() == [True, False]
    assert result_df["col3"].tolist() == [1, 2]


def test_parse_bool_strings_invalid_column_in_subset():
    """Test that invalid column in subset raises ValueError."""
    df = pd.DataFrame({"col": ["true"]})
    frame = ar.from_pandas(df)
    
    with pytest.raises(ValueError, match="Columns not found in frame"):
        ar.parse_bool_strings(frame, subset=["nonexistent"])


def test_parse_bool_strings_with_list():
    """Test that list works for true_values and false_values."""
    df = pd.DataFrame({
        "col": ["enabled", "disabled"],
    })
    frame = ar.from_pandas(df)
    result = ar.parse_bool_strings(
        frame,
        true_values=["enabled"],
        false_values=["disabled"],
    )
    result_df = ar.to_pandas(result)
    assert result_df["col"].tolist() == [True, False]


def test_parse_bool_strings_with_tuple():
    """Test that tuple works for true_values and false_values."""
    df = pd.DataFrame({
        "col": ["enabled", "disabled"],
    })
    frame = ar.from_pandas(df)
    result = ar.parse_bool_strings(
        frame,
        true_values=("enabled",),
        false_values=("disabled",),
    )
    result_df = ar.to_pandas(result)
    assert result_df["col"].tolist() == [True, False]


def test_parse_bool_strings_mixed_column_types():
    """Test that only string/object columns are processed when subset is None."""
    df = pd.DataFrame({
        "string_col": ["yes", "no"],
        "int_col": [1, 2],
        "float_col": [1.5, 2.5],
    })
    frame = ar.from_pandas(df)
    result = ar.parse_bool_strings(frame)
    result_df = ar.to_pandas(result)
    
    # Only string column should be processed
    assert result_df["string_col"].tolist() == [True, False]
    assert result_df["int_col"].tolist() == [1, 2]
    assert result_df["float_col"].tolist() == [1.5, 2.5]
