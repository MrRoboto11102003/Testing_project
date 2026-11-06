import pytest
import pandas as pd
import numpy as np
import arnio as ar


def test_basic_conversion_default_values():
    """Test basic conversion with default true/false values."""
    df = pd.DataFrame({
        "active": ["true", "false", "yes", "no"],
    })
    frame = ar.from_pandas(df)
    result = ar.parse_bool_strings(frame)
    df_result = ar.to_pandas(result)
    
    assert df_result["active"].tolist() == [True, False, True, False]


def test_default_values_all_variations():
    """Test all default true and false value variations."""
    df = pd.DataFrame({
        "col": ["true", "yes", "y", "1", "false", "no", "n", "0"],
    })
    frame = ar.from_pandas(df)
    result = ar.parse_bool_strings(frame)
    df_result = ar.to_pandas(result)
    
    expected = [True, True, True, True, False, False, False, False]
    assert df_result["col"].tolist() == expected


def test_custom_true_false_values():
    """Test with custom true_values and false_values."""
    df = pd.DataFrame({
        "col": ["yes_custom", "no_custom", "unknown"],
    })
    frame = ar.from_pandas(df)
    result = ar.parse_bool_strings(
        frame,
        true_values={"yes_custom"},
        false_values={"no_custom"},
    )
    df_result = ar.to_pandas(result)
    
    assert df_result["col"].tolist()[0] == True
    assert df_result["col"].tolist()[1] == False
    assert df_result["col"].tolist()[2] == "unknown"


def test_case_insensitivity():
    """Test that conversion is case insensitive."""
    df = pd.DataFrame({
        "col": ["TRUE", "False", "YES", "NO", "TrUe"],
    })
    frame = ar.from_pandas(df)
    result = ar.parse_bool_strings(frame)
    df_result = ar.to_pandas(result)
    
    assert df_result["col"].tolist() == [True, False, True, False, True]


def test_whitespace_handling():
    """Test that leading/trailing whitespace is handled."""
    df = pd.DataFrame({
        "col": ["  true  ", " false ", "yes\t", "\nno"],
    })
    frame = ar.from_pandas(df)
    result = ar.parse_bool_strings(frame)
    df_result = ar.to_pandas(result)
    
    assert df_result["col"].tolist() == [True, False, True, False]


def test_subset_parameter():
    """Test subset parameter applies to specific columns only."""
    df = pd.DataFrame({
        "active": ["true", "false"],
        "verified": ["true", "false"],
        "name": ["alice", "bob"],
    })
    frame = ar.from_pandas(df)
    result = ar.parse_bool_strings(frame, subset=["active"])
    df_result = ar.to_pandas(result)
    
    assert df_result["active"].tolist() == [True, False]
    assert df_result["verified"].tolist() == ["true", "false"]
    assert df_result["name"].tolist() == ["alice", "bob"]


def test_none_subset_applies_to_all_string_columns():
    """Test that None subset applies to all string columns."""
    df = pd.DataFrame({
        "active": ["true", "false"],
        "verified": ["yes", "no"],
        "count": [1, 2],
    })
    frame = ar.from_pandas(df)
    result = ar.parse_bool_strings(frame)
    df_result = ar.to_pandas(result)
    
    assert df_result["active"].tolist() == [True, False]
    assert df_result["verified"].tolist() == [True, False]
    assert df_result["count"].tolist() == [1, 2]


def test_mixed_values_preserves_unknown():
    """Test that unknown values are preserved unchanged."""
    df = pd.DataFrame({
        "col": ["true", "unknown", "false", "maybe"],
    })
    frame = ar.from_pandas(df)
    result = ar.parse_bool_strings(frame)
    df_result = ar.to_pandas(result)
    
    # Convert all to string for flexible comparison (mixed type columns)
    col_str = [str(v).lower() for v in df_result["col"].tolist()]
    assert "true" in col_str
    assert "false" in col_str
    assert "unknown" in col_str
    assert "maybe" in col_str


def test_nulls_preserved():
    """Test that null values are preserved."""
    df = pd.DataFrame({
        "col": ["true", None, "false", np.nan],
    })
    frame = ar.from_pandas(df)
    result = ar.parse_bool_strings(frame)
    df_result = ar.to_pandas(result)
    
    col_values = df_result["col"].tolist()
    assert col_values[0] is True or col_values[0] == True
    assert pd.isna(col_values[1])
    assert col_values[2] is False or col_values[2] == False
    assert pd.isna(col_values[3])


def test_error_true_values_bare_string():
    """Test that bare string for true_values raises TypeError."""
    df = pd.DataFrame({"col": ["true", "false"]})
    frame = ar.from_pandas(df)
    
    with pytest.raises(TypeError, match="bare string"):
        ar.parse_bool_strings(frame, true_values="true")


def test_error_false_values_bare_string():
    """Test that bare string for false_values raises TypeError."""
    df = pd.DataFrame({"col": ["true", "false"]})
    frame = ar.from_pandas(df)
    
    with pytest.raises(TypeError, match="bare string"):
        ar.parse_bool_strings(frame, false_values="false")


def test_error_overlap_after_normalization():
    """Test that overlapping true/false values raise ValueError."""
    df = pd.DataFrame({"col": ["true", "false"]})
    frame = ar.from_pandas(df)
    
    with pytest.raises(ValueError, match="overlap"):
        ar.parse_bool_strings(
            frame,
            true_values={"TRUE", "yes"},
            false_values={"true", "no"},
        )


def test_error_subset_column_not_found():
    """Test that subset with non-existent column raises ValueError."""
    df = pd.DataFrame({"col": ["true", "false"]})
    frame = ar.from_pandas(df)
    
    with pytest.raises(ValueError, match="not found"):
        ar.parse_bool_strings(frame, subset=["nonexistent"])


def test_error_non_string_in_true_values():
    """Test that non-string items in true_values raise TypeError."""
    df = pd.DataFrame({"col": ["true", "false"]})
    frame = ar.from_pandas(df)
    
    with pytest.raises(TypeError, match="must contain only strings"):
        ar.parse_bool_strings(frame, true_values={"true", 1})


def test_error_non_string_in_false_values():
    """Test that non-string items in false_values raise TypeError."""
    df = pd.DataFrame({"col": ["true", "false"]})
    frame = ar.from_pandas(df)
    
    with pytest.raises(TypeError, match="must contain only strings"):
        ar.parse_bool_strings(frame, false_values={"false", 1})
