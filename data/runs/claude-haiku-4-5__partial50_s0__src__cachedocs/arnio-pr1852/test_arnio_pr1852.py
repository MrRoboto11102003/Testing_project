import pytest
import pandas as pd
import numpy as np
import arnio as ar


def test_coalesce_columns_first_non_null():
    """Test coalesce_columns selects first non-null value from columns."""
    df = pd.DataFrame({
        "a": [1, None, 3],
        "b": [None, 2, None],
        "c": [10, 20, 30]
    })
    frame = ar.from_pandas(df)
    result = ar.coalesce_columns(frame, subset=["a", "b", "c"], output_column="first")
    result_df = ar.to_pandas(result)
    assert result_df["first"].tolist() == [1, 2, 3]


def test_combine_columns_with_separator():
    """Test combine_columns joins columns with custom separator."""
    df = pd.DataFrame({
        "first": ["Alice", "Bob"],
        "last": ["Smith", "Jones"]
    })
    frame = ar.from_pandas(df)
    result = ar.combine_columns(frame, subset=["first", "last"], separator=" ", output_column="full_name")
    result_df = ar.to_pandas(result)
    assert result_df["full_name"].tolist() == ["Alice Smith", "Bob Jones"]


def test_drop_columns_matching_regex():
    """Test drop_columns_matching removes columns matching regex pattern."""
    df = pd.DataFrame({"temp_1": [1], "temp_2": [2], "keep": [3]})
    frame = ar.from_pandas(df)
    result = ar.drop_columns_matching(frame, "^temp_")
    result_df = ar.to_pandas(result)
    assert list(result_df.columns) == ["keep"]


def test_drop_columns_matching_all_raises():
    """Test drop_columns_matching raises when pattern matches all columns."""
    df = pd.DataFrame({"a": [1], "b": [2]})
    frame = ar.from_pandas(df)
    with pytest.raises(ValueError, match="matches all columns"):
        ar.drop_columns_matching(frame, ".*")


def test_drop_constant_columns_removes_single_value():
    """Test drop_constant_columns removes columns with one unique value."""
    df = pd.DataFrame({
        "constant": [1, 1, 1],
        "varying": [1, 2, 3]
    })
    frame = ar.from_pandas(df)
    result = ar.drop_constant_columns(frame)
    result_df = ar.to_pandas(result)
    assert list(result_df.columns) == ["varying"]


def test_drop_nulls_removes_rows():
    """Test drop_nulls removes rows with any null values."""
    df = pd.DataFrame({
        "a": [1, 2, None],
        "b": [4, None, 6]
    })
    frame = ar.from_pandas(df)
    result = ar.drop_nulls(frame)
    result_df = ar.to_pandas(result)
    assert len(result_df) == 1


def test_drop_nulls_subset():
    """Test drop_nulls respects subset parameter for targeted null removal."""
    df = pd.DataFrame({
        "a": [1, 2, None],
        "b": [4, None, 6]
    })
    frame = ar.from_pandas(df)
    result = ar.drop_nulls(frame, subset=["a"])
    result_df = ar.to_pandas(result)
    assert len(result_df) == 2


def test_filter_rows_comparison_operators():
    """Test filter_rows with various comparison operators."""
    df = pd.DataFrame({"age": [15, 18, 25]})
    frame = ar.from_pandas(df)
    
    # Test > operator
    result = ar.filter_rows(frame, column="age", op=">", value=18)
    assert len(ar.to_pandas(result)) == 1
    
    # Test == operator
    result = ar.filter_rows(frame, column="age", op="==", value=18)
    assert len(ar.to_pandas(result)) == 1


def test_filter_rows_invalid_operator_raises():
    """Test filter_rows raises ValueError for unsupported operators."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    frame = ar.from_pandas(df)
    with pytest.raises(ValueError, match="Unsupported operator"):
        ar.filter_rows(frame, column="a", op="~", value=2)


def test_keep_rows_with_nulls_isolates_incomplete_data():
    """Test keep_rows_with_nulls returns only rows containing at least one null."""
    df = pd.DataFrame({
        "a": [1, 2, None],
        "b": [4, None, 6]
    })
    frame = ar.from_pandas(df)
    result = ar.keep_rows_with_nulls(frame)
    result_df = ar.to_pandas(result)
    assert len(result_df) == 2
    assert pd.isna(result_df["a"].iloc[1]) or pd.isna(result_df["b"].iloc[0])


def test_replace_values_in_column():
    """Test replace_values replaces values in a specific column."""
    df = pd.DataFrame({"status": ["yes", "no", "yes"]})
    frame = ar.from_pandas(df)
    result = ar.replace_values(frame, {"yes": "Y", "no": "N"}, column="status")
    result_df = ar.to_pandas(result)
    assert result_df["status"].tolist() == ["Y", "N", "Y"]


def test_replace_values_null_key():
    """Test replace_values handles None key to replace existing nulls."""
    df = pd.DataFrame({"val": [1.0, 2.0, np.nan]})
    frame = ar.from_pandas(df)
    result = ar.replace_values(frame, {None: -1}, column="val")
    result_df = ar.to_pandas(result)
    assert result_df["val"].iloc[2] == -1


def test_safe_divide_columns_normal_division():
    """Test safe_divide_columns with normal numeric values."""
    df = pd.DataFrame({
        "revenue": [100.0, 200.0],
        "cost": [10.0, 20.0]
    })
    frame = ar.from_pandas(df)
    result = ar.safe_divide_columns(frame, numerator="revenue", denominator="cost", output_column="ratio")
    result_df = ar.to_pandas(result)
    assert result_df["ratio"].tolist() == [10.0, 10.0]


def test_safe_divide_columns_by_zero_uses_fill_value():
    """Test safe_divide_columns uses fill_value when denominator is zero or null."""
    df = pd.DataFrame({
        "revenue": [100.0, 200.0],
        "cost": [10.0, 0.0]
    })
    frame = ar.from_pandas(df)
    result = ar.safe_divide_columns(frame, numerator="revenue", denominator="cost", 
                                   output_column="ratio", fill_value=-1.0)
    result_df = ar.to_pandas(result)
    assert result_df["ratio"].iloc[1] == -1.0


def test_standardize_missing_tokens_default_and_custom():
    """Test standardize_missing_tokens with default and custom token lists."""
    # Test default tokens
    df = pd.DataFrame({
        "val": ["data", "N/A", "null", "NA"]
    })
    frame = ar.from_pandas(df)
    result = ar.standardize_missing_tokens(frame)
    result_df = ar.to_pandas(result)
    assert pd.isna(result_df["val"].iloc[1])
    assert pd.isna(result_df["val"].iloc[2])
    assert result_df["val"].iloc[0] == "data"
    
    # Test custom tokens
    df2 = pd.DataFrame({"val": ["data", "MISSING", "other"]})
    frame2 = ar.from_pandas(df2)
    result2 = ar.standardize_missing_tokens(frame2, tokens=["MISSING"])
    result_df2 = ar.to_pandas(result2)
    assert pd.isna(result_df2["val"].iloc[1])
    assert result_df2["val"].iloc[0] == "data"
