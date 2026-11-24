import pytest
import numpy as np
import pandas as pd
import arnio as ar
import re


def test_coalesce_columns_basic():
    """Coalesce returns first non-null value from subset in order."""
    df = pd.DataFrame({"a": [1.0, np.nan, np.nan], "b": [10.0, 2.0, np.nan]})
    result = ar.coalesce_columns(df, subset=["a", "b"], output_column="c")
    assert result["c"].tolist()[:2] == [1.0, 2.0]
    assert pd.isna(result["c"].iloc[2])
    assert list(df.columns) == ["a", "b"]


def test_coalesce_columns_validation():
    """Coalesce validates empty subset and duplicate output columns."""
    df = pd.DataFrame({"a": [1], "b": [2]})
    with pytest.raises(ValueError, match="at least one column"):
        ar.coalesce_columns(df, subset=[])
    with pytest.raises(ValueError, match="already exists"):
        ar.coalesce_columns(df, subset=["a"], output_column="a")


def test_combine_columns_basic():
    """Combine joins values with separator."""
    df = pd.DataFrame({"a": ["x"], "b": ["y"]})
    result = ar.combine_columns(df, subset=["a", "b"], separator=" ", output_column="c")
    assert result["c"].tolist() == ["x y"]
    assert list(df.columns) == ["a", "b"]


def test_combine_columns_null_row():
    """Combine converts all-null rows to NA."""
    df = pd.DataFrame({"a": [None, "x"], "b": [None, "y"]})
    result = ar.combine_columns(df, subset=["a", "b"], separator="-")
    assert pd.isna(result.iloc[0, 2])
    assert isinstance(result.iloc[1, 2], str)


def test_drop_columns_matching_pattern():
    """Drop columns matching regex pattern."""
    df = pd.DataFrame({"temp_a": [1], "b": [2], "temp_c": [3]})
    result = ar.drop_columns_matching(df, "^temp_")
    assert list(result.columns) == ["b"]
    assert list(df.columns) == ["temp_a", "b", "temp_c"]


def test_drop_columns_matching_all_raises():
    """Dropping all columns raises ValueError."""
    df = pd.DataFrame({"a": [1], "b": [2]})
    with pytest.raises(ValueError, match="at least one column must remain"):
        ar.drop_columns_matching(df, "")


def test_drop_constant_columns_basic():
    """Drop columns with exactly one unique value."""
    df = pd.DataFrame({"a": [1, 1], "b": [1, 2], "c": ["x", "x"]})
    result = ar.drop_constant_columns(df)
    assert list(result.columns) == ["b"]
    assert df.shape == (2, 3)


def test_drop_constant_columns_null_as_value():
    """Nulls count as a unique value."""
    df = pd.DataFrame({"a": [1, 1, None], "b": [None, None]})
    result = ar.drop_constant_columns(df)
    assert "a" in result.columns
    assert "b" not in result.columns


def test_drop_nulls_subset():
    """Drop rows with nulls in specified subset columns."""
    df_in = pd.DataFrame({"a": [1.0, None, 3.0], "b": ["x", "y", None]})
    frame = ar.from_pandas(df_in)
    result = ar.drop_nulls(frame, subset=["a"])
    result_pd = ar.to_pandas(result)
    assert len(result_pd) == 2
    assert result_pd["a"].tolist() == [1.0, 3.0]
    assert pd.isna(result_pd["b"].iloc[1])


def test_filter_rows_basic():
    """Filter rows with comparison operators."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    result_gt = ar.filter_rows(df, "a", ">", 1)
    assert result_gt["a"].tolist() == [2, 3]
    result_eq = ar.filter_rows(df, "b", "==", "y")
    assert len(result_eq) == 1
    assert df.shape == (3, 2)


def test_filter_rows_validation():
    """Filter rows validates operator and column."""
    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(ValueError, match="Unsupported operator"):
        ar.filter_rows(df, "a", "=", 1)
    with pytest.raises(ValueError, match="Unknown column"):
        ar.filter_rows(df, "missing", ">", 1)


def test_keep_rows_with_nulls_subset():
    """Keep only rows with nulls in subset columns."""
    df_in = pd.DataFrame({"a": [1.0, None, 3.0], "b": ["x", "y", None]})
    frame = ar.from_pandas(df_in)
    result = ar.keep_rows_with_nulls(frame, subset=["a"])
    result_pd = ar.to_pandas(result)
    assert len(result_pd) == 1
    assert pd.isna(result_pd["a"].iloc[0])


def test_replace_values_column():
    """Replace values in specific column."""
    df = pd.DataFrame({"a": ["x", "y"], "b": ["x", "q"]})
    result = ar.replace_values(df, {"x": "X"}, column="a")
    assert result["a"].tolist() == ["X", "y"]
    assert result["b"].tolist() == ["x", "q"]


def test_replace_values_null_key():
    """Null key replaces pre-existing nulls."""
    df = pd.DataFrame({"a": ["x", None, "y"]})
    result = ar.replace_values(df, {None: "MISSING"})
    assert result["a"].tolist() == ["x", "MISSING", "y"]


def test_safe_divide_columns():
    """Divide columns safely with fill_value for zero/null denominator."""
    df = pd.DataFrame({"n": [10.0, 5.0, 3.0, None], "d": [2.0, 0.0, None, 1.0]})
    result = ar.safe_divide_columns(df, "n", "d", "r", fill_value=0.0)
    assert result["r"].tolist() == [5.0, 0.0, 0.0, 0.0]
    assert "r" not in df.columns


def test_standardize_missing_tokens():
    """Default tokens replaced with NaN."""
    df = pd.DataFrame({"v": [" NULL ", "\tNaN\t", "kept", "N/A"]})
    result = ar.standardize_missing_tokens(df)
    assert pd.isna(result["v"].iloc[0])
    assert pd.isna(result["v"].iloc[1])
    assert result["v"].iloc[2] == "kept"
    assert pd.isna(result["v"].iloc[3])
    assert df["v"].iloc[0] == " NULL "
