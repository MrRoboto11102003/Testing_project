import pytest
import pandas as pd
import numpy as np
from arnio import ArFrame
from arnio.cleaning import coalesce_columns, standardize_missing_tokens


def test_coalesce_columns_basic():
    """Test basic coalescing of non-null values."""
    df = pd.DataFrame({"a": [1, None, 2], "b": [None, 3, None]})
    result = coalesce_columns(df, subset=["a", "b"], output_column="c")
    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(result, pd.DataFrame({"c": [1, 3, 2]}))


def test_coalesce_columns_with_arframe():
    """Test coalesce_columns with ArFrame input."""
    af = ArFrame.from_pandas(df=pd.DataFrame({"a": [1, None, 2], "b": [None, 3, None]}))
    result = coalesce_columns(af, subset=["a", "b"], output_column="c")
    assert isinstance(result, ArFrame)
    pd.testing.assert_frame_equal(ar.to_pandas(result), pd.DataFrame({"c": [1, 3, 2]}))


def test_coalesce_columns_output_exists():
    """Test that coalesce_columns raises error when output column exists."""
    df = pd.DataFrame({"a": [1, None, 2], "b": [None, 3, None], "c": ["x", "y", "z"]})
    with pytest.raises(ValueError, match="Output column 'c' already exists"):
        coalesce_columns(df, subset=["a", "b"], output_column="c")


def test_coalesce_columns_empty_subset():
    """Test that coalesce_columns raises error for empty subset."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError, match="subset must contain at least one column"):
        coalesce_columns(df, subset=[], output_column="c")


def test_coalesce_columns_missing_column():
    """Test that coalesce_columns raises error for missing columns."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(KeyError, match="Missing columns"):
        coalesce_columns(df, subset=["b"], output_column="c")


def test_coalesce_columns_non_string_subset():
    """Test that coalesce_columns raises error for non-string subset."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(TypeError):
        coalesce_columns(df, subset=[1, 2], output_column="c")


def test_coalesce_columns_empty_output_name():
    """Test that coalesce_columns raises error for empty output column name."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError, match="output_column must be a non-empty string"):
        coalesce_columns(df, subset=["a"], output_column="")


def test_coalesce_columns_all_null():
    """Test coalesce_columns when all values are null."""
    df = pd.DataFrame({"a": [None, None, None], "b": [None, None, None]})
    result = coalesce_columns(df, subset=["a", "b"], output_column="c")
    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(result, pd.DataFrame({"c": [np.nan, np.nan, np.nan]}))


def test_coalesce_columns_multiple_sources():
    """Test coalesce_columns with multiple source columns."""
    df = pd.DataFrame({
        "a": ["x", None, "z"],
        "b": [None, "y", None],
        "c": ["p", "q", None]
    })
    result = coalesce_columns(df, subset=["a", "b", "c"], output_column="d")
    assert isinstance(result, pd.DataFrame)
    expected = pd.DataFrame({"d": ["x", "y", "z"]})
    pd.testing.assert_frame_equal(result, expected)


def test_standardize_missing_tokens_basic():
    """Test basic standardization of missing tokens."""
    df = pd.DataFrame({"value": ["N/A", "NA", "kept"]})
    result = standardize_missing_tokens(df)
    assert isinstance(result, pd.DataFrame)
    expected = pd.DataFrame({"value": [np.nan, np.nan, "kept"]})
    pd.testing.assert_frame_equal(result, expected)


def test_standardize_missing_tokens_custom_tokens():
    """Test custom token list for standardization."""
    df = pd.DataFrame({"value": ["MISSING", "n/a", "kept"]})
    result = standardize_missing_tokens(df, tokens=["MISSING", "n/a"])
    assert isinstance(result, pd.DataFrame)
    expected = pd.DataFrame({"value": [np.nan, np.nan, "kept"]})
    pd.testing.assert_frame_equal(result, expected)


def test_standardize_missing_tokens_case_insensitive():
    """Test case-insensitive token matching."""
    df = pd.DataFrame({"value": ["N/A", "NA", "n/a", "Kept"]})
    result = standardize_missing_tokens(df)
    assert isinstance(result, pd.DataFrame)
    expected = pd.DataFrame({"value": [np.nan, np.nan, np.nan, "Kept"]})
    pd.testing.assert_frame_equal(result, expected)


def test_standardize_missing_tokens_subset():
    """Test standardize_missing_tokens with subset parameter."""
    df = pd.DataFrame({
        "col1": ["N/A", "kept"],
        "col2": ["ignored", "kept"]
    })
    result = standardize_missing_tokens(df, subset=["col1"])
    assert isinstance(result, pd.DataFrame)
    expected = pd.DataFrame({"col1": [np.nan, "kept"], "col2": ["ignored", "kept"]})
    pd.testing.assert_frame_equal(result, expected)


def test_standardize_missing_tokens_non_string_subset():
    """Test that standardize_missing_tokens raises error for non-string subset."""
    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(TypeError):
        standardize_missing_tokens(df, subset=1)


def test_standardize_missing_tokens_non_list_tokens():
    """Test that standardize_missing_tokens raises error for non-list tokens."""
    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(TypeError):
        standardize_missing_tokens(df, tokens="string")


def test_standardize_missing_tokens_non_string_token_item():
    """Test that standardize_missing_tokens raises error for non-string token items."""
    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(TypeError):
        standardize_missing_tokens(df, tokens=[1, "N/A"])


def test_standardize_missing_tokens_whitespace_trim():
    """Test that whitespace is trimmed before checking tokens."""
    df = pd.DataFrame({"value": [" N/A ", "\tNaN\t", "kept"]})
    result = standardize_missing_tokens(df)
    assert isinstance(result, pd.DataFrame)
    expected = pd.DataFrame({"value": [np.nan, np.nan, "kept"]})
    pd.testing.assert_frame_equal(result, expected)


def test_standardize_missing_tokens_preserves_original_whitespace():
    """Test that non-matching values preserve their original whitespace."""
    df = pd.DataFrame({"value": [" N/A ", "  kept  "]})
    result = standardize_missing_tokens(df)
    assert isinstance(result, pd.DataFrame)
    expected = pd.DataFrame({"value": [np.nan, "  kept  "]})
    pd.testing.assert_frame_equal(result, expected)
