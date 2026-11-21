import pytest
import pandas as pd
import numpy as np
import re
import arnio as ar


class TestCoalesceColumns:
    """Test arnio.cleaning.coalesce_columns"""

    def test_coalesce_columns_basic_first_non_null(self):
        """First non-null value in subset order wins"""
        df = pd.DataFrame({"a": [1.0, np.nan, np.nan], "b": [10.0, 2.0, np.nan]})
        result = ar.coalesce_columns(df, subset=["a", "b"], output_column="c")
        assert result["c"].tolist()[:2] == [1.0, 2.0]
        assert pd.isna(result["c"].iloc[2])

    def test_coalesce_columns_subset_order_matters(self):
        """Subset order determines priority"""
        df = pd.DataFrame({"a": [1.0, 10.0], "b": [9.0, 2.0]})
        result1 = ar.coalesce_columns(df, subset=["a", "b"], output_column="c")
        result2 = ar.coalesce_columns(df, subset=["b", "a"], output_column="c")
        assert result1["c"].tolist() == [1.0, 10.0]
        assert result2["c"].tolist() == [9.0, 2.0]

    def test_coalesce_columns_error_duplicate_names(self):
        """Duplicate subset names raise ValueError"""
        df = pd.DataFrame({"a": [1.0], "b": [2.0]})
        with pytest.raises(ValueError, match="duplicate column names"):
            ar.coalesce_columns(df, subset=["a", "a"], output_column="c")

    def test_coalesce_columns_error_missing_column(self):
        """Missing subset column raises KeyError"""
        df = pd.DataFrame({"a": [1.0], "b": [2.0]})
        with pytest.raises(KeyError, match="Missing columns for coalesce_columns"):
            ar.coalesce_columns(df, subset=["z"], output_column="c")

    def test_coalesce_columns_error_output_exists(self):
        """Existing output_column raises ValueError"""
        df = pd.DataFrame({"a": [1.0], "b": [2.0]})
        with pytest.raises(ValueError, match="already exists"):
            ar.coalesce_columns(df, subset=["a"], output_column="b")


class TestCombineColumns:
    """Test arnio.cleaning.combine_columns"""

    def test_combine_columns_basic_join(self):
        """Columns joined with separator"""
        df = pd.DataFrame({"a": ["x"], "b": ["y"]})
        result = ar.combine_columns(df, subset=["a", "b"], separator="-", output_column="c")
        assert result["c"].iloc[0] == "x-y"

    def test_combine_columns_subset_none_uses_all(self):
        """subset=None uses all columns in order"""
        df = pd.DataFrame({"a": ["x"], "b": ["y"], "c": ["z"]})
        result = ar.combine_columns(df, subset=None, separator="|", output_column="d")
        assert result["d"].iloc[0] == "x|y|z"

    def test_combine_columns_null_handling(self):
        """Row with all nulls becomes NA"""
        df = pd.DataFrame({"a": ["x", None], "b": [None, None]})
        result = ar.combine_columns(df, subset=["a", "b"], separator="-", output_column="c")
        assert result["c"].iloc[0] == "x-"
        assert pd.isna(result["c"].iloc[1])

    def test_combine_columns_error_empty_separator_not_error(self):
        """Empty separator is allowed"""
        df = pd.DataFrame({"a": ["x"], "b": ["y"]})
        result = ar.combine_columns(df, subset=["a", "b"], separator="", output_column="c")
        assert result["c"].iloc[0] == "xy"

    def test_combine_columns_error_bad_separator_type(self):
        """Non-string separator raises TypeError"""
        df = pd.DataFrame({"a": ["x"], "b": ["y"]})
        with pytest.raises(TypeError, match="separator must be a string"):
            ar.combine_columns(df, subset=["a", "b"], separator=5, output_column="c")


class TestDropColumnsMatching:
    """Test arnio.cleaning.drop_columns_matching"""

    def test_drop_columns_matching_basic(self):
        """Regex pattern drops matching columns"""
        df = pd.DataFrame({"temp_a": [1], "b": [2], "temp_c": [3]})
        result = ar.drop_columns_matching(df, "^temp_")
        assert list(result.columns) == ["b"]

    def test_drop_columns_matching_substring_match(self):
        """Substring match anywhere in name"""
        df = pd.DataFrame({"my_temp": [1], "temp_x": [2], "keep": [3]})
        result = ar.drop_columns_matching(df, "temp")
        assert list(result.columns) == ["keep"]

    def test_drop_columns_matching_case_sensitive(self):
        """Matching is case-sensitive"""
        df = pd.DataFrame({"TEMP": [1], "temp": [2], "keep": [3]})
        result = ar.drop_columns_matching(df, "^temp$")
        assert list(result.columns) == ["TEMP", "keep"]

    def test_drop_columns_matching_error_matches_all(self):
        """Matching all columns raises ValueError"""
        df = pd.DataFrame({"a": [1], "b": [2]})
        with pytest.raises(ValueError, match="At least one column must remain"):
            ar.drop_columns_matching(df, "")


class TestDropConstantColumns:
    """Test arnio.cleaning.drop_constant_columns"""

    def test_drop_constant_columns_basic(self):
        """Columns with one unique value are dropped"""
        df = pd.DataFrame({"a": [1, 1, 1], "b": [1, 2, 3]})
        result = ar.drop_constant_columns(df)
        assert list(result.columns) == ["b"]

    def test_drop_constant_columns_nulls_count_as_values(self):
        """All-null column is dropped; partial null column kept"""
        df = pd.DataFrame({"a": [None, None], "b": [1, 1], "c": [1, None]})
        result = ar.drop_constant_columns(df)
        assert list(result.columns) == ["c"]

    def test_drop_constant_columns_zero_rows(self):
        """Zero-row frame keeps all columns"""
        df = pd.DataFrame({"a": pd.Series([], dtype="int64"), "b": pd.Series([], dtype="int64")})
        result = ar.drop_constant_columns(df)
        assert list(result.columns) == ["a", "b"]


class TestDropNulls:
    """Test arnio.cleaning.drop_nulls"""

    def test_drop_nulls_basic(self):
        """Rows with any null are dropped"""
        f = ar.from_pandas(pd.DataFrame({"a": [1.0, None, 3.0], "b": ["x", "y", None]}))
        result = ar.drop_nulls(f)
        pdf = ar.to_pandas(result)
        assert len(pdf) == 1
        assert pdf["a"].iloc[0] == 1.0

    def test_drop_nulls_with_subset(self):
        """Only nulls in subset columns trigger drop"""
        f = ar.from_pandas(pd.DataFrame({"a": [1.0, None, 3.0], "b": ["x", "y", None]}))
        result = ar.drop_nulls(f, subset=["a"])
        pdf = ar.to_pandas(result)
        assert len(pdf) == 2
        assert pd.isna(pdf["b"].iloc[1])

    def test_drop_nulls_error_empty_subset(self):
        """Empty subset raises ValueError"""
        f = ar.from_pandas(pd.DataFrame({"a": [1.0]}))
        with pytest.raises(ValueError, match="subset cannot be empty"):
            ar.drop_nulls(f, subset=[])


class TestFilterRows:
    """Test arnio.cleaning.filter_rows"""

    def test_filter_rows_gt_operator(self):
        """Greater than operator filters correctly"""
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        result = ar.filter_rows(df, "a", ">", 1)
        assert result["a"].tolist() == [2, 3]

    def test_filter_rows_eq_operator(self):
        """Equality operator"""
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = ar.filter_rows(df, "a", "==", 2)
        assert len(result) == 1
        assert result["a"].iloc[0] == 2

    def test_filter_rows_error_invalid_operator(self):
        """Invalid operator raises ValueError"""
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(ValueError, match="Unsupported operator"):
            ar.filter_rows(df, "a", "=>", 1)

    def test_filter_rows_error_missing_column(self):
        """Missing column raises ValueError"""
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="Unknown column"):
            ar.filter_rows(df, "z", ">", 1)


class TestKeepRowsWithNulls:
    """Test arnio.cleaning.keep_rows_with_nulls"""

    def test_keep_rows_with_nulls_basic(self):
        """Only rows with at least one null kept"""
        f = ar.from_pandas(pd.DataFrame({"a": [1.0, None, 3.0], "b": ["x", "y", None]}))
        result = ar.keep_rows_with_nulls(f)
        pdf = ar.to_pandas(result)
        assert len(pdf) == 2
        assert pd.isna(pdf["a"].iloc[0]) or pd.isna(pdf["b"].iloc[0])

    def test_keep_rows_with_nulls_subset(self):
        """subset parameter limits null check"""
        df = pd.DataFrame({"a": [1.0, None, 3.0], "b": ["x", "y", None]})
        result = ar.keep_rows_with_nulls(ar.from_pandas(df), subset=["a"])
        pdf = ar.to_pandas(result)
        assert len(pdf) == 1
        assert pd.isna(pdf["a"].iloc[0])

    def test_keep_rows_with_nulls_error_string_subset(self):
        """String subset raises helpful TypeError"""
        f = ar.from_pandas(pd.DataFrame({"a": [1.0]}))
        with pytest.raises(TypeError, match="Did you mean subset="):
            ar.keep_rows_with_nulls(f, subset="a")


class TestReplaceValues:
    """Test arnio.cleaning.replace_values"""

    def test_replace_values_column_specific(self):
        """Values replaced only in specified column"""
        df = pd.DataFrame({"a": ["x", "y"], "b": ["x", "q"]})
        result = ar.replace_values(df, {"x": "X"}, column="a")
        assert result["a"].tolist() == ["X", "y"]
        assert result["b"].tolist() == ["x", "q"]

    def test_replace_values_null_key(self):
        """Null key replaces pre-existing nulls"""
        df = pd.DataFrame({"a": ["x", None, "z"]})
        result = ar.replace_values(df, {None: "MISSING"})
        assert result["a"].tolist() == ["x", "MISSING", "z"]

    def test_replace_values_error_tuple_key(self):
        """Tuple keys raise TypeError"""
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(TypeError, match="does not support non-scalar mapping keys"):
            ar.replace_values(df, {(1, 2): "x"})


class TestSafeDivideColumns:
    """Test arnio.cleaning.safe_divide_columns"""

    def test_safe_divide_columns_basic(self):
        """Safe division with fill_value for zero/null"""
        df = pd.DataFrame({"n": [10.0, 5.0, 3.0], "d": [2.0, 0.0, None]})
        result = ar.safe_divide_columns(df, "n", "d", "r", fill_value=-1.0)
        assert result["r"].iloc[0] == 5.0
        assert result["r"].iloc[1] == -1.0
        assert result["r"].iloc[2] == -1.0

    def test_safe_divide_columns_error_missing_column(self):
        """Missing numerator/denominator raises ValueError"""
        df = pd.DataFrame({"a": [1.0]})
        with pytest.raises(ValueError, match="not found in frame"):
            ar.safe_divide_columns(df, "z", "a", "r")

    def test_safe_divide_columns_error_output_exists(self):
        """Existing output column raises ValueError"""
        df = pd.DataFrame({"n": [1.0], "d": [1.0], "r": [1.0]})
        with pytest.raises(ValueError, match="already exists"):
            ar.safe_divide_columns(df, "n", "d", "r")

    def test_safe_divide_columns_error_non_finite_fill(self):
        """Non-finite fill_value raises ValueError"""
        df = pd.DataFrame({"n": [1.0], "d": [1.0]})
        with pytest.raises(ValueError, match="must be finite"):
            ar.safe_divide_columns(df, "n", "d", "r", fill_value=float("inf"))


class TestStandardizeMissingTokens:
    """Test arnio.cleaning.standardize_missing_tokens"""

    def test_standardize_missing_tokens_default_tokens(self):
        """Default tokens replaced with NaN"""
        df = pd.DataFrame({"v": [" NULL ", "\tNaN\t", "kept", "N/A"]})
        result = ar.standardize_missing_tokens(df)
        assert pd.isna(result["v"].iloc[0])
        assert pd.isna(result["v"].iloc[1])
        assert result["v"].iloc[2] == "kept"
        assert pd.isna(result["v"].iloc[3])

    def test_standardize_missing_tokens_case_insensitive(self):
        """Matching is case-insensitive"""
        df = pd.DataFrame({"v": ["NULL", "null", "NuLl", "nope"]})
        result = ar.standardize_missing_tokens(df)
        assert pd.isna(result["v"].iloc[0])
        assert pd.isna(result["v"].iloc[1])
        assert pd.isna(result["v"].iloc[2])
        assert result["v"].iloc[3] == "nope"

    def test_standardize_missing_tokens_subset(self):
        """subset limits which columns are processed"""
        df = pd.DataFrame({"a": ["NA"], "b": ["NA"]})
        result = ar.standardize_missing_tokens(df, subset=["a"])
        assert pd.isna(result["a"].iloc[0])
        assert result["b"].iloc[0] == "NA"

    def test_standardize_missing_tokens_custom_tokens(self):
        """Custom tokens replace defaults"""
        df = pd.DataFrame({"v": ["MISSING", "null", "other"]})
        result = ar.standardize_missing_tokens(df, tokens=["MISSING"])
        assert pd.isna(result["v"].iloc[0])
        assert result["v"].iloc[1] == "null"
        assert result["v"].iloc[2] == "other"

    def test_standardize_missing_tokens_whitespace_preserved(self):
        """Non-matching strings preserve whitespace"""
        df = pd.DataFrame({"v": [" kept ", "   ", "data"]})
        result = ar.standardize_missing_tokens(df)
        assert result["v"].iloc[0] == " kept "
        assert pd.isna(result["v"].iloc[1])  # whitespace-only is a token
        assert result["v"].iloc[2] == "data"
