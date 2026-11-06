import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestCoalesceColumns:
    """Tests for arnio.cleaning.coalesce_columns"""

    def test_selects_first_non_null_value(self):
        """Coalesce selects first non-null value from subset columns"""
        df = pd.DataFrame({
            "col_a": [None, 1, None],
            "col_b": [2, None, 3],
            "col_c": [4, 5, 6],
        })
        frame = ar.from_pandas(df)
        result = ar.coalesce_columns(
            frame,
            subset=["col_a", "col_b", "col_c"],
            output_column="coalesced"
        )
        result_df = ar.to_pandas(result)
        
        assert result_df["coalesced"].tolist() == [2.0, 1.0, 3.0]

    def test_respects_subset_column_order(self):
        """Column order in subset determines selection priority"""
        df = pd.DataFrame({
            "x": [10, None],
            "y": [1, 2],
        })
        frame = ar.from_pandas(df)
        
        result_xy = ar.coalesce_columns(frame, subset=["x", "y"], output_column="result")
        df_xy = ar.to_pandas(result_xy)
        assert df_xy["result"].tolist()[0] == 10.0
        
        result_yx = ar.coalesce_columns(frame, subset=["y", "x"], output_column="result")
        df_yx = ar.to_pandas(result_yx)
        assert df_yx["result"].tolist()[0] == 1.0

    def test_all_null_source_columns_result_null(self):
        """All-null source columns produce null result"""
        df = pd.DataFrame({"a": [None, None], "b": [None, None]})
        frame = ar.from_pandas(df)
        result = ar.coalesce_columns(frame, subset=["a", "b"], output_column="c")
        result_df = ar.to_pandas(result)
        
        assert result_df["c"].isna().all()

    def test_single_column_in_subset(self):
        """Single column in subset works as expected"""
        df = pd.DataFrame({"col": [1, None, 3]})
        frame = ar.from_pandas(df)
        result = ar.coalesce_columns(frame, subset=["col"], output_column="out")
        result_df = ar.to_pandas(result)
        
        assert result_df["out"].tolist()[0] == 1.0
        assert pd.isna(result_df["out"].iloc[1])
        assert result_df["out"].tolist()[2] == 3.0

    def test_raises_on_empty_subset(self):
        """Empty subset raises ValueError"""
        df = pd.DataFrame({"a": [1]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(ValueError):
            ar.coalesce_columns(frame, subset=[], output_column="out")

    def test_raises_on_existing_output_column(self):
        """Existing output_column name raises ValueError"""
        df = pd.DataFrame({"a": [1], "existing": [10]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(ValueError):
            ar.coalesce_columns(frame, subset=["a"], output_column="existing")

    def test_raises_on_missing_column_in_subset(self):
        """Missing column in subset raises KeyError"""
        df = pd.DataFrame({"a": [1]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(KeyError):
            ar.coalesce_columns(frame, subset=["a", "missing"], output_column="out")

    def test_works_with_strings(self):
        """Coalesce works with string columns"""
        df = pd.DataFrame({
            "first": [None, "Alice", None],
            "second": ["Bob", None, "Charlie"],
        })
        frame = ar.from_pandas(df)
        result = ar.coalesce_columns(
            frame,
            subset=["first", "second"],
            output_column="name"
        )
        result_df = ar.to_pandas(result)
        
        assert result_df["name"].tolist() == ["Bob", "Alice", "Charlie"]


class TestStandardizeMissingTokens:
    """Tests for arnio.cleaning.standardize_missing_tokens"""

    def test_default_tokens_replaced(self):
        """Default missing tokens are replaced with NaN"""
        df = pd.DataFrame({
            "col": ["N/A", "NULL", "unknown", "valid", "NaN"]
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)
        
        assert pd.isna(result_df["col"].iloc[0])
        assert pd.isna(result_df["col"].iloc[1])
        assert pd.isna(result_df["col"].iloc[2])
        assert result_df["col"].iloc[3] == "valid"
        assert pd.isna(result_df["col"].iloc[4])

    def test_case_insensitive_matching(self):
        """Token matching is case-insensitive"""
        df = pd.DataFrame({
            "col": ["n/a", "N/A", "null", "NULL", "valid"]
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)
        
        assert pd.isna(result_df["col"].iloc[0])
        assert pd.isna(result_df["col"].iloc[1])
        assert pd.isna(result_df["col"].iloc[2])
        assert pd.isna(result_df["col"].iloc[3])
        assert result_df["col"].iloc[4] == "valid"

    def test_whitespace_trimmed_before_check(self):
        """Surrounding whitespace is trimmed before token matching"""
        df = pd.DataFrame({
            "col": ["  N/A  ", "\tNULL\t", " unknown ", "  value  "]
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)
        
        assert pd.isna(result_df["col"].iloc[0])
        assert pd.isna(result_df["col"].iloc[1])
        assert pd.isna(result_df["col"].iloc[2])
        assert result_df["col"].iloc[3] == "  value  "

    def test_custom_tokens(self):
        """Custom token list replaces defaults"""
        df = pd.DataFrame({
            "col": ["MISSING", "IGNORE", "N/A", "valid"]
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame, tokens=["MISSING", "IGNORE"])
        result_df = ar.to_pandas(result)
        
        assert pd.isna(result_df["col"].iloc[0])
        assert pd.isna(result_df["col"].iloc[1])
        assert result_df["col"].iloc[2] == "N/A"
        assert result_df["col"].iloc[3] == "valid"

    def test_subset_parameter(self):
        """Subset limits replacement to specified columns"""
        df = pd.DataFrame({
            "col1": ["N/A", "value1"],
            "col2": ["N/A", "value2"],
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame, subset=["col1"])
        result_df = ar.to_pandas(result)
        
        assert pd.isna(result_df["col1"].iloc[0])
        assert result_df["col2"].iloc[0] == "N/A"

    def test_empty_tokens_no_replacement(self):
        """Empty tokens list means no replacements"""
        df = pd.DataFrame({
            "col": ["N/A", "NULL", "valid"]
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame, tokens=[])
        result_df = ar.to_pandas(result)
        
        assert result_df["col"].tolist() == ["N/A", "NULL", "valid"]

    def test_all_columns_when_subset_none(self):
        """When subset=None, applies to all columns"""
        df = pd.DataFrame({
            "col1": ["N/A", "value1"],
            "col2": ["NULL", "value2"],
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame, subset=None)
        result_df = ar.to_pandas(result)
        
        assert pd.isna(result_df["col1"].iloc[0])
        assert pd.isna(result_df["col2"].iloc[0])
        assert result_df["col1"].iloc[1] == "value1"
        assert result_df["col2"].iloc[1] == "value2"
