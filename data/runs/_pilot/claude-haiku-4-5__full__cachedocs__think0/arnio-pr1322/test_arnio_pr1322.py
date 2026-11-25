import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestCoalesceColumns:
    """Tests for ar.coalesce_columns"""

    def test_coalesce_returns_first_non_null(self):
        """Select the first non-null value from subset columns."""
        frame = ar.from_pandas(pd.DataFrame({
            "col_a": [1, None, 3],
            "col_b": [10, 20, 30],
            "col_c": [100, 200, 300],
        }))
        result = ar.coalesce_columns(frame, subset=["col_a", "col_b"], output_column="first")
        df = ar.to_pandas(result)
        assert df["first"].tolist() == [1.0, 20.0, 3.0]

    def test_coalesce_preserves_order(self):
        """Coalesce respects the order of subset columns."""
        frame = ar.from_pandas(pd.DataFrame({
            "a": [None, 2, None],
            "b": [10, None, 30],
            "c": [100, 200, 300],
        }))
        result = ar.coalesce_columns(frame, subset=["a", "b", "c"], output_column="result")
        df = ar.to_pandas(result)
        assert df["result"].tolist() == [10.0, 2.0, 30.0]

    def test_coalesce_all_nulls_returns_null(self):
        """When all values in subset are null, output is null."""
        frame = ar.from_pandas(pd.DataFrame({
            "x": [None, None],
            "y": [None, None],
        }))
        result = ar.coalesce_columns(frame, subset=["x", "y"], output_column="out")
        df = ar.to_pandas(result)
        assert pd.isna(df["out"]).all()

    def test_coalesce_single_column(self):
        """Coalesce with a single column works correctly."""
        frame = ar.from_pandas(pd.DataFrame({
            "col": [1, None, 3],
        }))
        result = ar.coalesce_columns(frame, subset=["col"], output_column="single")
        df = ar.to_pandas(result)
        assert df["single"].tolist() == [1.0, np.nan, 3.0]

    def test_coalesce_default_output_column_name(self):
        """Default output_column is 'coalesced'."""
        frame = ar.from_pandas(pd.DataFrame({
            "a": [None, 2],
            "b": [10, 20],
        }))
        result = ar.coalesce_columns(frame, subset=["a", "b"])
        df = ar.to_pandas(result)
        assert "coalesced" in df.columns
        assert df["coalesced"].tolist() == [10.0, 2.0]

    def test_coalesce_raises_on_empty_subset(self):
        """Raises ValueError when subset is empty."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))
        with pytest.raises(ValueError):
            ar.coalesce_columns(frame, subset=[], output_column="out")

    def test_coalesce_raises_on_empty_output_column_name(self):
        """Raises ValueError when output_column is empty string."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))
        with pytest.raises(ValueError):
            ar.coalesce_columns(frame, subset=["a"], output_column="")

    def test_coalesce_raises_when_output_column_exists(self):
        """Raises ValueError when output_column already exists."""
        frame = ar.from_pandas(pd.DataFrame({
            "a": [1, 2],
            "existing": [10, 20],
        }))
        with pytest.raises(ValueError):
            ar.coalesce_columns(frame, subset=["a"], output_column="existing")

    def test_coalesce_raises_on_missing_column(self):
        """Raises KeyError when a subset column does not exist."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))
        with pytest.raises(KeyError):
            ar.coalesce_columns(frame, subset=["a", "nonexistent"], output_column="out")

    def test_coalesce_raises_on_non_sequence_subset(self):
        """Raises TypeError when subset is not a sequence."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))
        with pytest.raises(TypeError):
            ar.coalesce_columns(frame, subset="a", output_column="out")

    def test_coalesce_with_strings(self):
        """Coalesce works with string columns."""
        frame = ar.from_pandas(pd.DataFrame({
            "email": [None, "user2@example.com", None],
            "phone": ["555-1234", None, "555-5678"],
        }))
        result = ar.coalesce_columns(frame, subset=["email", "phone"], output_column="contact")
        df = ar.to_pandas(result)
        assert df["contact"].tolist() == ["555-1234", "user2@example.com", "555-5678"]


class TestStandardizeMissingTokens:
    """Tests for ar.standardize_missing_tokens"""

    def test_standardize_default_tokens(self):
        """Standardize uses built-in default tokens when not specified."""
        frame = ar.from_pandas(pd.DataFrame({
            "value": ["N/A", "NA", "null", "NULL", "valid"],
        }))
        result = ar.standardize_missing_tokens(frame)
        df = ar.to_pandas(result)
        assert pd.isna(df["value"][0])
        assert pd.isna(df["value"][1])
        assert pd.isna(df["value"][2])
        assert pd.isna(df["value"][3])
        assert df["value"][4] == "valid"

    def test_standardize_case_insensitive(self):
        """Token matching is case-insensitive."""
        frame = ar.from_pandas(pd.DataFrame({
            "col": ["n/a", "N/A", "Na", "nA"],
        }))
        result = ar.standardize_missing_tokens(frame, tokens=["n/a"])
        df = ar.to_pandas(result)
        assert pd.isna(df["col"]).all()

    def test_standardize_trims_whitespace(self):
        """Surrounding whitespace is trimmed before token matching."""
        frame = ar.from_pandas(pd.DataFrame({
            "col": [" NULL ", "\tNaN\t", "  NA  ", "valid"],
        }))
        result = ar.standardize_missing_tokens(frame, tokens=["NULL", "NaN", "NA"])
        df = ar.to_pandas(result)
        assert pd.isna(df["col"][0])
        assert pd.isna(df["col"][1])
        assert pd.isna(df["col"][2])
        assert df["col"][3] == "valid"

    def test_standardize_preserves_non_matching_whitespace(self):
        """Values that don't match a token preserve their original whitespace."""
        frame = ar.from_pandas(pd.DataFrame({
            "col": ["  kept  ", " value ", "N/A"],
        }))
        result = ar.standardize_missing_tokens(frame, tokens=["N/A"])
        df = ar.to_pandas(result)
        assert df["col"][0] == "  kept  "
        assert df["col"][1] == " value "
        assert pd.isna(df["col"][2])

    def test_standardize_with_subset(self):
        """Apply standardization only to specified columns."""
        frame = ar.from_pandas(pd.DataFrame({
            "col_a": ["N/A", "value"],
            "col_b": ["N/A", "value"],
        }))
        result = ar.standardize_missing_tokens(frame, tokens=["N/A"], subset=["col_a"])
        df = ar.to_pandas(result)
        assert pd.isna(df["col_a"][0])
        assert df["col_b"][0] == "N/A"

    def test_standardize_subset_ignores_other_columns(self):
        """Columns not in subset are left untouched."""
        frame = ar.from_pandas(pd.DataFrame({
            "process": ["null", "data"],
            "ignore": ["null", "data"],
        }))
        result = ar.standardize_missing_tokens(frame, tokens=["null"], subset=["process"])
        df = ar.to_pandas(result)
        assert pd.isna(df["process"][0])
        assert df["ignore"][0] == "null"

    def test_standardize_custom_tokens(self):
        """Custom token list overrides defaults."""
        frame = ar.from_pandas(pd.DataFrame({
            "col": ["MISSING", "unknown", "keep"],
        }))
        result = ar.standardize_missing_tokens(frame, tokens=["MISSING", "unknown"])
        df = ar.to_pandas(result)
        assert pd.isna(df["col"][0])
        assert pd.isna(df["col"][1])
        assert df["col"][2] == "keep"

    def test_standardize_empty_tokens_list(self):
        """Empty token list means no tokens are considered missing."""
        frame = ar.from_pandas(pd.DataFrame({
            "col": ["N/A", "null", "data"],
        }))
        result = ar.standardize_missing_tokens(frame, tokens=[])
        df = ar.to_pandas(result)
        assert df["col"].tolist() == ["N/A", "null", "data"]

    def test_standardize_all_columns_when_subset_none(self):
        """When subset is None, applies to all columns."""
        frame = ar.from_pandas(pd.DataFrame({
            "a": ["N/A", "value"],
            "b": ["null", "data"],
            "c": ["NA", "keep"],
        }))
        result = ar.standardize_missing_tokens(frame, tokens=["N/A", "null", "NA"])
        df = ar.to_pandas(result)
        assert pd.isna(df["a"][0])
        assert pd.isna(df["b"][0])
        assert pd.isna(df["c"][0])
        assert df["a"][1] == "value"
        assert df["b"][1] == "data"
        assert df["c"][1] == "keep"
