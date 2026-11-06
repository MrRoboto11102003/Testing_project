import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestCoalesceColumns:
    """Tests for arnio.cleaning.coalesce_columns"""

    def test_coalesce_first_nonnull_value(self):
        """Select first non-null value from subset columns."""
        df = pd.DataFrame({
            "phone": ["555-1234", None, None, "555-5678"],
            "mobile": ["777-1111", "777-2222", None, None],
            "email": ["a@x.com", "b@x.com", "c@x.com", "d@x.com"],
        })
        frame = ar.from_pandas(df)
        result = ar.coalesce_columns(frame, subset=["phone", "mobile"])
        result_df = ar.to_pandas(result)

        assert result_df["coalesced"].iloc[0] == "555-1234"
        assert result_df["coalesced"].iloc[1] == "777-2222"
        assert pd.isna(result_df["coalesced"].iloc[2])
        assert result_df["coalesced"].iloc[3] == "555-5678"

    def test_coalesce_all_null_produces_null(self):
        """When all subset columns are null for a row, result is null."""
        df = pd.DataFrame({
            "col1": [None, "A", None],
            "col2": [None, None, "C"],
        })
        frame = ar.from_pandas(df)
        result = ar.coalesce_columns(frame, subset=["col1", "col2"])
        result_df = ar.to_pandas(result)

        assert pd.isna(result_df["coalesced"].iloc[0])
        assert result_df["coalesced"].iloc[1] == "A"
        assert result_df["coalesced"].iloc[2] == "C"

    def test_coalesce_custom_output_column_name(self):
        """Output column can be customized."""
        df = pd.DataFrame({"a": ["X", None], "b": [None, "Y"]})
        frame = ar.from_pandas(df)
        result = ar.coalesce_columns(frame, subset=["a", "b"], output_column="contact")
        result_df = ar.to_pandas(result)

        assert "contact" in result_df.columns
        assert result_df["contact"].iloc[0] == "X"
        assert result_df["contact"].iloc[1] == "Y"

    def test_coalesce_missing_column_raises_error(self):
        """Raise error if column in subset does not exist."""
        df = pd.DataFrame({"col_a": [1, 2]})
        frame = ar.from_pandas(df)

        with pytest.raises((ValueError, KeyError)):
            ar.coalesce_columns(frame, subset=["col_a", "missing"])

    def test_coalesce_empty_dataframe(self):
        """Coalesce on empty frame."""
        df = pd.DataFrame({"col1": pd.Series([], dtype=object), "col2": pd.Series([], dtype=object)})
        frame = ar.from_pandas(df)
        result = ar.coalesce_columns(frame, subset=["col1", "col2"])
        result_df = ar.to_pandas(result)

        assert len(result_df) == 0
        assert "coalesced" in result_df.columns

    def test_coalesce_with_numeric_columns(self):
        """Coalesce works with numeric values."""
        df = pd.DataFrame({
            "price1": [None, 100.5, 200.0],
            "price2": [10.0, None, None],
        })
        frame = ar.from_pandas(df)
        result = ar.coalesce_columns(frame, subset=["price1", "price2"])
        result_df = ar.to_pandas(result)

        assert result_df["coalesced"].iloc[0] == 10.0
        assert result_df["coalesced"].iloc[1] == 100.5
        assert result_df["coalesced"].iloc[2] == 200.0

    def test_coalesce_preserves_original_columns(self):
        """Coalesce adds new column without removing originals."""
        df = pd.DataFrame({"id": [1, 2], "col1": ["A", None], "col2": [None, "B"]})
        frame = ar.from_pandas(df)
        result = ar.coalesce_columns(frame, subset=["col1", "col2"])
        result_df = ar.to_pandas(result)

        assert "id" in result_df.columns
        assert "col1" in result_df.columns
        assert "col2" in result_df.columns
        assert "coalesced" in result_df.columns


class TestStandardizeMissingTokens:
    """Tests for arnio.cleaning.standardize_missing_tokens"""

    def test_standardize_replaces_common_tokens(self):
        """Default tokens replace common missing-value strings with NaN."""
        df = pd.DataFrame({
            "col": ["valid", "NA", "NULL", "MISSING"],
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)

        assert result_df["col"].iloc[0] == "valid"
        assert pd.isna(result_df["col"].iloc[1])
        assert pd.isna(result_df["col"].iloc[2])
        assert pd.isna(result_df["col"].iloc[3])

    def test_standardize_case_insensitive(self):
        """Token matching is case-insensitive."""
        df = pd.DataFrame({"col": ["NA", "na", "Na", "valid"]})
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)

        assert pd.isna(result_df["col"].iloc[0])
        assert pd.isna(result_df["col"].iloc[1])
        assert pd.isna(result_df["col"].iloc[2])
        assert result_df["col"].iloc[3] == "valid"

    def test_standardize_custom_tokens(self):
        """Use custom token list instead of defaults."""
        df = pd.DataFrame({"col": ["CUSTOM", "custom", "other", "valid"]})
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame, tokens=["CUSTOM", "OTHER"])
        result_df = ar.to_pandas(result)

        assert pd.isna(result_df["col"].iloc[0])
        assert pd.isna(result_df["col"].iloc[1])
        assert pd.isna(result_df["col"].iloc[2])
        assert result_df["col"].iloc[3] == "valid"

    def test_standardize_subset_columns(self):
        """Apply standardization only to specified columns."""
        df = pd.DataFrame({
            "target": ["NA", "valid"],
            "ignore": ["NA", "NA"],
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame, subset=["target"])
        result_df = ar.to_pandas(result)

        assert pd.isna(result_df["target"].iloc[0])
        assert result_df["target"].iloc[1] == "valid"
        assert result_df["ignore"].iloc[0] == "NA"
        assert result_df["ignore"].iloc[1] == "NA"

    def test_standardize_no_matching_tokens_unchanged(self):
        """Values not matching any token remain unchanged."""
        df = pd.DataFrame({"col": ["apple", "banana", "cherry"]})
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)

        assert result_df["col"].tolist() == ["apple", "banana", "cherry"]

    def test_standardize_all_values_are_tokens(self):
        """When all values match tokens, all become NaN."""
        df = pd.DataFrame({"col": ["NA", "NULL", "MISSING"]})
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)

        assert result_df["col"].isna().all()

    def test_standardize_empty_dataframe(self):
        """Standardize on empty frame."""
        df = pd.DataFrame({"col": pd.Series([], dtype=object)})
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)

        assert len(result_df) == 0

    def test_standardize_subset_nonexistent_column_raises_error(self):
        """Raise error if subset contains nonexistent column."""
        df = pd.DataFrame({"col1": ["NA", "valid"]})
        frame = ar.from_pandas(df)

        with pytest.raises((ValueError, KeyError)):
            ar.standardize_missing_tokens(frame, subset=["col1", "missing"])
