import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestCoalesceColumns:
    """Tests for arnio.cleaning.coalesce_columns"""

    def test_coalesce_selects_first_non_null(self):
        """Select first non-null value from columns in subset order"""
        frame = ar.from_records([
            {"phone": "555-1234", "mobile": None},
            {"phone": None, "mobile": "555-5678"},
            {"phone": "555-9999", "mobile": "555-8888"},
        ])
        result = ar.coalesce_columns(frame, subset=["phone", "mobile"])
        df = ar.to_pandas(result)
        assert "coalesced" in df.columns
        values = df["coalesced"].tolist()
        assert values[0] == "555-1234"
        assert values[1] == "555-5678"
        assert values[2] == "555-9999"

    def test_coalesce_custom_output_column(self):
        """Custom output column name"""
        frame = ar.from_records([
            {"a": None, "b": "value_b"},
        ])
        result = ar.coalesce_columns(frame, subset=["a", "b"], output_column="contact")
        df = ar.to_pandas(result)
        assert "contact" in df.columns
        assert df["contact"].iloc[0] == "value_b"

    def test_coalesce_default_output_column_name(self):
        """Default output column name is 'coalesced'"""
        frame = ar.from_records([{"a": "val"}])
        result = ar.coalesce_columns(frame, subset=["a"])
        df = ar.to_pandas(result)
        assert "coalesced" in df.columns

    def test_coalesce_all_nulls_result_is_null(self):
        """All null values in subset produce null output"""
        frame = ar.from_records([
            {"a": None, "b": None},
        ])
        result = ar.coalesce_columns(frame, subset=["a", "b"])
        df = ar.to_pandas(result)
        assert pd.isna(df["coalesced"].iloc[0])

    def test_coalesce_respects_subset_column_order(self):
        """First non-null respects subset column order"""
        frame = ar.from_records([
            {"z": "z_val", "a": "a_val"},
        ])
        result = ar.coalesce_columns(frame, subset=["z", "a"])
        df = ar.to_pandas(result)
        assert df["coalesced"].iloc[0] == "z_val"

    def test_coalesce_subset_keyword_only(self):
        """subset parameter must be keyword-only"""
        frame = ar.from_records([{"a": "val"}])
        with pytest.raises(TypeError):
            ar.coalesce_columns(frame, ["a"])

    def test_coalesce_missing_column_raises(self):
        """Non-existent column in subset raises error"""
        frame = ar.from_records([{"exists": "val"}])
        with pytest.raises((ValueError, KeyError)):
            ar.coalesce_columns(frame, subset=["exists", "missing"])

    def test_coalesce_single_column_copies_values(self):
        """Coalescing single column effectively copies values"""
        frame = ar.from_records([
            {"col": "value1"},
            {"col": None},
        ])
        result = ar.coalesce_columns(frame, subset=["col"])
        df = ar.to_pandas(result)
        assert df["coalesced"].iloc[0] == "value1"
        assert pd.isna(df["coalesced"].iloc[1])


class TestStandardizeMissingTokens:
    """Tests for arnio.cleaning.standardize_missing_tokens"""

    def test_standardize_replaces_common_missing_tokens(self):
        """Replace common missing-value strings with NaN"""
        frame = ar.from_records([
            {"col": "N/A"},
            {"col": "valid"},
            {"col": "NA"},
        ])
        result = ar.standardize_missing_tokens(frame)
        df = ar.to_pandas(result)
        assert pd.isna(df.loc[0, "col"])
        assert df.loc[1, "col"] == "valid"
        assert pd.isna(df.loc[2, "col"])

    def test_standardize_with_custom_tokens(self):
        """Specify custom list of tokens to standardize"""
        frame = ar.from_records([
            {"col": "CUSTOM"},
            {"col": "N/A"},
            {"col": "data"},
        ])
        result = ar.standardize_missing_tokens(frame, tokens=["CUSTOM"])
        df = ar.to_pandas(result)
        assert pd.isna(df.loc[0, "col"])
        assert df.loc[1, "col"] == "N/A"
        assert df.loc[2, "col"] == "data"

    def test_standardize_subset_columns_only(self):
        """Apply standardization only to specified columns"""
        frame = ar.from_records([
            {"col_a": "N/A", "col_b": "N/A"},
        ])
        result = ar.standardize_missing_tokens(frame, subset=["col_a"])
        df = ar.to_pandas(result)
        assert pd.isna(df.loc[0, "col_a"])
        assert df.loc[0, "col_b"] == "N/A"

    def test_standardize_preserves_existing_nulls(self):
        """Actual null values remain null"""
        frame = ar.from_records([
            {"col": None},
        ])
        result = ar.standardize_missing_tokens(frame)
        df = ar.to_pandas(result)
        assert pd.isna(df.loc[0, "col"])

    def test_standardize_mixed_nulls_and_tokens(self):
        """Mix of actual nulls and missing tokens handled correctly"""
        frame = ar.from_records([
            {"col": None},
            {"col": "N/A"},
            {"col": "NA"},
        ])
        result = ar.standardize_missing_tokens(frame)
        df = ar.to_pandas(result)
        assert pd.isna(df.loc[0, "col"])
        assert pd.isna(df.loc[1, "col"])
        assert pd.isna(df.loc[2, "col"])

    def test_standardize_nonexistent_column_raises(self):
        """Non-existent column in subset raises error"""
        frame = ar.from_records([{"col": "val"}])
        with pytest.raises((ValueError, KeyError)):
            ar.standardize_missing_tokens(frame, subset=["missing"])

    def test_standardize_valid_values_unchanged(self):
        """Valid non-missing values are unchanged"""
        frame = ar.from_records([
            {"col": "valid"},
            {"col": "123"},
        ])
        result = ar.standardize_missing_tokens(frame)
        df = ar.to_pandas(result)
        assert df.loc[0, "col"] == "valid"
        assert df.loc[1, "col"] == "123"
