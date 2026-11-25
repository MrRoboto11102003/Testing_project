import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestCoalesceColumns:
    """Tests for ar.coalesce_columns."""

    def test_coalesce_basic_first_non_null(self):
        """Test basic coalesce selects first non-null value."""
        df = pd.DataFrame({
            "phone": ["123-456", None, None, "789"],
            "mobile": [None, "555-1234", "555-5678", None],
            "fax": [None, None, "999-9999", None],
        })
        frame = ar.from_pandas(df)
        result = ar.coalesce_columns(frame, subset=["phone", "mobile", "fax"])
        result_df = ar.to_pandas(result)

        assert result_df["coalesced"].tolist() == ["123-456", "555-1234", "555-5678", "789"]

    def test_coalesce_custom_output_column_name(self):
        """Test coalesce with custom output column name."""
        df = pd.DataFrame({"a": [None, "B"], "b": ["A", None]})
        frame = ar.from_pandas(df)
        result = ar.coalesce_columns(frame, subset=["a", "b"], output_column="contact")
        result_df = ar.to_pandas(result)

        assert "contact" in result_df.columns
        assert result_df["contact"].tolist() == ["A", "B"]

    def test_coalesce_all_nulls_returns_null(self):
        """Test coalesce returns null when all values in row are null."""
        df = pd.DataFrame({"a": [None, "value"], "b": [None, None]})
        frame = ar.from_pandas(df)
        result = ar.coalesce_columns(frame, subset=["a", "b"])
        result_df = ar.to_pandas(result)

        assert pd.isna(result_df["coalesced"].iloc[0])
        assert result_df["coalesced"].iloc[1] == "value"

    def test_coalesce_stops_at_first_non_null(self):
        """Test coalesce uses first non-null, ignoring later values."""
        df = pd.DataFrame({
            "col1": ["first", None, None],
            "col2": ["second", "second_val", None],
            "col3": ["third", "third_val", "third_val2"],
        })
        frame = ar.from_pandas(df)
        result = ar.coalesce_columns(frame, subset=["col1", "col2", "col3"])
        result_df = ar.to_pandas(result)

        assert result_df["coalesced"].iloc[0] == "first"
        assert result_df["coalesced"].iloc[1] == "second_val"
        assert result_df["coalesced"].iloc[2] == "third_val2"

    def test_coalesce_single_column(self):
        """Test coalesce with single column subset."""
        df = pd.DataFrame({"col": ["a", None, "c"]})
        frame = ar.from_pandas(df)
        result = ar.coalesce_columns(frame, subset=["col"])
        result_df = ar.to_pandas(result)

        assert result_df["coalesced"].tolist() == ["a", None, "c"]

    def test_coalesce_preserves_original_columns(self):
        """Test coalesce adds new column without removing originals."""
        df = pd.DataFrame({"phone": ["123", None], "mobile": [None, "456"]})
        frame = ar.from_pandas(df)
        result = ar.coalesce_columns(frame, subset=["phone", "mobile"])
        result_df = ar.to_pandas(result)

        assert "phone" in result_df.columns
        assert "mobile" in result_df.columns
        assert "coalesced" in result_df.columns

    def test_coalesce_nonexistent_column_raises_error(self):
        """Test coalesce raises error when column doesn't exist."""
        df = pd.DataFrame({"a": [1, 2]})
        frame = ar.from_pandas(df)

        with pytest.raises((ValueError, KeyError)):
            ar.coalesce_columns(frame, subset=["a", "nonexistent"])


class TestStandardizeMissingTokens:
    """Tests for ar.standardize_missing_tokens."""

    def test_standardize_common_missing_tokens(self):
        """Test standardizing common missing value tokens to NaN."""
        df = pd.DataFrame({
            "col": ["value", "", "NA", "NULL"],
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)

        assert result_df["col"].iloc[0] == "value"
        # Common tokens should be standardized to NaN
        na_count = result_df["col"].isna().sum()
        assert na_count >= 1

    def test_standardize_with_custom_tokens(self):
        """Test standardizing with custom missing tokens."""
        df = pd.DataFrame({
            "col": ["valid", "MISSING", "UNKNOWN", "data"],
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame, tokens=["MISSING", "UNKNOWN"])
        result_df = ar.to_pandas(result)

        assert result_df["col"].iloc[0] == "valid"
        assert pd.isna(result_df["col"].iloc[1])
        assert pd.isna(result_df["col"].iloc[2])
        assert result_df["col"].iloc[3] == "data"

    def test_standardize_subset_specific_columns(self):
        """Test standardizing only specified columns."""
        df = pd.DataFrame({
            "col1": ["NA", "value"],
            "col2": ["NA", "data"],
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame, subset=["col1"])
        result_df = ar.to_pandas(result)

        assert pd.isna(result_df["col1"].iloc[0])
        assert result_df["col2"].iloc[0] == "NA"  # Unchanged

    def test_standardize_no_changes_when_no_tokens_present(self):
        """Test standardizing when no missing tokens are present."""
        df = pd.DataFrame({"col": ["valid", "data", "values"]})
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)

        assert result_df["col"].tolist() == ["valid", "data", "values"]

    def test_standardize_multiple_token_types(self):
        """Test standardizing multiple types of missing tokens."""
        df = pd.DataFrame({
            "col": ["value", "", "NULL", "MISSING", "N/A", "data"],
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(
            frame, tokens=["", "NULL", "MISSING", "N/A"]
        )
        result_df = ar.to_pandas(result)

        assert result_df["col"].iloc[0] == "value"
        for i in [1, 2, 3, 4]:
            assert pd.isna(result_df["col"].iloc[i])
        assert result_df["col"].iloc[5] == "data"

    def test_standardize_preserves_frame_structure(self):
        """Test standardizing preserves all columns and non-string columns."""
        df = pd.DataFrame({
            "a": ["value", "NA"],
            "b": [1, 2],
            "c": ["x", "y"],
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)

        assert set(result_df.columns) == {"a", "b", "c"}
        assert result_df["b"].tolist() == [1, 2]
        assert result_df["c"].tolist() == ["x", "y"]

    def test_standardize_empty_dataframe(self):
        """Test standardizing an empty frame."""
        df = pd.DataFrame({"col": []})
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)

        assert len(result_df) == 0
        assert "col" in result_df.columns

    def test_standardize_mixed_data_types(self):
        """Test standardizing across mixed data types."""
        df = pd.DataFrame({
            "str_col": ["valid", "NA", ""],
            "int_col": [1, 2, 3],
            "float_col": [1.1, 2.2, 3.3],
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame, tokens=["NA", ""])
        result_df = ar.to_pandas(result)

        # String column should be updated
        assert result_df["str_col"].iloc[0] == "valid"
        assert pd.isna(result_df["str_col"].iloc[1]) or pd.isna(result_df["str_col"].iloc[2])
        # Numeric columns should be unchanged
        assert result_df["int_col"].tolist() == [1, 2, 3]
        assert result_df["float_col"].tolist() == [1.1, 2.2, 3.3]
