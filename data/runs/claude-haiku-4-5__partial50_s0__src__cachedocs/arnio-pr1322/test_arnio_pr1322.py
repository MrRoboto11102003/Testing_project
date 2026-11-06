import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestCoalesceColumns:
    """Tests for arnio.cleaning.coalesce_columns"""

    def test_basic_coalesce_with_mixed_nulls(self):
        """Test basic coalescing selects first non-null value."""
        df = pd.DataFrame({
            "col_a": [1, None, 3, None],
            "col_b": [None, 2, None, 4],
            "col_c": [None, None, None, None],
        })
        frame = ar.from_pandas(df)
        result = ar.coalesce_columns(frame, subset=["col_a", "col_b", "col_c"])
        result_df = ar.to_pandas(result)

        assert "coalesced" in result_df.columns
        assert result_df["coalesced"].tolist() == [1.0, 2.0, 3.0, 4.0]

    def test_coalesce_respects_column_order(self):
        """Test that coalesce picks first non-null in specified order."""
        df = pd.DataFrame({
            "priority_1": [None, 10, None],
            "priority_2": [100, None, 20],
            "priority_3": [1000, 1000, 30],
        })
        frame = ar.from_pandas(df)
        result = ar.coalesce_columns(
            frame,
            subset=["priority_1", "priority_2", "priority_3"],
            output_column="first_pick",
        )
        result_df = ar.to_pandas(result)

        assert result_df["first_pick"].tolist() == [100.0, 10.0, 20.0]

    def test_coalesce_all_nulls_in_row(self):
        """Test coalesce when all subset columns are null in a row."""
        df = pd.DataFrame({
            "a": [1, None],
            "b": [2, None],
        })
        frame = ar.from_pandas(df)
        result = ar.coalesce_columns(frame, subset=["a", "b"])
        result_df = ar.to_pandas(result)

        assert result_df["coalesced"].iloc[0] == 1.0
        assert pd.isna(result_df["coalesced"].iloc[1])

    def test_coalesce_output_column_already_exists_raises_error(self):
        """Test ValueError when output_column already exists in frame."""
        df = pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]})
        frame = ar.from_pandas(df)

        with pytest.raises(ValueError, match="already exists"):
            ar.coalesce_columns(frame, subset=["col_b"], output_column="col_a")

    def test_coalesce_missing_column_in_subset_raises_error(self):
        """Test error when subset references non-existent column."""
        df = pd.DataFrame({"a": [1], "b": [2]})
        frame = ar.from_pandas(df)

        with pytest.raises((KeyError, ValueError)):
            ar.coalesce_columns(frame, subset=["a", "nonexistent"])

    def test_coalesce_custom_output_column_name(self):
        """Test that custom output_column parameter is respected."""
        df = pd.DataFrame({"a": [1, None], "b": [None, 2]})
        frame = ar.from_pandas(df)
        result = ar.coalesce_columns(
            frame,
            subset=["a", "b"],
            output_column="my_coalesced",
        )
        result_df = ar.to_pandas(result)

        assert "my_coalesced" in result_df.columns
        assert result_df["my_coalesced"].tolist() == [1.0, 2.0]

    def test_coalesce_with_pandas_dataframe(self):
        """Test that coalesce_columns works with pandas DataFrame input."""
        df = pd.DataFrame({"a": [None, 2], "b": [1, None]})
        result = ar.coalesce_columns(df, subset=["a", "b"])

        assert isinstance(result, pd.DataFrame)
        assert "coalesced" in result.columns
        assert result["coalesced"].tolist() == [1.0, 2.0]


class TestStandardizeMissingTokens:
    """Tests for arnio.cleaning.standardize_missing_tokens"""

    def test_default_tokens_standardization(self):
        """Test standardization with built-in default tokens."""
        df = pd.DataFrame({
            "value": ["N/A", "NA", "10", "n/a", "nan", "null", "none"],
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)

        for i in [0, 1, 3, 4, 5, 6]:
            assert pd.isna(result_df["value"].iloc[i])
        assert result_df["value"].iloc[2] == "10"

    def test_custom_tokens_list(self):
        """Test standardization with custom token list."""
        df = pd.DataFrame({
            "value": ["CUSTOM_NULL", "OK", "MISSING", "data"],
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(
            frame,
            tokens=["CUSTOM_NULL", "MISSING"],
        )
        result_df = ar.to_pandas(result)

        assert pd.isna(result_df["value"].iloc[0])
        assert result_df["value"].iloc[1] == "OK"
        assert pd.isna(result_df["value"].iloc[2])
        assert result_df["value"].iloc[3] == "data"

    def test_case_insensitive_token_matching(self):
        """Test that token matching is case-insensitive."""
        df = pd.DataFrame({
            "value": ["NaN", "NAN", "nAn", "nan", "valid"],
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)

        for i in range(4):
            assert pd.isna(result_df["value"].iloc[i])
        assert result_df["value"].iloc[4] == "valid"

    def test_whitespace_trimming_before_matching(self):
        """Test that whitespace is trimmed before token matching."""
        df = pd.DataFrame({
            "value": ["  N/A  ", "\tnull\t", " - ", "  value  "],
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)

        assert pd.isna(result_df["value"].iloc[0])
        assert pd.isna(result_df["value"].iloc[1])
        assert pd.isna(result_df["value"].iloc[2])
        assert result_df["value"].iloc[3] == "  value  "  # outer spaces preserved

    def test_subset_parameter_limits_columns(self):
        """Test that subset parameter applies tokens only to specified columns."""
        df = pd.DataFrame({
            "col_a": ["N/A", "value"],
            "col_b": ["N/A", "value"],
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame, subset=["col_a"])
        result_df = ar.to_pandas(result)

        assert pd.isna(result_df["col_a"].iloc[0])
        assert result_df["col_b"].iloc[0] == "N/A"  # col_b unchanged

    def test_tokens_not_a_list_raises_typeerror(self):
        """Test TypeError when tokens argument is not a list."""
        df = pd.DataFrame({"value": ["N/A"]})
        frame = ar.from_pandas(df)

        with pytest.raises(TypeError, match="must be a list"):
            ar.standardize_missing_tokens(frame, tokens="N/A")

    def test_subset_as_string_raises_typeerror(self):
        """Test TypeError when subset is a string instead of list."""
        df = pd.DataFrame({"col_a": ["N/A"], "col_b": ["10"]})
        frame = ar.from_pandas(df)

        with pytest.raises(TypeError, match="must be a list"):
            ar.standardize_missing_tokens(frame, subset="col_a")

    def test_empty_tokens_list_disables_standardization(self):
        """Test that empty tokens list disables all token replacement."""
        df = pd.DataFrame({
            "value": ["N/A", "null", "nan", "10"],
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame, tokens=[])
        result_df = ar.to_pandas(result)

        assert result_df["value"].iloc[0] == "N/A"
        assert result_df["value"].iloc[1] == "null"
        assert result_df["value"].iloc[2] == "nan"
        assert result_df["value"].iloc[3] == "10"

    def test_standardize_with_pandas_dataframe(self):
        """Test that standardize_missing_tokens works with pandas DataFrame."""
        df = pd.DataFrame({"value": ["N/A", "null", "10"]})
        result = ar.standardize_missing_tokens(df)

        assert isinstance(result, pd.DataFrame)
        assert pd.isna(result["value"].iloc[0])
        assert pd.isna(result["value"].iloc[1])
        assert result["value"].iloc[2] == "10"

    def test_non_string_values_not_affected(self):
        """Test that non-string values are preserved unchanged."""
        df = pd.DataFrame({
            "mixed": [1, 2.5, "N/A", None, True],
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)

        assert result_df["mixed"].iloc[0] == 1
        assert result_df["mixed"].iloc[1] == 2.5
        assert pd.isna(result_df["mixed"].iloc[2])
        assert pd.isna(result_df["mixed"].iloc[3]) or result_df["mixed"].iloc[3] is None
        assert result_df["mixed"].iloc[4] == True

    def test_subset_none_applies_to_all_columns(self):
        """Test that subset=None applies token standardization to all columns."""
        df = pd.DataFrame({
            "col_a": ["N/A"],
            "col_b": ["null"],
            "col_c": ["nan"],
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame, subset=None)
        result_df = ar.to_pandas(result)

        assert pd.isna(result_df["col_a"].iloc[0])
        assert pd.isna(result_df["col_b"].iloc[0])
        assert pd.isna(result_df["col_c"].iloc[0])
