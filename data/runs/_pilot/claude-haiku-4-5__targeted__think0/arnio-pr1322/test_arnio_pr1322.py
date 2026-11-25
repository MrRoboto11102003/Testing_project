import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestCoalesceColumns:
    """Tests for arnio.cleaning.coalesce_columns"""

    def test_coalesce_basic_first_non_null(self):
        """Test selecting first non-null value from multiple columns"""
        frame = ar.from_dict({
            "col_a": [1, None, 3],
            "col_b": [None, 2, None],
            "col_c": [None, None, 4]
        })
        result = ar.cleaning.coalesce_columns(
            frame, 
            subset=["col_a", "col_b", "col_c"],
            output_column="coalesced"
        )
        df = ar.to_pandas(result)
        assert list(df["coalesced"]) == [1.0, 2.0, 3.0]

    def test_coalesce_respects_order(self):
        """Test that coalesce respects column order"""
        frame = ar.from_dict({
            "col_a": [1, 2],
            "col_b": [3, 4]
        })
        result = ar.cleaning.coalesce_columns(
            frame,
            subset=["col_a", "col_b"],
            output_column="first"
        )
        df = ar.to_pandas(result)
        assert list(df["first"]) == [1, 2]

    def test_coalesce_all_null_values(self):
        """Test coalesce when all values in a row are null"""
        frame = ar.from_dict({
            "col_a": [None, 1],
            "col_b": [None, 2]
        })
        result = ar.cleaning.coalesce_columns(
            frame,
            subset=["col_a", "col_b"],
            output_column="coalesced"
        )
        df = ar.to_pandas(result)
        assert pd.isna(df["coalesced"].iloc[0])
        assert df["coalesced"].iloc[1] == 1

    def test_coalesce_single_column(self):
        """Test coalesce with a single column"""
        frame = ar.from_dict({
            "col_a": [1, None, 3]
        })
        result = ar.cleaning.coalesce_columns(
            frame,
            subset=["col_a"],
            output_column="single"
        )
        df = ar.to_pandas(result)
        assert list(df["single"]) == [1.0, np.nan, 3.0]

    def test_coalesce_default_output_name(self):
        """Test default output_column parameter"""
        frame = ar.from_dict({
            "col_a": [None, 2],
            "col_b": [1, None]
        })
        result = ar.cleaning.coalesce_columns(
            frame,
            subset=["col_a", "col_b"]
        )
        df = ar.to_pandas(result)
        assert "coalesced" in df.columns
        assert list(df["coalesced"]) == [1, 2]

    def test_coalesce_missing_column_raises_keyerror(self):
        """Test that missing column in subset raises KeyError"""
        frame = ar.from_dict({
            "col_a": [1, 2]
        })
        with pytest.raises(KeyError):
            ar.cleaning.coalesce_columns(
                frame,
                subset=["col_a", "missing_col"],
                output_column="coalesced"
            )

    def test_coalesce_empty_subset_raises_valueerror(self):
        """Test that empty subset raises ValueError"""
        frame = ar.from_dict({
            "col_a": [1, 2]
        })
        with pytest.raises(ValueError):
            ar.cleaning.coalesce_columns(
                frame,
                subset=[],
                output_column="coalesced"
            )

    def test_coalesce_output_column_already_exists_raises_valueerror(self):
        """Test that existing output_column raises ValueError"""
        frame = ar.from_dict({
            "col_a": [1, 2],
            "existing": [3, 4]
        })
        with pytest.raises(ValueError):
            ar.cleaning.coalesce_columns(
                frame,
                subset=["col_a"],
                output_column="existing"
            )

    def test_coalesce_empty_output_column_name_raises_valueerror(self):
        """Test that empty output_column name raises ValueError"""
        frame = ar.from_dict({
            "col_a": [1, 2]
        })
        with pytest.raises(ValueError):
            ar.cleaning.coalesce_columns(
                frame,
                subset=["col_a"],
                output_column=""
            )

    def test_coalesce_with_pandas_dataframe(self):
        """Test that coalesce works with pandas DataFrame"""
        df = pd.DataFrame({
            "col_a": [None, 2],
            "col_b": [1, None]
        })
        result = ar.cleaning.coalesce_columns(
            df,
            subset=["col_a", "col_b"],
            output_column="coalesced"
        )
        assert isinstance(result, pd.DataFrame)
        assert list(result["coalesced"]) == [1, 2]

    def test_coalesce_non_sequence_subset_raises_typeerror(self):
        """Test that non-sequence subset raises TypeError"""
        frame = ar.from_dict({
            "col_a": [1, 2]
        })
        with pytest.raises(TypeError):
            ar.cleaning.coalesce_columns(
                frame,
                subset="col_a",
                output_column="coalesced"
            )


class TestStandardizeMissingTokens:
    """Tests for arnio.cleaning.standardize_missing_tokens"""

    def test_standardize_default_tokens(self):
        """Test standardization with default missing tokens"""
        df = pd.DataFrame({
            "value": ["N/A", "NA", "null", "data"]
        })
        frame = ar.from_dict({"value": df["value"].tolist()})
        result = ar.cleaning.standardize_missing_tokens(frame)
        df_result = ar.to_pandas(result)
        assert pd.isna(df_result["value"].iloc[0])
        assert pd.isna(df_result["value"].iloc[1])
        assert pd.isna(df_result["value"].iloc[2])
        assert df_result["value"].iloc[3] == "data"

    def test_standardize_custom_tokens(self):
        """Test standardization with custom tokens list"""
        frame = ar.from_dict({
            "value": ["MISSING", "custom", "UNKNOWN"]
        })
        result = ar.cleaning.standardize_missing_tokens(
            frame,
            tokens=["MISSING", "UNKNOWN"]
        )
        df = ar.to_pandas(result)
        assert pd.isna(df["value"].iloc[0])
        assert df["value"].iloc[1] == "custom"
        assert pd.isna(df["value"].iloc[2])

    def test_standardize_case_insensitive(self):
        """Test that matching is case-insensitive"""
        frame = ar.from_dict({
            "value": ["n/a", "N/A", "Na", "N/a"]
        })
        result = ar.cleaning.standardize_missing_tokens(
            frame,
            tokens=["N/A"]
        )
        df = ar.to_pandas(result)
        assert all(pd.isna(df["value"]))

    def test_standardize_whitespace_trimmed(self):
        """Test that whitespace is trimmed before token matching"""
        frame = ar.from_dict({
            "value": [" N/A ", "\tNaN\t", "  null  ", "data"]
        })
        result = ar.cleaning.standardize_missing_tokens(frame)
        df = ar.to_pandas(result)
        assert pd.isna(df["value"].iloc[0])
        assert pd.isna(df["value"].iloc[1])
        assert pd.isna(df["value"].iloc[2])
        assert df["value"].iloc[3] == "data"

    def test_standardize_preserves_non_matching_whitespace(self):
        """Test that non-matching values preserve original whitespace"""
        frame = ar.from_dict({
            "value": [" preserved ", "\tkeep\t"]
        })
        result = ar.cleaning.standardize_missing_tokens(
            frame,
            tokens=["N/A"]
        )
        df = ar.to_pandas(result)
        assert df["value"].iloc[0] == " preserved "
        assert df["value"].iloc[1] == "\tkeep\t"

    def test_standardize_subset_columns(self):
        """Test standardization on subset of columns"""
        frame = ar.from_dict({
            "col_a": ["N/A", "value"],
            "col_b": ["N/A", "keep"]
        })
        result = ar.cleaning.standardize_missing_tokens(
            frame,
            subset=["col_a"]
        )
        df = ar.to_pandas(result)
        assert pd.isna(df["col_a"].iloc[0])
        assert df["col_b"].iloc[0] == "N/A"

    def test_standardize_all_columns_when_subset_none(self):
        """Test that None subset applies to all columns"""
        frame = ar.from_dict({
            "col_a": ["null"],
            "col_b": ["null"]
        })
        result = ar.cleaning.standardize_missing_tokens(frame, subset=None)
        df = ar.to_pandas(result)
        assert pd.isna(df["col_a"].iloc[0])
        assert pd.isna(df["col_b"].iloc[0])

    def test_standardize_none_tokens_uses_default(self):
        """Test that None tokens parameter uses default list"""
        frame = ar.from_dict({
            "value": ["", "NULL", "N/A"]
        })
        result = ar.cleaning.standardize_missing_tokens(frame, tokens=None)
        df = ar.to_pandas(result)
        # NULL and N/A should be in defaults
        assert pd.isna(df["value"].iloc[1]) or pd.isna(df["value"].iloc[2])

    def test_standardize_empty_string_token(self):
        """Test handling of empty string as a token"""
        frame = ar.from_dict({
            "value": ["", "data"]
        })
        result = ar.cleaning.standardize_missing_tokens(
            frame,
            tokens=[""]
        )
        df = ar.to_pandas(result)
        assert pd.isna(df["value"].iloc[0])
        assert df["value"].iloc[1] == "data"
