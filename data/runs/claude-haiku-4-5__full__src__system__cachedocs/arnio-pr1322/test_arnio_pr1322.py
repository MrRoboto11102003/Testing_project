"""
End-to-end pytest tests for arnio.cleaning.coalesce_columns and 
arnio.cleaning.standardize_missing_tokens.

Tests exercise real workflows: CSV files on disk, cleaning pipelines,
pandas conversion, and error cases. Uses public API only.
"""

import pytest
import pandas as pd
import numpy as np
import arnio as ar
from pathlib import Path


class TestCoalesceColumns:
    """End-to-end tests for ar.coalesce_columns()"""

    def test_coalesce_basic_selects_first_non_null(self, tmp_path):
        """Basic test: coalesce picks first non-null value from columns in order"""
        csv_file = tmp_path / "contacts.csv"
        csv_file.write_text("name,phone_1,phone_2,phone_3\n"
                           "Alice,555-1111,,\n"
                           "Bob,,555-2222,555-3333\n"
                           "Charlie,,,555-4444\n")
        
        frame = ar.read_csv(str(csv_file))
        result = ar.coalesce_columns(
            frame,
            subset=["phone_1", "phone_2", "phone_3"],
            output_column="primary_phone"
        )
        df = ar.to_pandas(result)
        
        assert df["primary_phone"].tolist() == ["555-1111", "555-2222", "555-4444"]
        assert len(df.columns) == 5  # original 4 + new column

    def test_coalesce_accepts_pandas_dataframe(self):
        """Test that coalesce_columns works with pandas DataFrame input"""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "email_work": ["alice@corp.com", None, None],
            "email_personal": [None, "bob@home.com", None],
            "email_other": [None, None, "charlie@alt.com"]
        })
        
        result = ar.coalesce_columns(
            df,
            subset=["email_work", "email_personal", "email_other"],
            output_column="best_email"
        )
        
        assert isinstance(result, pd.DataFrame)
        assert result["best_email"].tolist() == [
            "alice@corp.com",
            "bob@home.com",
            "charlie@alt.com"
        ]

    def test_coalesce_in_pipeline(self, tmp_path):
        """Test coalesce_columns as a pipeline step"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,first,second,third\n"
                           "1,A,,\n"
                           "2,,B,C\n"
                           "3,,,D\n")
        
        frame = ar.read_csv(str(csv_file))
        result = ar.pipeline(frame, [
            ("coalesce_columns", {
                "subset": ["first", "second", "third"],
                "output_column": "best_value"
            })
        ])
        df = ar.to_pandas(result)
        
        assert "best_value" in df.columns
        assert df["best_value"].tolist() == ["A", "B", "D"]

    def test_coalesce_round_trip_csv_in_csv_out(self, tmp_path):
        """Full round-trip: read CSV -> coalesce -> write CSV"""
        input_csv = tmp_path / "input.csv"
        input_csv.write_text("name,addr1,addr2\n"
                            "Alice,123 Main St,\n"
                            "Bob,,456 Oak Ave\n")
        output_csv = tmp_path / "output.csv"
        
        frame = ar.read_csv(str(input_csv))
        result = ar.coalesce_columns(
            frame,
            subset=["addr1", "addr2"],
            output_column="address"
        )
        ar.write_csv(result, str(output_csv))
        
        # Verify output file contains coalesced column
        result_df = pd.read_csv(output_csv)
        assert result_df["address"].tolist() == ["123 Main St", "456 Oak Ave"]

    def test_coalesce_error_empty_subset(self):
        """Error case: empty subset list"""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(ValueError, match="subset must contain at least one column"):
            ar.coalesce_columns(frame, subset=[], output_column="result")

    def test_coalesce_error_output_column_already_exists(self):
        """Error case: output column already in frame"""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(ValueError, match="already exists"):
            ar.coalesce_columns(frame, subset=["a"], output_column="a")

    def test_coalesce_error_missing_column_in_subset(self):
        """Error case: requested column in subset doesn't exist"""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(KeyError):
            ar.coalesce_columns(frame, subset=["a", "nonexistent"], output_column="result")

    def test_coalesce_error_invalid_output_column(self):
        """Error case: invalid output_column (empty string)"""
        df = pd.DataFrame({"a": [1, 2]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(ValueError, match="output_column must be a non-empty string"):
            ar.coalesce_columns(frame, subset=["a"], output_column="")


class TestStandardizeMissingTokens:
    """End-to-end tests for ar.standardize_missing_tokens()"""

    def test_standardize_default_tokens_replaced(self, tmp_path):
        """Default tokens (N/A, NA, nan, none, etc.) converted to NaN"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name,status\n"
                           "Alice,N/A\n"
                           "Bob,NA\n"
                           "Charlie,active\n"
                           "David,none\n"
                           "Eve,nan\n")
        
        frame = ar.read_csv(str(csv_file))
        result = ar.standardize_missing_tokens(frame)
        df = ar.to_pandas(result)
        
        assert pd.isna(df.loc[0, "status"])  # N/A
        assert pd.isna(df.loc[1, "status"])  # NA
        assert df.loc[2, "status"] == "active"
        assert pd.isna(df.loc[3, "status"])  # none
        assert pd.isna(df.loc[4, "status"])  # nan

    def test_standardize_custom_tokens(self):
        """Test with custom token list"""
        df = pd.DataFrame({
            "value": ["valid", "MISSING", "UNKNOWN", "valid", "skip"]
        })
        frame = ar.from_pandas(df)
        
        result = ar.standardize_missing_tokens(frame, tokens=["MISSING", "UNKNOWN"])
        result_df = ar.to_pandas(result)
        
        assert result_df.loc[0, "value"] == "valid"
        assert pd.isna(result_df.loc[1, "value"])  # MISSING
        assert pd.isna(result_df.loc[2, "value"])  # UNKNOWN
        assert result_df.loc[3, "value"] == "valid"
        assert result_df.loc[4, "value"] == "skip"  # not in custom tokens

    def test_standardize_case_insensitive_matching(self):
        """Token matching is case-insensitive"""
        df = pd.DataFrame({
            "value": ["N/A", "n/a", "N/A", "Na", "valid"]
        })
        frame = ar.from_pandas(df)
        
        result = ar.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)
        
        for i in range(4):
            assert pd.isna(result_df.loc[i, "value"]), f"Row {i} should be NaN"
        assert result_df.loc[4, "value"] == "valid"

    def test_standardize_whitespace_trimmed_before_matching(self):
        """Whitespace is trimmed before token matching, non-tokens preserve whitespace"""
        df = pd.DataFrame({
            "value": ["  N/A  ", "\tNA\t", " valid ", " data "]
        })
        frame = ar.from_pandas(df)
        
        result = ar.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)
        
        assert pd.isna(result_df.loc[0, "value"])  # "  N/A  " → NaN
        assert pd.isna(result_df.loc[1, "value"])  # "\tNA\t" → NaN
        assert result_df.loc[2, "value"] == " valid "  # whitespace preserved for non-tokens
        assert result_df.loc[3, "value"] == " data "

    def test_standardize_subset_columns_only(self):
        """Only standardize specified columns via subset"""
        df = pd.DataFrame({
            "col_a": ["valid", "N/A", "data"],
            "col_b": ["NA", "none", "value"]
        })
        frame = ar.from_pandas(df)
        
        result = ar.standardize_missing_tokens(frame, subset=["col_a"])
        result_df = ar.to_pandas(result)
        
        # col_a should be standardized
        assert result_df.loc[0, "col_a"] == "valid"
        assert pd.isna(result_df.loc[1, "col_a"])
        assert result_df.loc[2, "col_a"] == "data"
        
        # col_b should NOT be changed (not in subset)
        assert result_df.loc[0, "col_b"] == "NA"
        assert result_df.loc[1, "col_b"] == "none"
        assert result_df.loc[2, "col_b"] == "value"

    def test_standardize_error_tokens_as_string(self):
        """Error: tokens passed as string instead of list"""
        df = pd.DataFrame({"value": ["test"]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(TypeError, match="tokens must be a list"):
            ar.standardize_missing_tokens(frame, tokens="N/A")

    def test_standardize_error_subset_as_string(self):
        """Error: subset passed as string instead of list"""
        df = pd.DataFrame({"value": ["test"]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(TypeError, match="subset must be a list"):
            ar.standardize_missing_tokens(frame, subset="value")

    def test_standardize_in_pipeline(self, tmp_path):
        """Test standardize_missing_tokens as pipeline step"""
        csv_file = tmp_path / "messy.csv"
        csv_file.write_text("id,name,score\n"
                           "1,Alice,100\n"
                           "2,N/A,N/A\n"
                           "3,Bob,none\n")
        
        frame = ar.read_csv(str(csv_file))
        result = ar.pipeline(frame, [
            ("standardize_missing_tokens",)
        ])
        df = ar.to_pandas(result)
        
        assert df.loc[0, "name"] == "Alice"
        assert pd.isna(df.loc[1, "name"])  # N/A
        assert pd.isna(df.loc[1, "score"])  # N/A
        assert df.loc[2, "name"] == "Bob"
        assert pd.isna(df.loc[2, "score"])  # none


class TestCombinedCoalesceAndStandardize:
    """Integration tests combining both functions"""

    def test_standardize_then_coalesce_workflow(self, tmp_path):
        """Real workflow: standardize missing tokens, then coalesce columns"""
        input_csv = tmp_path / "contacts.csv"
        input_csv.write_text(
            "id,phone_1,phone_2,email_1,email_2\n"
            "1,555-1111,N/A,alice@ex.com,\n"
            "2,none,,bob@ex.com,bob.backup@ex.com\n"
            "3,NA,555-9999,,charlie@ex.com\n"
        )
        output_csv = tmp_path / "clean.csv"
        
        frame = ar.read_csv(str(input_csv))
        clean = ar.pipeline(frame, [
            ("standardize_missing_tokens",),
            ("coalesce_columns", {
                "subset": ["phone_1", "phone_2"],
                "output_column": "phone"
            }),
            ("coalesce_columns", {
                "subset": ["email_1", "email_2"],
                "output_column": "email"
            })
        ])
        ar.write_csv(clean, str(output_csv))
        
        result_df = pd.read_csv(output_csv)
        assert result_df.loc[0, "phone"] == "555-1111"
        assert result_df.loc[0, "email"] == "alice@ex.com"
        assert result_df.loc[1, "phone"] == "555-1111"  # second row: "none" becomes NaN
        assert result_df.loc[1, "email"] == "bob@ex.com"
        assert result_df.loc[2, "phone"] == "555-9999"
        assert result_df.loc[2, "email"] == "charlie@ex.com"

    def test_full_cleanup_with_profile_and_validation(self, tmp_path):
        """Full end-to-end: read, clean, profile, write"""
        input_csv = tmp_path / "data.csv"
        input_csv.write_text("user_id,name,status_1,status_2\n"
                            "1,Alice,active,N/A\n"
                            "2,none,none,approved\n"
                            "3,Charlie,NA,active\n")
        output_csv = tmp_path / "output.csv"
        
        # Read and clean
        frame = ar.read_csv(str(input_csv))
        clean = ar.pipeline(frame, [
            ("standardize_missing_tokens",),
            ("coalesce_columns", {
                "subset": ["status_1", "status_2"],
                "output_column": "status"
            })
        ])
        
        # Profile to verify quality
        report = ar.profile(clean)
        assert report.row_count == 3
        
        # Write result
        ar.write_csv(clean, str(output_csv))
        result_df = pd.read_csv(output_csv)
        
        assert result_df.loc[0, "status"] == "active"
        assert result_df.loc[1, "status"] == "approved"
        assert result_df.loc[2, "status"] == "active"

    def test_pandas_accessor_with_standardize_and_coalesce(self):
        """Test using pandas .arnio accessor with both functions"""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "first_name": ["Alice", "none", "Charlie"],
            "phone_1": ["555-1111", "N/A", None],
            "phone_2": [None, "555-2222", "555-3333"]
        })
        
        # Use accessor to chain cleaning
        result = df.arnio.clean(steps=[
            ("standardize_missing_tokens",),
            ("coalesce_columns", {
                "subset": ["phone_1", "phone_2"],
                "output_column": "phone"
            })
        ])
        
        assert pd.isna(result.loc[1, "first_name"])
        assert result.loc[0, "phone"] == "555-1111"
        assert result.loc[1, "phone"] == "555-2222"
        assert result.loc[2, "phone"] == "555-3333"
