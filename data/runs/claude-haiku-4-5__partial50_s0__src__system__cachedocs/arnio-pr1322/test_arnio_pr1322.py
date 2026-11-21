import pytest
import pandas as pd
import numpy as np
import arnio as ar
import io


class TestCoalesceColumnsIntegration:
    """End-to-end tests for arnio.cleaning.coalesce_columns."""

    def test_coalesce_basic_arframe(self, tmp_path):
        """Test coalesce_columns with ArFrame from CSV file."""
        csv_file = tmp_path / "contacts.csv"
        csv_file.write_text(
            "name,work_phone,cell_phone,home_phone\n"
            "Alice,555-0001,,\n"
            "Bob,,555-0002,\n"
            "Charlie,,,555-0003\n"
        )

        frame = ar.read_csv(str(csv_file))
        result = ar.coalesce_columns(
            frame,
            subset=["work_phone", "cell_phone", "home_phone"],
            output_column="contact",
        )

        df = ar.to_pandas(result)
        assert "contact" in df.columns
        assert df.loc[0, "contact"] == "555-0001"
        assert df.loc[1, "contact"] == "555-0002"
        assert df.loc[2, "contact"] == "555-0003"

    def test_coalesce_with_pandas_dataframe(self):
        """Test coalesce_columns accepts pandas DataFrame as input."""
        df = pd.DataFrame({
            "first_choice": [None, "B", None],
            "second_choice": ["A", None, "C"],
            "third_choice": ["X", "Y", None],
        })

        result = ar.coalesce_columns(
            df, subset=["first_choice", "second_choice", "third_choice"], output_column="best"
        )

        assert isinstance(result, pd.DataFrame)
        assert result.loc[0, "best"] == "A"
        assert result.loc[1, "best"] == "B"
        assert result.loc[2, "best"] == "C"

    def test_coalesce_roundtrip_csv(self, tmp_path):
        """Full workflow: CSV -> coalesce -> CSV."""
        input_file = tmp_path / "input.csv"
        output_file = tmp_path / "output.csv"

        input_file.write_text(
            "id,alt1,alt2,alt3\n"
            "1,a,,\n"
            "2,,b,\n"
            "3,,,c\n"
        )

        frame = ar.read_csv(str(input_file))
        result = ar.coalesce_columns(
            frame, subset=["alt1", "alt2", "alt3"], output_column="chosen"
        )
        ar.write_csv(result, str(output_file))

        # Verify output file contains the coalesced column
        output_text = output_file.read_text()
        assert "chosen" in output_text
        lines = output_text.strip().split("\n")
        assert len(lines) == 4

    def test_coalesce_error_output_column_exists(self, tmp_path):
        """Error when output_column already exists in frame."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b,c\n1,2,3\n")

        frame = ar.read_csv(str(csv_file))
        with pytest.raises(ValueError, match="already exists"):
            ar.coalesce_columns(frame, subset=["a", "b"], output_column="c")

    def test_coalesce_error_empty_subset(self, tmp_path):
        """Error when subset is empty."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2\n")

        frame = ar.read_csv(str(csv_file))
        with pytest.raises(ValueError, match="subset must contain"):
            ar.coalesce_columns(frame, subset=[], output_column="result")

    def test_coalesce_error_missing_column_in_subset(self, tmp_path):
        """Error when a column in subset doesn't exist."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2\n")

        frame = ar.read_csv(str(csv_file))
        with pytest.raises((KeyError, ValueError)):
            ar.coalesce_columns(frame, subset=["a", "nonexistent"], output_column="result")

    def test_coalesce_error_empty_output_column_name(self, tmp_path):
        """Error when output_column is empty string."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2\n")

        frame = ar.read_csv(str(csv_file))
        with pytest.raises(ValueError, match="output_column must be"):
            ar.coalesce_columns(frame, subset=["a"], output_column="")


class TestStandardizeMissingTokensIntegration:
    """End-to-end tests for arnio.cleaning.standardize_missing_tokens."""

    def test_standardize_default_tokens(self, tmp_path):
        """Test standardize with default null tokens from CSV."""
        csv_file = tmp_path / "survey.csv"
        csv_file.write_text(
            "id,score,comment\n"
            "1,5,good\n"
            "2,N/A,skipped\n"
            "3,null,error\n"
            "4,,no entry\n"
        )

        frame = ar.read_csv(str(csv_file))
        result = ar.standardize_missing_tokens(frame)
        df = ar.to_pandas(result)

        # Verify default tokens are converted to NaN
        assert df.loc[0, "score"] == "5"
        assert pd.isna(df.loc[1, "score"])  # N/A -> NaN
        assert pd.isna(df.loc[2, "score"])  # null -> NaN
        assert pd.isna(df.loc[3, "score"])  # empty -> NaN

    def test_standardize_custom_tokens(self, tmp_path):
        """Test standardize with custom token list."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            "value\n"
            "100\n"
            "MISSING\n"
            "999\n"
            "ERROR\n"
        )

        frame = ar.read_csv(str(csv_file))
        result = ar.standardize_missing_tokens(frame, tokens=["MISSING", "999", "ERROR"])
        df = ar.to_pandas(result)

        assert df.loc[0, "value"] == "100"
        assert pd.isna(df.loc[1, "value"])
        assert pd.isna(df.loc[2, "value"])
        assert pd.isna(df.loc[3, "value"])

    def test_standardize_subset_columns(self, tmp_path):
        """Test standardize applies only to subset columns."""
        csv_file = tmp_path / "mixed.csv"
        csv_file.write_text(
            "id,name,status\n"
            "1,Alice,active\n"
            "2,N/A,N/A\n"
        )

        frame = ar.read_csv(str(csv_file))
        result = ar.standardize_missing_tokens(frame, subset=["status"])
        df = ar.to_pandas(result)

        # 'name' column should NOT be standardized
        assert df.loc[1, "name"] == "N/A"
        # 'status' column should be standardized
        assert pd.isna(df.loc[1, "status"])

    def test_standardize_case_insensitive(self, tmp_path):
        """Test case-insensitive token matching."""
        csv_file = tmp_path / "case.csv"
        csv_file.write_text(
            "value\n"
            "DATA\n"
            "null\n"
            "NULL\n"
            "NuLl\n"
        )

        frame = ar.read_csv(str(csv_file))
        result = ar.standardize_missing_tokens(frame)
        df = ar.to_pandas(result)

        assert df.loc[0, "value"] == "DATA"
        assert pd.isna(df.loc[1, "value"])
        assert pd.isna(df.loc[2, "value"])
        assert pd.isna(df.loc[3, "value"])

    def test_standardize_in_pipeline(self, tmp_path):
        """Test standardize as a pipeline step."""
        csv_file = tmp_path / "messy.csv"
        csv_file.write_text(
            "id,amount,status\n"
            "1,100,active\n"
            "2,N/A,pending\n"
            "3,,null\n"
        )

        frame = ar.read_csv(str(csv_file))
        clean = ar.pipeline(
            frame,
            [
                ("standardize_missing_tokens", {}),
                ("drop_nulls",),
            ],
        )

        df = ar.to_pandas(clean)
        # Only first row has no nulls after standardize
        assert len(df) == 1
        assert df.loc[0, "id"] == "1"

    def test_standardize_with_pandas_input(self):
        """Test standardize accepts pandas DataFrame."""
        df = pd.DataFrame({
            "code": ["ABC", "N/A", "XYZ", None],
            "value": [10, 20, 30, 40],
        })

        result = ar.standardize_missing_tokens(df)
        assert isinstance(result, pd.DataFrame)
        assert result.loc[0, "code"] == "ABC"
        assert pd.isna(result.loc[1, "code"])
        assert result.loc[2, "code"] == "XYZ"

    def test_standardize_error_subset_as_string(self, tmp_path):
        """Error when subset is a string instead of list."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2\n")

        frame = ar.read_csv(str(csv_file))
        with pytest.raises(TypeError, match="subset must be"):
            ar.standardize_missing_tokens(frame, subset="a")

    def test_standardize_error_tokens_as_string(self, tmp_path):
        """Error when tokens is a string instead of list."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a\n1\n")

        frame = ar.read_csv(str(csv_file))
        with pytest.raises(TypeError, match="tokens must be"):
            ar.standardize_missing_tokens(frame, tokens="null")

    def test_standardize_error_unknown_subset_column(self, tmp_path):
        """Error when subset contains unknown column."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2\n")

        frame = ar.read_csv(str(csv_file))
        with pytest.raises((ValueError, KeyError)):
            ar.standardize_missing_tokens(frame, subset=["nonexistent"])

    def test_standardize_full_workflow_with_profile(self, tmp_path):
        """Full workflow: CSV -> standardize -> profile -> write."""
        input_file = tmp_path / "input.csv"
        output_file = tmp_path / "output.csv"

        input_file.write_text(
            "customer,amount,status\n"
            "Alice,1000,active\n"
            "Bob,N/A,prospect\n"
            "Charlie,,null\n"
        )

        frame = ar.read_csv(str(input_file))

        # Profile before cleaning
        report_before = ar.profile(frame)
        summary_before = report_before.summary()
        assert summary_before["row_count"] == 3

        # Apply standardization
        result = ar.standardize_missing_tokens(frame)

        # Optionally write cleaned output
        ar.write_csv(result, str(output_file))
        assert output_file.exists()

        # Verify output file is valid
        frame_read_back = ar.read_csv(str(output_file))
        assert frame_read_back.shape[0] == 3
