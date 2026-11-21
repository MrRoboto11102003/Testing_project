import arnio as ar
import pandas as pd
import numpy as np
import pytest
from pathlib import Path


class TestStandardizeMissingTokens:
    """End-to-end tests for arnio.cleaning.standardize_missing_tokens."""

    def test_default_tokens_basic(self, tmp_path):
        """Test basic usage with default missing value tokens."""
        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5, 6],
            "value": ["valid", "N/A", "NA", "n/a", "null", "valid"],
        })
        csv_path = tmp_path / "data.csv"
        df.to_csv(csv_path, index=False)

        frame = ar.read_csv(str(csv_path))
        result = ar.standardize_missing_tokens(frame)
        df_result = ar.to_pandas(result)

        assert df_result.loc[0, "value"] == "valid"
        assert pd.isna(df_result.loc[1, "value"])
        assert pd.isna(df_result.loc[2, "value"])
        assert pd.isna(df_result.loc[3, "value"])
        assert pd.isna(df_result.loc[4, "value"])
        assert df_result.loc[5, "value"] == "valid"

    def test_custom_tokens(self, tmp_path):
        """Test with custom token list."""
        df = pd.DataFrame({
            "field": ["okay", "MISSING", "unknown", "okay"],
        })
        csv_path = tmp_path / "data.csv"
        df.to_csv(csv_path, index=False)

        frame = ar.read_csv(str(csv_path))
        result = ar.standardize_missing_tokens(
            frame, tokens=["MISSING", "unknown"]
        )
        df_result = ar.to_pandas(result)

        assert df_result.loc[0, "field"] == "okay"
        assert pd.isna(df_result.loc[1, "field"])
        assert pd.isna(df_result.loc[2, "field"])
        assert df_result.loc[3, "field"] == "okay"

    def test_subset_columns(self, tmp_path):
        """Test applying standardization to only specific columns."""
        df = pd.DataFrame({
            "col_a": ["valid", "N/A", "valid"],
            "col_b": ["x", "N/A", "z"],
        })
        csv_path = tmp_path / "data.csv"
        df.to_csv(csv_path, index=False)

        frame = ar.read_csv(str(csv_path))
        result = ar.standardize_missing_tokens(frame, subset=["col_a"])
        df_result = ar.to_pandas(result)

        # col_a should be cleaned
        assert pd.isna(df_result.loc[1, "col_a"])
        # col_b should NOT be cleaned (N/A remains as string)
        assert df_result.loc[1, "col_b"] == "N/A"

    def test_case_insensitive_matching(self, tmp_path):
        """Test that token matching is case-insensitive."""
        df = pd.DataFrame({
            "value": ["NULL", "null", "NuLl", "Null", "data"],
        })
        csv_path = tmp_path / "data.csv"
        df.to_csv(csv_path, index=False)

        frame = ar.read_csv(str(csv_path))
        result = ar.standardize_missing_tokens(frame)
        df_result = ar.to_pandas(result)

        for i in range(4):
            assert pd.isna(df_result.loc[i, "value"]), f"Row {i} should be NaN"
        assert df_result.loc[4, "value"] == "data"

    def test_whitespace_trimming_before_match(self, tmp_path):
        """Test that whitespace is trimmed before token matching."""
        df = pd.DataFrame({
            "value": ["  N/A  ", "\tNA\t", " null ", "data"],
        })
        csv_path = tmp_path / "data.csv"
        df.to_csv(csv_path, index=False)

        frame = ar.read_csv(str(csv_path))
        result = ar.standardize_missing_tokens(frame)
        df_result = ar.to_pandas(result)

        assert pd.isna(df_result.loc[0, "value"])
        assert pd.isna(df_result.loc[1, "value"])
        assert pd.isna(df_result.loc[2, "value"])
        assert df_result.loc[3, "value"] == "data"

    def test_preserves_whitespace_for_non_matching(self, tmp_path):
        """Test that non-matching values preserve their original whitespace."""
        df = pd.DataFrame({
            "value": ["  data  ", "\ttext\t", " valid "],
        })
        csv_path = tmp_path / "data.csv"
        df.to_csv(csv_path, index=False)

        frame = ar.read_csv(str(csv_path))
        result = ar.standardize_missing_tokens(frame)
        df_result = ar.to_pandas(result)

        assert df_result.loc[0, "value"] == "  data  "
        assert df_result.loc[1, "value"] == "\ttext\t"
        assert df_result.loc[2, "value"] == " valid "

    def test_non_string_values_unchanged(self, tmp_path):
        """Test that non-string values are not affected."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "numeric": [1.5, 2.5, 3.5],
            "text": ["data", "N/A", "data"],
        })
        csv_path = tmp_path / "data.csv"
        df.to_csv(csv_path, index=False)

        frame = ar.read_csv(str(csv_path))
        result = ar.standardize_missing_tokens(frame)
        df_result = ar.to_pandas(result)

        assert df_result.loc[0, "numeric"] == 1.5
        assert df_result.loc[1, "numeric"] == 2.5
        assert df_result.loc[2, "numeric"] == 3.5
        assert pd.isna(df_result.loc[1, "text"])

    def test_in_pipeline(self, tmp_path):
        """Test using standardize_missing_tokens within a pipeline."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["alice", "N/A", "charlie"],
            "value": ["  x  ", "  y  ", "  ?  "],
        })
        csv_path = tmp_path / "data.csv"
        df.to_csv(csv_path, index=False)

        frame = ar.read_csv(str(csv_path))
        clean = ar.pipeline(
            frame,
            [
                ("standardize_missing_tokens",),
                ("strip_whitespace",),
            ],
        )
        df_result = ar.to_pandas(clean)

        assert df_result.loc[0, "name"] == "alice"
        assert pd.isna(df_result.loc[1, "name"])
        assert df_result.loc[0, "value"] == "x"
        assert df_result.loc[2, "value"] == "?"

    def test_pandas_accessor(self, tmp_path):
        """Test using pandas .arnio accessor with standardize_missing_tokens."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "status": ["active", "null", "active"],
        })
        csv_path = tmp_path / "data.csv"
        df.to_csv(csv_path, index=False)

        df_loaded = pd.read_csv(csv_path)
        result_df = df_loaded.arnio.clean(
            [
                ("standardize_missing_tokens",),
            ]
        )

        assert result_df.loc[0, "status"] == "active"
        assert pd.isna(result_df.loc[1, "status"])
        assert result_df.loc[2, "status"] == "active"

    def test_csv_roundtrip(self, tmp_path):
        """Test full workflow: load CSV, clean, save CSV, and verify output."""
        input_path = tmp_path / "input.csv"
        output_path = tmp_path / "output.csv"

        df = pd.DataFrame({
            "id": [1, 2, 3, 4],
            "email": ["user@example.com", "N/A", "admin@test.com", "NA"],
        })
        df.to_csv(input_path, index=False)

        frame = ar.read_csv(str(input_path))
        clean = ar.standardize_missing_tokens(frame)
        ar.write_csv(clean, str(output_path))

        result_df = pd.read_csv(output_path)
        assert result_df.loc[0, "email"] == "user@example.com"
        assert result_df.loc[2, "email"] == "admin@test.com"

    def test_error_subset_string(self, tmp_path):
        """Test that passing subset as string raises TypeError."""
        df = pd.DataFrame({"value": ["data", "N/A"]})
        csv_path = tmp_path / "data.csv"
        df.to_csv(csv_path, index=False)

        frame = ar.read_csv(str(csv_path))

        with pytest.raises(TypeError, match="subset must be a list"):
            ar.standardize_missing_tokens(frame, subset="value")

    def test_error_tokens_string(self, tmp_path):
        """Test that passing tokens as string raises TypeError."""
        df = pd.DataFrame({"value": ["data", "N/A"]})
        csv_path = tmp_path / "data.csv"
        df.to_csv(csv_path, index=False)

        frame = ar.read_csv(str(csv_path))

        with pytest.raises(TypeError, match="tokens must be a list"):
            ar.standardize_missing_tokens(frame, tokens="N/A")

    def test_empty_frame(self, tmp_path):
        """Test with empty frame preserves schema."""
        df = pd.DataFrame(
            {"id": pd.Series([], dtype=int), "value": pd.Series([], dtype=str)}
        )
        csv_path = tmp_path / "data.csv"
        df.to_csv(csv_path, index=False)

        frame = ar.read_csv(str(csv_path))
        result = ar.standardize_missing_tokens(frame)
        df_result = ar.to_pandas(result)

        assert len(df_result) == 0
        assert list(df_result.columns) == ["id", "value"]

    def test_all_values_are_missing_tokens(self, tmp_path):
        """Test when all column values match missing tokens."""
        df = pd.DataFrame({
            "value": ["N/A", "null", "NA", "-"],
        })
        csv_path = tmp_path / "data.csv"
        df.to_csv(csv_path, index=False)

        frame = ar.read_csv(str(csv_path))
        result = ar.standardize_missing_tokens(frame)
        df_result = ar.to_pandas(result)

        assert all(pd.isna(df_result["value"]))

    def test_accepts_pandas_dataframe_directly(self):
        """Test that standardize_missing_tokens accepts pandas DataFrame directly."""
        df = pd.DataFrame({
            "value": ["valid", "N/A", "null", "valid"],
        })

        # Pass pandas DataFrame directly (not ArFrame)
        result = ar.standardize_missing_tokens(df)

        # Result should be pandas DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.loc[0, "value"] == "valid"
        assert pd.isna(result.loc[1, "value"])
        assert pd.isna(result.loc[2, "value"])
        assert result.loc[3, "value"] == "valid"
