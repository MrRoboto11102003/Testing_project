import sys
import subprocess
import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestCoalesceColumns:
    """End-to-end tests for arnio.cleaning.coalesce_columns."""

    def test_coalesce_basic_first_non_null(self, tmp_path):
        """Test basic coalescing selects first non-null value across columns."""
        df = pd.DataFrame({
            "first": [None, 2.0, 3.0],
            "second": [1.0, None, None],
            "third": [10.0, 20.0, 30.0]
        })
        
        input_file = tmp_path / "input.csv"
        df.to_csv(input_file, index=False)
        
        frame = ar.read_csv(str(input_file))
        result = ar.coalesce_columns(
            frame,
            subset=["first", "second", "third"],
            output_column="coalesced"
        )
        
        result_df = result.to_pandas() if hasattr(result, 'to_pandas') else result
        assert list(result_df["coalesced"]) == [1.0, 2.0, 3.0]

    def test_coalesce_all_null_row_produces_nan(self, tmp_path):
        """Test coalescing when entire row is null produces NaN."""
        df = pd.DataFrame({
            "a": [None, 1.0],
            "b": [None, None]
        })
        
        input_file = tmp_path / "input.csv"
        df.to_csv(input_file, index=False)
        
        frame = ar.read_csv(str(input_file))
        result = ar.coalesce_columns(frame, subset=["a", "b"], output_column="result")
        
        result_df = result.to_pandas() if hasattr(result, 'to_pandas') else result
        assert pd.isna(result_df["result"].iloc[0])
        assert result_df["result"].iloc[1] == 1.0

    def test_coalesce_errors_on_existing_output_column(self, tmp_path):
        """Test error when output column already exists in frame."""
        df = pd.DataFrame({"a": [1.0], "b": [2.0], "result": [3.0]})
        
        input_file = tmp_path / "input.csv"
        df.to_csv(input_file, index=False)
        
        frame = ar.read_csv(str(input_file))
        with pytest.raises(ValueError, match="already exists"):
            ar.coalesce_columns(frame, subset=["a", "b"], output_column="result")

    def test_coalesce_errors_on_missing_subset_column(self, tmp_path):
        """Test error when subset references non-existent column."""
        df = pd.DataFrame({"a": [1.0]})
        
        input_file = tmp_path / "input.csv"
        df.to_csv(input_file, index=False)
        
        frame = ar.read_csv(str(input_file))
        with pytest.raises(KeyError):
            ar.coalesce_columns(frame, subset=["a", "nonexistent"], output_column="result")

    def test_coalesce_errors_on_empty_subset(self, tmp_path):
        """Test error when subset is empty."""
        df = pd.DataFrame({"a": [1.0]})
        
        input_file = tmp_path / "input.csv"
        df.to_csv(input_file, index=False)
        
        frame = ar.read_csv(str(input_file))
        with pytest.raises(ValueError, match="at least one column"):
            ar.coalesce_columns(frame, subset=[], output_column="result")

    def test_coalesce_write_result_to_file(self, tmp_path):
        """Test coalesced result can be written to file."""
        df = pd.DataFrame({"a": [None, 1.0], "b": [2.0, None]})
        
        input_file = tmp_path / "input.csv"
        df.to_csv(input_file, index=False)
        output_file = tmp_path / "output.csv"
        
        frame = ar.read_csv(str(input_file))
        result = ar.coalesce_columns(frame, subset=["a", "b"], output_column="coalesced")
        result.to_csv(str(output_file))
        
        assert output_file.exists()
        output_df = pd.read_csv(output_file)
        assert "coalesced" in output_df.columns
        assert output_df["coalesced"].iloc[0] == 2.0
        assert output_df["coalesced"].iloc[1] == 1.0


class TestStandardizeMissingTokens:
    """End-to-end tests for arnio.cleaning.standardize_missing_tokens."""

    def test_standardize_default_missing_tokens(self, tmp_path):
        """Test standardizing with default missing tokens list."""
        df = pd.DataFrame({
            "col": ["value", "N/A", "NA", "null", "nil", "none", "-", "?", "nan", ""]
        })
        
        input_file = tmp_path / "input.csv"
        df.to_csv(input_file, index=False)
        
        frame = ar.read_csv(str(input_file))
        result = ar.standardize_missing_tokens(frame)
        
        result_df = result.to_pandas() if hasattr(result, 'to_pandas') else result
        assert result_df["col"].iloc[0] == "value"
        for i in range(1, 10):
            assert pd.isna(result_df["col"].iloc[i])

    def test_standardize_custom_tokens(self, tmp_path):
        """Test standardizing with custom missing tokens."""
        df = pd.DataFrame({"col": ["valid", "CUSTOM", "SPECIAL", "NORMAL"]})
        
        input_file = tmp_path / "input.csv"
        df.to_csv(input_file, index=False)
        
        frame = ar.read_csv(str(input_file))
        result = ar.standardize_missing_tokens(frame, tokens=["CUSTOM", "SPECIAL"])
        
        result_df = result.to_pandas() if hasattr(result, 'to_pandas') else result
        assert result_df["col"].iloc[0] == "valid"
        assert pd.isna(result_df["col"].iloc[1])
        assert pd.isna(result_df["col"].iloc[2])
        assert result_df["col"].iloc[3] == "NORMAL"

    def test_standardize_case_and_whitespace_insensitive(self, tmp_path):
        """Test case-insensitive and whitespace-trimming behavior."""
        df = pd.DataFrame({"col": [" N/A ", "  NULL  ", "\tnan\t", "nOrMaL"]})
        
        input_file = tmp_path / "input.csv"
        df.to_csv(input_file, index=False)
        
        frame = ar.read_csv(str(input_file))
        result = ar.standardize_missing_tokens(frame)
        
        result_df = result.to_pandas() if hasattr(result, 'to_pandas') else result
        assert pd.isna(result_df["col"].iloc[0])
        assert pd.isna(result_df["col"].iloc[1])
        assert pd.isna(result_df["col"].iloc[2])
        assert result_df["col"].iloc[3] == "nOrMaL"

    def test_standardize_subset_columns(self, tmp_path):
        """Test applying standardization only to specified columns."""
        df = pd.DataFrame({
            "standardize_me": ["NA", "value"],
            "ignore_me": ["NA", "data"]
        })
        
        input_file = tmp_path / "input.csv"
        df.to_csv(input_file, index=False)
        
        frame = ar.read_csv(str(input_file))
        result = ar.standardize_missing_tokens(frame, subset=["standardize_me"])
        
        result_df = result.to_pandas() if hasattr(result, 'to_pandas') else result
        assert pd.isna(result_df["standardize_me"].iloc[0])
        assert result_df["standardize_me"].iloc[1] == "value"
        assert result_df["ignore_me"].iloc[0] == "NA"

    def test_standardize_errors_on_tokens_as_string(self, tmp_path):
        """Test error when tokens parameter is string instead of list."""
        df = pd.DataFrame({"col": ["value"]})
        
        input_file = tmp_path / "input.csv"
        df.to_csv(input_file, index=False)
        
        frame = ar.read_csv(str(input_file))
        with pytest.raises(TypeError, match="must be a list"):
            ar.standardize_missing_tokens(frame, tokens="NA")

    def test_standardize_errors_on_subset_as_string(self, tmp_path):
        """Test error when subset parameter is string instead of list."""
        df = pd.DataFrame({"col": ["value"]})
        
        input_file = tmp_path / "input.csv"
        df.to_csv(input_file, index=False)
        
        frame = ar.read_csv(str(input_file))
        with pytest.raises(TypeError, match="must be a list"):
            ar.standardize_missing_tokens(frame, subset="col")

    def test_standardize_write_result_to_file(self, tmp_path):
        """Test standardized result can be written to file."""
        df = pd.DataFrame({"col": ["valid", "N/A", "data"]})
        
        input_file = tmp_path / "input.csv"
        df.to_csv(input_file, index=False)
        output_file = tmp_path / "output.csv"
        
        frame = ar.read_csv(str(input_file))
        result = ar.standardize_missing_tokens(frame)
        result.to_csv(str(output_file))
        
        assert output_file.exists()
        output_df = pd.read_csv(output_file)
        assert output_df["col"].iloc[0] == "valid"
        assert pd.isna(output_df["col"].iloc[1])
        assert output_df["col"].iloc[2] == "data"
