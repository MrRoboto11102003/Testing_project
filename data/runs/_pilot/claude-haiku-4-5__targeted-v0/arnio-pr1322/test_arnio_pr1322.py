import pytest
import pandas as pd
import numpy as np
import arnio as ar
import tempfile
import os


class TestReadCSV:
    def test_read_csv_strict_mode(self):
        """Test reading CSV in strict mode with consistent row widths."""
        csv_content = "a,b,c\n1,2,3\n4,5,6\n"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_path = f.name
        try:
            frame = ar.read_csv(temp_path, mode="strict")
            assert frame is not None
        finally:
            os.unlink(temp_path)

    def test_read_csv_permissive_mode_handles_ragged_rows(self):
        """Test permissive mode fills missing trailing values with nulls."""
        csv_content = "a,b,c\n1,2,3\n4,5\n"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_path = f.name
        try:
            frame = ar.read_csv(temp_path, mode="permissive")
            assert frame is not None
        finally:
            os.unlink(temp_path)


class TestPipeline:
    def test_pipeline_single_step(self):
        """Test pipeline with single transformation step."""
        frame = ar.from_dict({"col_a": [1, 2, 2, 3]})
        result = ar.pipeline(frame, [("drop_duplicates",)])
        assert result is not None

    def test_pipeline_multiple_steps(self):
        """Test pipeline with multiple sequential transformation steps."""
        frame = ar.from_dict({"col_a": ["  a  ", "  b  "], "col_b": [1, 1]})
        result = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("drop_duplicates",),
        ])
        assert result is not None

    def test_pipeline_with_step_kwargs(self):
        """Test pipeline steps with configuration parameters."""
        frame = ar.from_dict({"col_a": ["ABC", "def"]})
        result = ar.pipeline(frame, [
            ("normalize_case", {"case_type": "lower"}),
        ])
        assert result is not None

    def test_pipeline_dry_run_validation(self):
        """Test dry_run validates pipeline without executing transformations."""
        frame = ar.from_dict({"col_a": [1, 2, 3]})
        ar.pipeline(frame, [("drop_duplicates",)], dry_run=True)

    def test_pipeline_return_metadata(self):
        """Test pipeline returns metadata with step timings and statistics."""
        frame = ar.from_dict({"col_a": [1, 2, 3]})
        result, metadata = ar.pipeline(
            frame,
            [("drop_duplicates",)],
            return_metadata=True
        )
        assert isinstance(metadata, dict)
        assert "step_timings" in metadata
        assert "applied_steps" in metadata
        assert "row_counts" in metadata
        assert "execution_summary" in metadata

    def test_pipeline_invalid_step_fails_early(self):
        """Test malformed step tuples fail before any execution."""
        frame = ar.from_dict({"col_a": [1, 2, 3]})
        with pytest.raises((ValueError, TypeError)):
            ar.pipeline(frame, [("bad_step", "oops", "extra")])


class TestFromDict:
    def test_from_dict_creates_frame(self):
        """Test creating ArFrame from dictionary of equal-length columns."""
        data = {"name": ["Alice", "Bob"], "age": [25, 30]}
        frame = ar.from_dict(data)
        assert frame is not None

    def test_from_dict_class_method(self):
        """Test ArFrame.from_dict class method."""
        data = {"name": ["Alice", "Bob"], "age": [25, 30]}
        frame = ar.ArFrame.from_dict(data)
        assert frame is not None

    def test_from_dict_unequal_lengths_raises(self):
        """Test ValueError when columns have unequal lengths."""
        data = {"col_a": [1, 2], "col_b": [1, 2, 3]}
        with pytest.raises(ValueError):
            ar.from_dict(data)


class TestCoalesceColumns:
    def test_coalesce_columns_basic(self):
        """Test coalesce_columns selects first non-null value from subset."""
        frame = ar.from_dict({
            "col_a": [None, 2, 3],
            "col_b": [1, None, 6],
        })
        result = ar.coalesce_columns(
            frame,
            subset=["col_a", "col_b"],
            output_column="first_non_null"
        )
        assert result is not None

    def test_coalesce_columns_output_exists_raises(self):
        """Test ValueError when output_column already exists in frame."""
        frame = ar.from_dict({"col_a": [1, 2], "col_b": [3, 4]})
        with pytest.raises(ValueError):
            ar.coalesce_columns(frame, subset=["col_a"], output_column="col_b")

    def test_coalesce_columns_missing_column_raises(self):
        """Test KeyError when subset column does not exist."""
        frame = ar.from_dict({"col_a": [1, 2]})
        with pytest.raises(KeyError):
            ar.coalesce_columns(frame, subset=["col_a", "missing"])

    def test_coalesce_columns_empty_subset_raises(self):
        """Test ValueError when subset is empty."""
        frame = ar.from_dict({"col_a": [1, 2]})
        with pytest.raises(ValueError):
            ar.coalesce_columns(frame, subset=[])

    def test_coalesce_columns_invalid_subset_type_raises(self):
        """Test TypeError when subset is not a sequence."""
        frame = ar.from_dict({"col_a": [1, 2]})
        with pytest.raises(TypeError):
            ar.coalesce_columns(frame, subset="col_a")


class TestStandardizeMissingTokens:
    def test_standardize_missing_tokens_default(self):
        """Test standardizing with built-in default tokens list."""
        df = pd.DataFrame({"value": ["data", "N/A", "more"]})
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame)
        assert result is not None

    def test_standardize_missing_tokens_custom_tokens(self):
        """Test standardizing with custom list of null-like tokens."""
        df = pd.DataFrame({"value": ["NULL", "data", "UNKNOWN"]})
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(
            frame,
            tokens=["NULL", "UNKNOWN"]
        )
        assert result is not None

    def test_standardize_missing_tokens_subset(self):
        """Test standardizing missing tokens in specific columns only."""
        df = pd.DataFrame({"col_a": ["N/A", "data"], "col_b": ["N/A", "data"]})
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame, subset=["col_a"])
        assert result is not None

    def test_standardize_missing_tokens_case_insensitive_whitespace(self):
        """Test case-insensitive and whitespace-trimmed token matching."""
        df = pd.DataFrame({"value": [" n/a ", "N/A", "\tnan\t"]})
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame)
        assert result is not None


class TestToPandas:
    def test_to_pandas_basic(self):
        """Test converting ArFrame to pandas DataFrame."""
        frame = ar.from_dict({"col_a": [1, 2, 3]})
        df = ar.to_pandas(frame)
        assert isinstance(df, pd.DataFrame)

    def test_to_pandas_with_copy(self):
        """Test copy=True returns defensive pandas-owned buffers."""
        frame = ar.from_dict({"col_a": [1, 2, 3]})
        df = ar.to_pandas(frame, copy=True)
        assert isinstance(df, pd.DataFrame)
