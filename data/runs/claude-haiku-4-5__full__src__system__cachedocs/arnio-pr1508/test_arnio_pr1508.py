"""
System (end-to-end) tests for arnio.pipeline.pipeline

Tests the complete user workflow: loading CSVs, running pipelines with various options,
tracking lineage, returning metadata, handling errors, and converting results.
"""

import pytest
import csv
import logging
import sys
from pathlib import Path
from time import sleep

import pandas as pd
import numpy as np
import arnio as ar


@pytest.fixture
def sample_csv(tmp_path):
    """Create a sample CSV file with messy data."""
    csv_path = tmp_path / "sample.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "name", "age", "city"])
        writer.writerow(["1", "  alice  ", "30", "new york"])
        writer.writerow(["2", "  bob  ", "25", "london"])
        writer.writerow(["3", "  alice  ", "30", "new york"])  # duplicate
        writer.writerow(["4", "charlie", "", "paris"])  # empty age
        writer.writerow(["5", "diana", "28", "tokyo"])
    return csv_path


@pytest.fixture
def numeric_csv(tmp_path):
    """Create a CSV with numeric columns."""
    csv_path = tmp_path / "numeric.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "value", "flag"])
        writer.writerow(["1", "100", "true"])
        writer.writerow(["2", "50", "false"])
        writer.writerow(["3", "200", "true"])
        writer.writerow(["4", "75", ""])  # empty flag
    return csv_path


class TestPipelineBasicExecution:
    """Test basic pipeline execution workflows."""

    def test_pipeline_simple_strip_and_drop_nulls(self, sample_csv):
        """Test simple pipeline: strip whitespace, drop rows with null age."""
        frame = ar.read_csv(sample_csv)
        
        clean = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("drop_nulls", {"subset": ["age"]}),
        ])
        
        df = ar.to_pandas(clean)
        
        # Should have 4 rows (charlie with empty age removed)
        assert len(df) == 4
        # Names should be stripped
        assert df.loc[0, "name"] == "alice"
        assert df.loc[1, "name"] == "bob"

    def test_pipeline_with_deduplication(self, sample_csv):
        """Test pipeline with drop_duplicates step."""
        frame = ar.read_csv(sample_csv)
        
        clean = ar.pipeline(frame, [
            ("drop_duplicates",),
        ])
        
        df = ar.to_pandas(clean)
        # Row 3 (0-indexed) is a duplicate of row 1
        assert len(df) == 4

    def test_pipeline_multiple_sequential_steps(self, sample_csv):
        """Test pipeline combining multiple cleaning steps."""
        frame = ar.read_csv(sample_csv)
        
        clean = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("drop_nulls", {"subset": ["age"]}),
            ("drop_duplicates",),
        ])
        
        df = ar.to_pandas(clean)
        # 5 rows - 1 null - 1 duplicate = 3
        assert len(df) == 3

    def test_pipeline_csv_to_csv_full_workflow(self, sample_csv, tmp_path):
        """Test complete workflow: read CSV, clean, write CSV."""
        output_csv = tmp_path / "output.csv"
        
        frame = ar.read_csv(sample_csv)
        clean = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("drop_nulls", {"subset": ["age"]}),
            ("drop_duplicates",),
        ])
        
        ar.write_csv(clean, output_csv)
        
        # Verify output exists and can be read
        assert output_csv.exists()
        result_frame = ar.read_csv(output_csv)
        result_df = ar.to_pandas(result_frame)
        assert len(result_df) == 3


class TestPipelineMetadata:
    """Test metadata collection and reporting."""

    def test_pipeline_return_metadata_structure(self, sample_csv):
        """Test that return_metadata provides complete timing and row count info."""
        frame = ar.read_csv(sample_csv)
        
        clean, metadata = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("drop_duplicates",),
        ], return_metadata=True)
        
        # Check metadata keys exist
        assert "step_timings" in metadata
        assert "applied_steps" in metadata
        assert "row_counts" in metadata
        assert "execution_summary" in metadata
        
        # Check timing entries
        assert len(metadata["step_timings"]) == 2
        assert all(isinstance(t["seconds"], (float, int)) 
                   for t in metadata["step_timings"])
        assert all("dry_run" in t for t in metadata["step_timings"])
        
        # Check applied steps
        assert metadata["applied_steps"] == ["strip_whitespace", "drop_duplicates"]
        
        # Check row counts
        assert len(metadata["row_counts"]) == 2
        assert metadata["row_counts"][0]["step"] == "strip_whitespace"
        assert metadata["row_counts"][1]["step"] == "drop_duplicates"

    def test_pipeline_metadata_row_counts_accurate(self, sample_csv):
        """Test that row count tracking in metadata is accurate."""
        frame = ar.read_csv(sample_csv)
        initial_rows = frame.shape[0]
        
        clean, metadata = ar.pipeline(frame, [
            ("drop_nulls", {"subset": ["age"]}),
        ], return_metadata=True)
        
        row_counts = metadata["row_counts"]
        assert row_counts[0]["before"] == initial_rows
        assert row_counts[0]["after"] == initial_rows - 1
        # charlie row (index 3) has empty age

    def test_pipeline_metadata_execution_summary(self, sample_csv):
        """Test that execution_summary contains diagnostics for each step."""
        frame = ar.read_csv(sample_csv)
        
        clean, metadata = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("drop_nulls", {"subset": ["age"]}),
        ], return_metadata=True)
        
        summary = metadata["execution_summary"]
        assert "total_runtime_ms" in summary
        assert isinstance(summary["total_runtime_ms"], (int, float))
        assert len(summary["steps"]) == 2
        
        for step_diag in summary["steps"]:
            assert "name" in step_diag
            assert "runtime_ms" in step_diag
            assert "rows_before" in step_diag
            assert "rows_after" in step_diag


class TestPipelineLineageTracking:
    """Test row-level lineage tracking functionality."""

    def test_pipeline_track_lineage_single_drop_step(self, sample_csv):
        """Test lineage tracking identifies which rows were dropped by drop_nulls."""
        frame = ar.read_csv(sample_csv)
        
        result, lineage = ar.pipeline(frame, [
            ("drop_nulls", {"subset": ["age"]}),
        ], track_lineage=True)
        
        # Check lineage report structure
        assert hasattr(lineage, "dropped_by_step")
        assert hasattr(lineage, "total_dropped")
        assert isinstance(lineage.dropped_by_step, dict)
        
        # drop_nulls should have dropped row 3 (charlie with empty age)
        assert "drop_nulls" in lineage.dropped_by_step
        assert 3 in lineage.dropped_by_step["drop_nulls"]
        assert lineage.total_dropped == 1

    def test_pipeline_track_lineage_multiple_steps(self, sample_csv):
        """Test lineage tracking with multiple row-dropping steps."""
        frame = ar.read_csv(sample_csv)
        
        result, lineage = ar.pipeline(frame, [
            ("drop_nulls", {"subset": ["age"]}),
            ("drop_duplicates",),
        ], track_lineage=True)
        
        # Both steps should be tracked
        assert "drop_nulls" in lineage.dropped_by_step
        assert "drop_duplicates" in lineage.dropped_by_step
        assert lineage.total_dropped == 2

    def test_pipeline_track_lineage_repeated_steps_merge(self, numeric_csv):
        """Test that repeated step names merge their dropped indices."""
        frame = ar.read_csv(numeric_csv)
        
        # First drop_nulls drops rows with null 'flag',
        # second drop_nulls drops nothing (no more nulls)
        result, lineage = ar.pipeline(frame, [
            ("drop_nulls", {"subset": ["flag"]}),
            ("drop_nulls", {"subset": ["value"]}),
        ], track_lineage=True)
        
        # All drops should be under single "drop_nulls" key
        assert "drop_nulls" in lineage.dropped_by_step
        # Indices should be sorted
        indices = lineage.dropped_by_step["drop_nulls"]
        assert indices == sorted(indices)

    def test_pipeline_lineage_incompatible_with_metadata(self, sample_csv):
        """Test that track_lineage=True and return_metadata=True raise ValueError."""
        frame = ar.read_csv(sample_csv)
        
        with pytest.raises(ValueError, match="cannot be used together"):
            ar.pipeline(frame, [
                ("strip_whitespace",),
            ], track_lineage=True, return_metadata=True)

    def test_pipeline_lineage_sentinel_column_stripped(self, sample_csv):
        """Test that internal lineage sentinel column is removed from result."""
        frame = ar.read_csv(sample_csv)
        
        result, lineage = ar.pipeline(frame, [
            ("strip_whitespace",),
        ], track_lineage=True)
        
        df = ar.to_pandas(result)
        # Sentinel column should not be in final result
        assert "__arnio_lineage_id__" not in df.columns
        # Original columns should still be there
        assert "name" in df.columns


class TestPipelineDryRun:
    """Test dry_run mode behavior."""

    def test_pipeline_dry_run_does_not_modify_frame(self, sample_csv):
        """Test that dry_run validates without modifying the frame."""
        frame = ar.read_csv(sample_csv)
        rows_before = frame.shape[0]
        
        # Dry run with drop_duplicates - should not modify
        result_dry = ar.pipeline(frame, [
            ("drop_duplicates",),
        ], dry_run=True)
        
        assert result_dry.shape[0] == rows_before
        
        # Regular execution should actually drop duplicates
        result_real = ar.pipeline(frame, [
            ("drop_duplicates",),
        ])
        assert result_real.shape[0] == rows_before - 1

    def test_pipeline_dry_run_with_metadata(self, sample_csv):
        """Test that metadata is still populated during dry_run."""
        frame = ar.read_csv(sample_csv)
        
        result, metadata = ar.pipeline(frame, [
            ("drop_duplicates",),
        ], dry_run=True, return_metadata=True)
        
        # All timing entries should have dry_run=True
        assert all(t["dry_run"] for t in metadata["step_timings"])
        
        # Row counts should show no actual changes
        assert metadata["row_counts"][0]["before"] == metadata["row_counts"][0]["after"]


class TestPipelineVerboseLogging:
    """Test verbose logging functionality."""

    def test_pipeline_verbose_logs_step_execution(self, sample_csv, caplog):
        """Test that verbose mode logs each step's execution."""
        frame = ar.read_csv(sample_csv)
        
        with caplog.at_level(logging.INFO):
            ar.pipeline(frame, [
                ("strip_whitespace",),
                ("drop_duplicates",),
            ], verbose=True)
        
        # Should have log messages for each step
        messages = [r.message for r in caplog.records]
        assert any("strip_whitespace" in str(m) for m in messages)
        assert any("drop_duplicates" in str(m) for m in messages)


class TestPipelineErrorHandling:
    """Test error detection and reporting."""

    def test_pipeline_unknown_step_raises_error(self, sample_csv):
        """Test that unknown step names raise UnknownStepError."""
        frame = ar.read_csv(sample_csv)
        
        with pytest.raises(ar.UnknownStepError):
            ar.pipeline(frame, [
                ("nonexistent_step_xyz",),
            ])

    def test_pipeline_invalid_step_tuple_format(self, sample_csv):
        """Test that malformed step tuples raise ValueError."""
        frame = ar.read_csv(sample_csv)
        
        with pytest.raises(ValueError, match="Invalid step format"):
            ar.pipeline(frame, [
                ("step1", "kwarg", "extra"),  # Too many elements
            ])

    def test_pipeline_non_dict_kwargs_raises_error(self, sample_csv):
        """Test that non-dict kwargs argument raises ValueError."""
        frame = ar.read_csv(sample_csv)
        
        with pytest.raises(ValueError):
            ar.pipeline(frame, [
                ("drop_nulls", "not_a_dict"),
            ])

    def test_pipeline_non_arframe_input_raises_error(self):
        """Test that non-ArFrame input raises TypeError."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        
        with pytest.raises(TypeError, match="must be an ArFrame"):
            ar.pipeline(df, [("strip_whitespace",)])

    def test_pipeline_lineage_sentinel_collision_raises_error(self, tmp_path):
        """Test that pre-existing lineage sentinel column raises ValueError."""
        csv_path = tmp_path / "collision.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["__arnio_lineage_id__", "value"])
            writer.writerow(["1", "100"])
        
        frame = ar.read_csv(csv_path)
        
        with pytest.raises(ValueError, match="__arnio_lineage_id__"):
            ar.pipeline(frame, [
                ("strip_whitespace",),
            ], track_lineage=True)


class TestPipelineCustomSteps:
    """Test custom step registration and usage in pipelines."""

    def test_pipeline_with_custom_registered_step(self, sample_csv):
        """Test pipeline execution with a custom registered step."""
        def uppercase_names(df):
            df = df.copy()
            if "name" in df.columns:
                df["name"] = df["name"].str.upper()
            return df
        
        ar.register_step("uppercase_names", uppercase_names)
        
        try:
            frame = ar.read_csv(sample_csv)
            
            clean = ar.pipeline(frame, [
                ("strip_whitespace",),
                ("uppercase_names",),
            ])
            
            df = ar.to_pandas(clean)
            assert df.loc[0, "name"] == "ALICE"
            assert df.loc[1, "name"] == "BOB"
        finally:
            ar.reset_steps()

    def test_pipeline_custom_step_in_metadata(self, sample_csv):
        """Test that custom steps appear in metadata with timings."""
        def identity_step(df):
            return df
        
        ar.register_step("identity_step", identity_step)
        
        try:
            frame = ar.read_csv(sample_csv)
            
            clean, metadata = ar.pipeline(frame, [
                ("identity_step",),
                ("strip_whitespace",),
            ], return_metadata=True)
            
            assert "identity_step" in metadata["applied_steps"]
            assert len(metadata["step_timings"]) == 2
        finally:
            ar.reset_steps()


class TestPipelineWithPandasAccessor:
    """Test pipeline execution via pandas .arnio accessor."""

    def test_pandas_accessor_clean_method(self, sample_csv):
        """Test df.arnio.clean() convenience method."""
        df = pd.read_csv(sample_csv)
        
        clean_df = df.arnio.clean(
            strip_whitespace=True,
            drop_duplicates=True,
        )
        
        assert isinstance(clean_df, pd.DataFrame)
        # Names should be stripped
        assert clean_df.loc[0, "name"] == "alice"
        # Duplicates removed
        assert len(clean_df) == 4

    def test_pandas_accessor_pipeline_method(self, sample_csv):
        """Test df.arnio.pipeline() for explicit step control."""
        df = pd.read_csv(sample_csv)
        
        clean_df = df.arnio.pipeline([
            ("strip_whitespace",),
            ("drop_nulls", {"subset": ["age"]}),
        ])
        
        assert isinstance(clean_df, pd.DataFrame)
        assert len(clean_df) == 4
        assert clean_df.loc[0, "name"] == "alice"


class TestPipelineEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_pipeline_empty_frame(self, tmp_path):
        """Test pipeline on an empty frame (header only)."""
        csv_path = tmp_path / "empty.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "name", "value"])
        
        frame = ar.read_csv(csv_path)
        
        clean = ar.pipeline(frame, [
            ("strip_whitespace",),
        ])
        
        assert clean.shape[0] == 0

    def test_pipeline_single_row(self, tmp_path):
        """Test pipeline on a single-row frame."""
        csv_path = tmp_path / "single.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "value"])
            writer.writerow(["1", "100"])
        
        frame = ar.read_csv(csv_path)
        
        clean = ar.pipeline(frame, [
            ("drop_duplicates",),
        ])
        
        assert clean.shape[0] == 1

    def test_pipeline_no_steps(self, sample_csv):
        """Test pipeline with empty step list."""
        frame = ar.read_csv(sample_csv)
        rows_before = frame.shape[0]
        
        result = ar.pipeline(frame, [])
        
        # Frame should be unchanged
        assert result.shape[0] == rows_before

    def test_pipeline_preserves_data_types(self, numeric_csv):
        """Test that pipeline preserves column data types through cleaning."""
        frame = ar.read_csv(numeric_csv)
        dtypes_before = frame.dtypes
        
        clean = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("drop_nulls", {"subset": ["flag"]}),
        ])
        
        dtypes_after = clean.dtypes
        df = ar.to_pandas(clean)
        
        # Numeric columns should remain numeric
        assert pd.api.types.is_integer_dtype(df["id"])
        assert pd.api.types.is_integer_dtype(df["value"])
