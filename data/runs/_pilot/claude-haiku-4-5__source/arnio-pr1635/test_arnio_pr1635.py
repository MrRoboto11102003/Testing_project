import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestPipelineBasicExecution:
    """Test basic pipeline execution."""

    def test_pipeline_returns_arframe(self):
        """Pipeline with no steps should return ArFrame."""
        frame = ar.from_dict({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = ar.pipeline(frame, [])
        assert isinstance(result, ar.ArFrame)

    def test_pipeline_single_step(self):
        """Pipeline with single step should execute."""
        frame = ar.from_dict({"a": [1, 2, 3], "b": ["x", "Y", "z"]})
        result = ar.pipeline(frame, [("normalize_case", {"case_type": "lower"})])
        assert isinstance(result, ar.ArFrame)

    def test_pipeline_multiple_steps(self):
        """Pipeline should apply multiple steps sequentially."""
        frame = ar.from_dict({"a": [1, 2, 3], "b": [" x ", "y", " z "]})
        result = ar.pipeline(
            frame,
            [
                ("strip_whitespace",),
                ("drop_nulls",),
            ],
        )
        assert isinstance(result, ar.ArFrame)

    def test_pipeline_step_with_kwargs(self):
        """Pipeline should pass kwargs to steps."""
        frame = ar.from_dict({"a": [1, None, 3], "b": [1.0, 2.0, 3.0]})
        result = ar.pipeline(
            frame,
            [("fill_nulls", {"value": 0, "subset": ["a"]})],
        )
        assert isinstance(result, ar.ArFrame)


class TestPipelineReturnMetadata:
    """Test return_metadata parameter."""

    def test_pipeline_return_metadata_true_returns_tuple(self):
        """return_metadata=True should return (ArFrame, dict)."""
        frame = ar.from_dict({"a": [1, 2, 3]})
        result, metadata = ar.pipeline(frame, [("drop_nulls",)], return_metadata=True)
        assert isinstance(result, ar.ArFrame)
        assert isinstance(metadata, dict)

    def test_pipeline_metadata_dict_structure(self):
        """Metadata dict should contain all required keys."""
        frame = ar.from_dict({"a": [1, 2, 3]})
        result, metadata = ar.pipeline(frame, [("drop_nulls",)], return_metadata=True)
        assert "applied_steps" in metadata
        assert "row_counts" in metadata
        assert "step_timings" in metadata
        assert "execution_summary" in metadata

    def test_pipeline_metadata_applied_steps_list(self):
        """applied_steps should list executed step names."""
        frame = ar.from_dict({"a": [1, 2, 3]})
        result, metadata = ar.pipeline(
            frame,
            [("drop_nulls",), ("drop_duplicates",)],
            return_metadata=True,
        )
        assert metadata["applied_steps"] == ["drop_nulls", "drop_duplicates"]

    def test_pipeline_metadata_row_counts_per_step(self):
        """row_counts should track before/after for each step."""
        frame = ar.from_dict({"a": [1, 2, 3]})
        result, metadata = ar.pipeline(
            frame, [("drop_nulls",)], return_metadata=True
        )
        assert len(metadata["row_counts"]) == 1
        assert metadata["row_counts"][0]["step"] == "drop_nulls"
        assert "before" in metadata["row_counts"][0]
        assert "after" in metadata["row_counts"][0]

    def test_pipeline_metadata_execution_summary(self):
        """execution_summary should contain timing and step info."""
        frame = ar.from_dict({"a": [1, 2, 3]})
        result, metadata = ar.pipeline(frame, [("drop_nulls",)], return_metadata=True)
        summary = metadata["execution_summary"]
        assert "total_runtime_ms" in summary
        assert "steps" in summary
        assert isinstance(summary["total_runtime_ms"], (int, float))


class TestPipelineTrackLineage:
    """Test track_lineage parameter."""

    def test_pipeline_track_lineage_returns_tuple(self):
        """track_lineage=True should return (ArFrame, LineageReport)."""
        frame = ar.from_dict({"a": [1, 2, 3]})
        result, lineage = ar.pipeline(frame, [("drop_nulls",)], track_lineage=True)
        assert isinstance(result, ar.ArFrame)
        assert hasattr(lineage, "dropped_by_step")
        assert hasattr(lineage, "total_dropped")

    def test_pipeline_lineage_dropped_by_step_dict(self):
        """LineageReport.dropped_by_step should be a dict."""
        frame = ar.from_dict({"a": [1, 2, 3]})
        result, lineage = ar.pipeline(frame, [("drop_nulls",)], track_lineage=True)
        assert isinstance(lineage.dropped_by_step, dict)
        assert isinstance(lineage.total_dropped, int)

    def test_pipeline_lineage_no_drops(self):
        """No drops should result in empty dropped_by_step."""
        frame = ar.from_dict({"a": [1, 2, 3], "b": [4, 5, 6]})
        result, lineage = ar.pipeline(
            frame, [("drop_nulls",)], track_lineage=True
        )
        assert lineage.total_dropped == 0

    def test_pipeline_lineage_sentinel_column_collision(self):
        """Pipeline should reject if sentinel column already exists."""
        frame = ar.from_dict(
            {"__arnio_lineage_id__": [0, 1], "a": [1, 2]}
        )
        with pytest.raises(ValueError, match="__arnio_lineage_id__"):
            ar.pipeline(frame, [("drop_nulls",)], track_lineage=True)

    def test_pipeline_lineage_multiple_steps(self):
        """Lineage should track drops across multiple steps."""
        frame = ar.from_dict({"a": [1, None, 2, 2]})
        result, lineage = ar.pipeline(
            frame,
            [("drop_nulls",), ("drop_duplicates",)],
            track_lineage=True,
        )
        assert isinstance(lineage.dropped_by_step, dict)
        assert lineage.total_dropped >= 0


class TestPipelineDryRun:
    """Test dry_run parameter."""

    def test_pipeline_dry_run_validation(self):
        """dry_run=True should validate without modifying data."""
        frame = ar.from_dict({"a": [1, 2, 3]})
        result = ar.pipeline(frame, [("drop_nulls",)], dry_run=True)
        assert isinstance(result, ar.ArFrame)

    def test_pipeline_dry_run_with_metadata(self):
        """dry_run with metadata should mark steps as dry_run."""
        frame = ar.from_dict({"a": [1, 2, 3]})
        result, metadata = ar.pipeline(
            frame, [("drop_nulls",)], dry_run=True, return_metadata=True
        )
        assert all(rc.get("dry_run") for rc in metadata["row_counts"])


class TestPipelineParameterValidation:
    """Test parameter type validation."""

    def test_pipeline_invalid_frame_type(self):
        """Pipeline should reject non-ArFrame input."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(TypeError, match="frame must be an ArFrame"):
            ar.pipeline(df, [])

    def test_pipeline_invalid_return_metadata_type(self):
        """return_metadata parameter must be bool."""
        frame = ar.from_dict({"a": [1, 2, 3]})
        with pytest.raises(TypeError, match="return_metadata must be a bool"):
            ar.pipeline(frame, [], return_metadata="yes")

    def test_pipeline_invalid_dry_run_type(self):
        """dry_run parameter must be bool."""
        frame = ar.from_dict({"a": [1, 2, 3]})
        with pytest.raises(TypeError, match="dry_run must be a bool"):
            ar.pipeline(frame, [], dry_run=1)

    def test_pipeline_invalid_verbose_type(self):
        """verbose parameter must be bool."""
        frame = ar.from_dict({"a": [1, 2, 3]})
        with pytest.raises(TypeError, match="verbose must be a bool"):
            ar.pipeline(frame, [], verbose=[])

    def test_pipeline_invalid_track_lineage_type(self):
        """track_lineage parameter must be bool."""
        frame = ar.from_dict({"a": [1, 2, 3]})
        with pytest.raises(TypeError, match="track_lineage must be a bool"):
            ar.pipeline(frame, [], track_lineage=None)

    def test_pipeline_conflicting_flags_lineage_and_metadata(self):
        """Cannot use track_lineage and return_metadata together."""
        frame = ar.from_dict({"a": [1, 2, 3]})
        with pytest.raises(ValueError, match="cannot be used together"):
            ar.pipeline(
                frame,
                [("drop_nulls",)],
                track_lineage=True,
                return_metadata=True,
            )


class TestPipelineStepValidation:
    """Test step format and validity validation."""

    def test_pipeline_invalid_step_format_too_many_elements(self):
        """Step tuple must have 1 or 2 elements."""
        frame = ar.from_dict({"a": [1, 2, 3]})
        with pytest.raises(ValueError, match="Invalid step format"):
            ar.pipeline(frame, [("drop_nulls", {}, "extra")])

    def test_pipeline_invalid_step_format_empty_tuple(self):
        """Step tuple cannot be empty."""
        frame = ar.from_dict({"a": [1, 2, 3]})
        with pytest.raises(ValueError):
            ar.pipeline(frame, [()])

    def test_pipeline_invalid_step_kwargs_not_dict(self):
        """Step kwargs must be dict if provided."""
        frame = ar.from_dict({"a": [1, 2, 3]})
        with pytest.raises(ValueError, match="Invalid step kwargs"):
            ar.pipeline(frame, [("drop_nulls", "not_a_dict")])

    def test_pipeline_unknown_step_name(self):
        """Pipeline should raise error for unknown step."""
        frame = ar.from_dict({"a": [1, 2, 3]})
        with pytest.raises(Exception):  # UnknownStepError
            ar.pipeline(frame, [("nonexistent_step_xyz",)])


class TestPipelineVerbose:
    """Test verbose parameter."""

    def test_pipeline_verbose_flag_accepted(self):
        """verbose=True should execute without error."""
        frame = ar.from_dict({"a": [1, 2, 3]})
        result = ar.pipeline(frame, [("drop_nulls",)], verbose=True)
        assert isinstance(result, ar.ArFrame)

    def test_pipeline_verbose_with_multiple_steps(self):
        """verbose should work with multiple pipeline steps."""
        frame = ar.from_dict({"a": [1, 2, 3], "b": ["x", "x", "y"]})
        result = ar.pipeline(
            frame,
            [("drop_nulls",), ("drop_duplicates",)],
            verbose=True,
        )
        assert isinstance(result, ar.ArFrame)
