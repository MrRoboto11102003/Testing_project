import re
import logging
import pytest
import pandas as pd
import numpy as np
import arnio as ar
from arnio.exceptions import UnknownStepError, PipelineStepError


class TestPipelineInputValidation:
    """Test input parameter validation."""

    def test_frame_must_be_arframe(self):
        """Input frame must be an ArFrame, not pandas DataFrame."""
        with pytest.raises(TypeError, match="frame must be an ArFrame"):
            ar.pipeline(pd.DataFrame({"a": [1]}), [])

    def test_bool_parameters_must_be_bool(self):
        """Boolean parameters must be actual bools, not truthy values."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
        
        with pytest.raises(TypeError, match=r"return_metadata must be a bool"):
            ar.pipeline(frame, [], return_metadata=1)
        
        with pytest.raises(TypeError, match=r"dry_run must be a bool"):
            ar.pipeline(frame, [], dry_run="yes")
        
        with pytest.raises(TypeError, match=r"verbose must be a bool"):
            ar.pipeline(frame, [], verbose=None)
        
        with pytest.raises(TypeError, match=r"track_lineage must be a bool"):
            ar.pipeline(frame, [], track_lineage=[])

    def test_track_lineage_and_return_metadata_mutually_exclusive(self):
        """Cannot use track_lineage and return_metadata together."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        with pytest.raises(ValueError, match="cannot be used together"):
            ar.pipeline(frame, [], track_lineage=True, return_metadata=True)

    def test_steps_must_be_list(self):
        """steps parameter must be a list."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        
        with pytest.raises(TypeError, match="steps must be a list"):
            ar.pipeline(frame, None)
        
        with pytest.raises(TypeError, match="steps must be a list"):
            ar.pipeline(frame, "drop_nulls")
        
        with pytest.raises(TypeError, match="bare step tuple"):
            ar.pipeline(frame, ("drop_nulls",))

    def test_invalid_step_format(self):
        """Each step must be (name,) or (name, kwargs_dict)."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        
        with pytest.raises(ValueError, match="Invalid step format"):
            ar.pipeline(frame, [["drop_nulls"]])
        
        with pytest.raises(ValueError, match="Invalid pipeline step name"):
            ar.pipeline(frame, [(123,)])
        
        with pytest.raises(ValueError, match="Invalid step kwargs"):
            ar.pipeline(frame, [("drop_nulls", "not_a_dict")])

    def test_unknown_step_raises(self):
        """Unknown step name raises UnknownStepError before execution."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
        with pytest.raises(UnknownStepError):
            ar.pipeline(frame, [("nonexistent_step",)])


class TestPipelineExecution:
    """Test pipeline execution and basic functionality."""

    def test_empty_steps_returns_input_frame(self):
        """Empty steps list returns input frame object itself."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
        result = ar.pipeline(frame, [])
        assert result is frame

    def test_single_step_pipeline(self):
        """Single cleaning step executes correctly."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 1, 2]}))
        result = ar.pipeline(frame, [("drop_duplicates",)])
        df = ar.to_pandas(result)
        assert df.shape[0] == 2

    def test_sequential_steps(self):
        """Multiple steps execute in sequence."""
        frame = ar.from_pandas(pd.DataFrame({
            "a": [1, 1, 2],
            "b": [" x ", " x ", " y "]
        }))
        result = ar.pipeline(frame, [
            ("drop_duplicates",),
            ("strip_whitespace",),
        ])
        df = ar.to_pandas(result)
        assert df.shape[0] == 2
        assert df["b"].iloc[0] == "x"  # whitespace stripped

    def test_rename_columns_shorthand_mapping(self):
        """rename_columns accepts shorthand dict as mapping."""
        frame = ar.from_pandas(pd.DataFrame({"old_name": [1, 2, 3]}))
        result = ar.pipeline(frame, [
            ("rename_columns", {"old_name": "new_name"})
        ])
        df = ar.to_pandas(result)
        assert "new_name" in df.columns
        assert "old_name" not in df.columns

    def test_cast_types_shorthand_mapping(self):
        """cast_types accepts shorthand dict as mapping."""
        frame = ar.from_pandas(pd.DataFrame({"id": ["1", "2", "3"]}))
        result = ar.pipeline(frame, [
            ("cast_types", {"id": "int64"})
        ])
        df = ar.to_pandas(result)
        assert df["id"].dtype == np.int64


class TestPipelineMetadata:
    """Test return_metadata functionality."""

    def test_return_metadata_structure(self):
        """return_metadata=True returns (frame, dict) with metadata."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 1, 2]}))
        result, metadata = ar.pipeline(frame, [
            ("drop_duplicates",),
        ], return_metadata=True)
        
        assert isinstance(result, ar.ArFrame)
        assert isinstance(metadata, dict)
        assert "applied_steps" in metadata
        assert "row_counts" in metadata
        assert "step_timings" in metadata
        assert "execution_summary" in metadata

    def test_metadata_applied_steps_and_row_counts(self):
        """Metadata tracks applied steps and row count changes."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 1, 2]}))
        _, metadata = ar.pipeline(frame, [
            ("drop_duplicates",),
        ], return_metadata=True)
        
        assert metadata["applied_steps"] == ["drop_duplicates"]
        assert metadata["row_counts"][0]["before"] == 3
        assert metadata["row_counts"][0]["after"] == 2
        assert metadata["row_counts"][0]["step"] == "drop_duplicates"

    def test_metadata_step_timings_and_summary(self):
        """Metadata includes timing and execution summary."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
        _, metadata = ar.pipeline(frame, [
            ("strip_whitespace",),
        ], return_metadata=True)
        
        assert len(metadata["step_timings"]) == 1
        assert isinstance(metadata["step_timings"][0]["seconds"], (int, float))
        assert metadata["step_timings"][0]["seconds"] >= 0
        
        summary = metadata["execution_summary"]
        assert "total_runtime_ms" in summary
        assert isinstance(summary["total_runtime_ms"], (int, float))


class TestPipelineLineage:
    """Test track_lineage functionality."""

    def test_track_lineage_returns_report(self):
        """track_lineage=True returns (frame, LineageReport) tuple."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 1, 2]}))
        result, lineage = ar.pipeline(frame, [
            ("drop_duplicates",),
        ], track_lineage=True)
        
        assert isinstance(result, ar.ArFrame)
        assert hasattr(lineage, "dropped_by_step")
        assert hasattr(lineage, "total_dropped")

    def test_lineage_tracks_dropped_indices(self):
        """Lineage correctly tracks which row indices were dropped."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 1, 2]}))
        _, lineage = ar.pipeline(frame, [
            ("drop_duplicates",),
        ], track_lineage=True)
        
        assert lineage.dropped_by_step["drop_duplicates"] == [1]
        assert lineage.total_dropped == 1

    def test_lineage_sentinel_column_collision(self):
        """Input frame with sentinel column name raises ValueError."""
        df = pd.DataFrame({"__arnio_lineage_id__": [1, 2, 3]})
        frame = ar.from_pandas(df)
        with pytest.raises(ValueError, match="__arnio_lineage_id__"):
            ar.pipeline(frame, [("drop_duplicates",)], track_lineage=True)


class TestPipelineSpecialCases:
    """Test special modes and error handling."""

    def test_dry_run_returns_input_unchanged(self):
        """dry_run=True executes steps but returns input frame."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 1, 2]}))
        result = ar.pipeline(frame, [
            ("drop_duplicates",),
        ], dry_run=True)
        
        assert result is frame  # input object unchanged

    def test_custom_step_exception_wrapping(self):
        """Custom step exceptions wrapped in PipelineStepError."""
        ar.reset_steps()
        
        def failing_step(df):
            raise RuntimeError("step failed")
        
        ar.register_step("failing_step", failing_step)
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
        
        with pytest.raises(PipelineStepError) as exc_info:
            ar.pipeline(frame, [("failing_step",)])
        
        assert "failing_step" in str(exc_info.value)
        ar.reset_steps()

    def test_verbose_logging(self, caplog):
        """verbose=True logs step execution details."""
        caplog.set_level(logging.INFO, logger="arnio")
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
        ar.pipeline(frame, [("strip_whitespace",)], verbose=True)
        
        assert "strip_whitespace" in caplog.text
        assert "rows:" in caplog.text
