import pytest
import logging
import re
import pandas as pd
import numpy as np
import arnio as ar
from arnio.exceptions import UnknownStepError, PipelineStepError


class TestPipelineBasic:
    """Test basic pipeline functionality."""
    
    def test_pipeline_empty_steps_returns_input_frame(self):
        """With empty steps, pipeline returns the input frame unchanged."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
        result = ar.pipeline(frame, [])
        assert result is frame  # Same object
    
    def test_pipeline_single_step_execution(self):
        """Single step in pipeline executes correctly."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 1, 2]}))
        result = ar.pipeline(frame, [("drop_duplicates",)])
        df = ar.to_pandas(result)
        assert len(df) == 2
    
    def test_pipeline_multiple_steps_chain(self):
        """Multiple steps execute in order and chain correctly."""
        frame = ar.from_pandas(pd.DataFrame({
            "a": [1, 2, 3],
            "b": [" x ", " y ", " z "]
        }))
        result = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("drop_columns", {"columns": ["a"]}),
        ])
        df = ar.to_pandas(result)
        assert list(df.columns) == ["b"]
        assert df["b"].tolist() == ["x", "y", "z"]


class TestTypeValidation:
    """Test input type validation."""
    
    def test_pipeline_rejects_non_arframe(self):
        """Pipeline rejects non-ArFrame input."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(TypeError, match="frame must be an ArFrame"):
            ar.pipeline(df, [])
    
    def test_pipeline_rejects_non_bool_flags(self):
        """All flag parameters must be bool."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        
        with pytest.raises(TypeError, match="return_metadata must be a bool"):
            ar.pipeline(frame, [], return_metadata=1)
        
        with pytest.raises(TypeError, match="dry_run must be a bool"):
            ar.pipeline(frame, [], dry_run="yes")
        
        with pytest.raises(TypeError, match="verbose must be a bool"):
            ar.pipeline(frame, [], verbose=None)
        
        with pytest.raises(TypeError, match="track_lineage must be a bool"):
            ar.pipeline(frame, [], track_lineage=[])


class TestStepValidation:
    """Test step validation before execution."""
    
    def test_pipeline_rejects_steps_not_list(self):
        """steps must be a list, not a bare tuple or other type."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        
        with pytest.raises(TypeError, match="steps must be a list of step tuples"):
            ar.pipeline(frame, None)
        
        with pytest.raises(TypeError, match="steps must be a list of step tuples"):
            ar.pipeline(frame, ("drop_nulls",))
    
    def test_pipeline_validates_all_steps_before_execution(self):
        """All steps validated before any step executes; first error wins."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 1, 2]}))
        # First step valid, second invalid
        with pytest.raises(ValueError, match="Invalid step format"):
            ar.pipeline(frame, [
                ("drop_duplicates",),
                ["bad_format"],  # Detected before drop_duplicates runs
            ])
        # Frame untouched
        df = ar.to_pandas(frame)
        assert len(df) == 3
    
    def test_pipeline_rejects_unknown_step_name(self):
        """Unknown step name raises UnknownStepError with available steps listed."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        with pytest.raises(UnknownStepError):
            ar.pipeline(frame, [("unknown_step_xyz",)])
    
    def test_pipeline_rejects_invalid_step_format(self):
        """Step must be (name,) or (name, kwargs); kwargs must be dict."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        
        with pytest.raises(ValueError, match="Invalid step format"):
            ar.pipeline(frame, [{"drop_nulls": {}}])
        
        with pytest.raises(ValueError, match="Invalid pipeline step name"):
            ar.pipeline(frame, [(1,)])
        
        with pytest.raises(ValueError, match="Invalid step kwargs"):
            ar.pipeline(frame, [("drop_nulls", [1, 2])])


class TestReturnMetadata:
    """Test return_metadata behavior."""
    
    def test_pipeline_return_metadata_structure(self):
        """return_metadata returns (frame, dict) with timing and row info."""
        frame = ar.from_pandas(pd.DataFrame({
            "a": [1, 1, 2],
            "b": ["x", "y", "z"]
        }))
        result, metadata = ar.pipeline(frame, [
            ("drop_duplicates",),
        ], return_metadata=True)
        
        assert isinstance(result, ar.ArFrame)
        assert isinstance(metadata, dict)
        assert set(metadata.keys()) == {
            "applied_steps", "row_counts", "step_timings", "execution_summary"
        }
        
        assert metadata["applied_steps"] == ["drop_duplicates"]
        assert len(metadata["row_counts"]) == 1
        
        row_info = metadata["row_counts"][0]
        assert row_info["step"] == "drop_duplicates"
        assert row_info["before"] == 3
        assert row_info["after"] == 2
        assert row_info["dry_run"] is False
        
        timing_info = metadata["step_timings"][0]
        assert timing_info["step"] == "drop_duplicates"
        assert isinstance(timing_info["seconds"], float)
        assert timing_info["seconds"] >= 0


class TestDryRun:
    """Test dry_run behavior."""
    
    def test_pipeline_dry_run_returns_input_unchanged(self):
        """With dry_run=True, returned frame is input frame; no rows dropped."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 1, 2]}))
        result = ar.pipeline(frame, [
            ("drop_duplicates",)
        ], dry_run=True)
        
        assert result is frame
        df = ar.to_pandas(result)
        assert len(df) == 3
    
    def test_pipeline_dry_run_with_metadata_shows_flag(self):
        """dry_run metadata has dry_run=True; before==after."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 1, 2]}))
        result, metadata = ar.pipeline(frame, [
            ("drop_duplicates",)
        ], dry_run=True, return_metadata=True)
        
        assert metadata["row_counts"][0]["dry_run"] is True
        assert metadata["row_counts"][0]["before"] == 3
        assert metadata["row_counts"][0]["after"] == 3


class TestTrackLineage:
    """Test track_lineage row-drop tracking."""
    
    def test_pipeline_track_lineage_basic(self):
        """track_lineage returns (frame, LineageReport) tracking drops."""
        frame = ar.from_pandas(pd.DataFrame({
            "a": [1, 1, 2],
            "b": ["x", "y", "z"]
        }))
        result, lineage = ar.pipeline(frame, [
            ("drop_duplicates",)
        ], track_lineage=True)
        
        assert isinstance(result, ar.ArFrame)
        assert hasattr(lineage, "dropped_by_step")
        assert hasattr(lineage, "total_dropped")
        
        # Row index 1 is duplicate of row 0, so it's dropped
        assert lineage.dropped_by_step["drop_duplicates"] == [1]
        assert lineage.total_dropped == 1
    
    def test_pipeline_track_lineage_no_sentinel_in_output(self):
        """Sentinel column is stripped from output frame."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))
        result, _ = ar.pipeline(frame, [
            ("drop_nulls",)
        ], track_lineage=True)
        
        df = ar.to_pandas(result)
        assert "__arnio_lineage_id__" not in df.columns
    
    def test_pipeline_track_lineage_rejects_existing_sentinel(self):
        """Input with sentinel column is rejected."""
        df = pd.DataFrame({
            "__arnio_lineage_id__": [0, 1],
            "a": [1, 2]
        })
        frame = ar.from_pandas(df)
        
        with pytest.raises(ValueError, match="__arnio_lineage_id__"):
            ar.pipeline(frame, [], track_lineage=True)
    
    def test_pipeline_track_lineage_mutually_exclusive_with_metadata(self):
        """track_lineage and return_metadata cannot both be True."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        
        with pytest.raises(ValueError, match="cannot be used together"):
            ar.pipeline(frame, [], track_lineage=True, return_metadata=True)


class TestCustomSteps:
    """Test custom pipeline steps."""
    
    def test_pipeline_custom_step_execution(self):
        """Custom registered steps execute in pipeline."""
        def uppercase_col(df, column="a"):
            df = df.copy()
            if column in df.columns and df[column].dtype == "object":
                df[column] = df[column].str.upper()
            return df
        
        ar.register_step("uppercase_col", uppercase_col)
        try:
            frame = ar.from_pandas(pd.DataFrame({"a": ["x", "y", "z"]}))
            result = ar.pipeline(frame, [
                ("uppercase_col", {"column": "a"})
            ])
            df = ar.to_pandas(result)
            assert df["a"].tolist() == ["X", "Y", "Z"]
        finally:
            ar.unregister_step("uppercase_col")
    
    def test_pipeline_custom_step_error_wrapped_in_pipelinestep_error(self):
        """Exceptions in custom steps wrapped in PipelineStepError."""
        def bad_step(df):
            raise ValueError("custom step error")
        
        ar.register_step("bad_step", bad_step)
        try:
            frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
            with pytest.raises(PipelineStepError) as exc_info:
                ar.pipeline(frame, [("bad_step",)])
            
            assert "bad_step" in str(exc_info.value)
            assert isinstance(exc_info.value.__cause__, ValueError)
        finally:
            ar.unregister_step("bad_step")
    
    def test_pipeline_custom_step_return_validation(self):
        """Custom step must return pandas DataFrame."""
        def bad_return_none(df):
            return None
        
        def bad_return_list(df):
            return [1, 2, 3]
        
        ar.register_step("bad_none", bad_return_none)
        ar.register_step("bad_list", bad_return_list)
        try:
            frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
            
            with pytest.raises(TypeError, match="returned None"):
                ar.pipeline(frame, [("bad_none",)])
            
            with pytest.raises(TypeError, match="returned 'list'"):
                ar.pipeline(frame, [("bad_list",)])
        finally:
            ar.unregister_step("bad_none")
            ar.unregister_step("bad_list")


class TestVerboseLogging:
    """Test verbose logging output."""
    
    def test_pipeline_verbose_logs_step_execution(self, caplog):
        """verbose=True logs step execution with timing and row changes."""
        frame = ar.from_pandas(pd.DataFrame({
            "a": [1, 1, 2],
            "b": ["x", "y", "z"]
        }))
        
        with caplog.at_level(logging.INFO, logger="arnio"):
            ar.pipeline(frame, [
                ("drop_duplicates",),
                ("strip_whitespace",),
            ], verbose=True)
        
        # Check log contains step names and row info
        assert "drop_duplicates" in caplog.text
        assert "strip_whitespace" in caplog.text
        assert "rows:" in caplog.text
        assert "ms" in caplog.text or "ms" in caplog.text.lower()
