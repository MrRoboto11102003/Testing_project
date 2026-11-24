import pytest
import pandas as pd
import logging
import re
from arnio.exceptions import UnknownStepError, PipelineStepError
import arnio as ar


class TestPipelineTypeValidation:
    """Test type validation for parameters."""
    
    def test_frame_must_be_arframe(self):
        with pytest.raises(TypeError, match="frame must be an ArFrame"):
            ar.pipeline(pd.DataFrame({"a": [1]}), [])
    
    def test_return_metadata_must_be_bool(self):
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        with pytest.raises(TypeError, match="return_metadata must be a bool"):
            ar.pipeline(frame, [], return_metadata=1)
    
    def test_dry_run_must_be_bool(self):
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        with pytest.raises(TypeError, match="dry_run must be a bool"):
            ar.pipeline(frame, [], dry_run="yes")
    
    def test_track_lineage_must_be_bool(self):
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        with pytest.raises(TypeError, match="track_lineage must be a bool"):
            ar.pipeline(frame, [], track_lineage=0)


class TestPipelineStepsValidation:
    """Test steps list and format validation."""
    
    def test_steps_must_be_list_not_none(self):
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        with pytest.raises(TypeError, match="steps must be a list"):
            ar.pipeline(frame, None)
    
    def test_steps_cannot_be_bare_tuple(self):
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        with pytest.raises(TypeError, match="Use"):
            ar.pipeline(frame, ("drop_duplicates",))
    
    def test_step_name_must_be_string(self):
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        with pytest.raises(ValueError, match="Invalid pipeline step name"):
            ar.pipeline(frame, [(1,)])
    
    def test_step_kwargs_must_be_dict(self):
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        with pytest.raises(ValueError, match="Invalid step kwargs"):
            ar.pipeline(frame, [("drop_duplicates", [1, 2])])


class TestPipelineConflictingFlags:
    """Test mutually exclusive options."""
    
    def test_track_lineage_and_return_metadata_conflict(self):
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        with pytest.raises(ValueError, match="track_lineage=True and return_metadata=True"):
            ar.pipeline(frame, [], track_lineage=True, return_metadata=True)


class TestPipelineBasicExecution:
    """Test basic pipeline execution."""
    
    def test_empty_pipeline_returns_input_object(self):
        """Empty steps returns input frame object itself."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
        result = ar.pipeline(frame, [])
        assert result is frame
    
    def test_single_step_execution(self):
        """Single step transforms frame correctly."""
        df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
        frame = ar.from_pandas(df)
        result = ar.pipeline(frame, [("drop_duplicates",)])
        result_df = ar.to_pandas(result)
        assert result_df.shape[0] == 2
    
    def test_unknown_step_raises_error(self):
        """Unregistered step name raises UnknownStepError."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        with pytest.raises(UnknownStepError):
            ar.pipeline(frame, [("unknown_xyz",)])


class TestPipelineReturnMetadata:
    """Test return_metadata feature."""
    
    def test_return_metadata_structure(self):
        """return_metadata returns tuple with metadata dict."""
        df = pd.DataFrame({"a": [1, 1, 2]})
        frame = ar.from_pandas(df)
        result, meta = ar.pipeline(
            frame,
            [("drop_duplicates",)],
            return_metadata=True
        )
        
        assert isinstance(meta, dict)
        assert "applied_steps" in meta
        assert "row_counts" in meta
        assert "step_timings" in meta
        assert meta["applied_steps"] == ["drop_duplicates"]
        assert len(meta["row_counts"]) == 1
        assert meta["row_counts"][0]["before"] == 3
        assert meta["row_counts"][0]["after"] == 2


class TestPipelineTrackLineage:
    """Test track_lineage feature."""
    
    def test_track_lineage_returns_report(self):
        """track_lineage returns tuple with LineageReport."""
        df = pd.DataFrame({"a": [1, 1, 2]})
        frame = ar.from_pandas(df)
        result, lineage = ar.pipeline(
            frame,
            [("drop_duplicates",)],
            track_lineage=True
        )
        
        assert hasattr(lineage, "dropped_by_step")
        assert hasattr(lineage, "total_dropped")
        assert "drop_duplicates" in lineage.dropped_by_step
        assert lineage.dropped_by_step["drop_duplicates"] == [1]
        assert lineage.total_dropped == 1
    
    def test_track_lineage_no_sentinel_in_output(self):
        """Sentinel column is stripped from output frame."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        frame = ar.from_pandas(df)
        result, lineage = ar.pipeline(frame, [], track_lineage=True)
        result_df = ar.to_pandas(result)
        
        assert "__arnio_lineage_id__" not in result_df.columns
    
    def test_track_lineage_rejects_existing_sentinel_column(self):
        """track_lineage rejects input with __arnio_lineage_id__ column."""
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "__arnio_lineage_id__": [0, 1, 2]
        })
        frame = ar.from_pandas(df)
        
        with pytest.raises(ValueError, match="__arnio_lineage_id__"):
            ar.pipeline(frame, [], track_lineage=True)


class TestPipelineDryRun:
    """Test dry_run feature."""
    
    def test_dry_run_returns_input_object(self):
        """dry_run returns input frame object itself."""
        df = pd.DataFrame({"a": [1, 1, 2]})
        frame = ar.from_pandas(df)
        result = ar.pipeline(frame, [("drop_duplicates",)], dry_run=True)
        
        assert result is frame
    
    def test_dry_run_with_return_metadata_no_changes(self):
        """dry_run with return_metadata shows no rows affected."""
        df = pd.DataFrame({"a": [1, 1, 2]})
        frame = ar.from_pandas(df)
        result, meta = ar.pipeline(
            frame,
            [("drop_duplicates",)],
            dry_run=True,
            return_metadata=True
        )
        
        row_count = meta["row_counts"][0]
        assert row_count["before"] == row_count["after"]
        assert row_count["dry_run"] is True


class TestPipelineVerbose:
    """Test verbose logging."""
    
    def test_verbose_logs_step_execution(self, caplog):
        """verbose logs each step execution."""
        df = pd.DataFrame({"a": [1, 1, 2]})
        frame = ar.from_pandas(df)
        
        caplog.set_level(logging.INFO, logger="arnio")
        ar.pipeline(frame, [("drop_duplicates",)], verbose=True)
        
        assert "drop_duplicates" in caplog.text
        assert "rows:" in caplog.text


class TestPipelineCustomSteps:
    """Test custom step execution and error handling."""
    
    def test_custom_step_execution(self):
        """Custom registered step executes correctly."""
        try:
            def custom_step(df):
                return df.copy()
            
            ar.register_step("test_custom_xyz", custom_step)
            df = pd.DataFrame({"a": [1, 2, 3]})
            frame = ar.from_pandas(df)
            result = ar.pipeline(frame, [("test_custom_xyz",)])
            
            assert result is not None
        finally:
            ar.unregister_step("test_custom_xyz")
    
    def test_custom_step_error_wrapped_in_pipelinestep_error(self):
        """Custom step exceptions wrapped in PipelineStepError."""
        try:
            def failing_step(df):
                raise ValueError("Inner error")
            
            ar.register_step("failing_xyz", failing_step)
            df = pd.DataFrame({"a": [1]})
            frame = ar.from_pandas(df)
            
            with pytest.raises(PipelineStepError):
                ar.pipeline(frame, [("failing_xyz",)])
        finally:
            ar.unregister_step("failing_xyz")


class TestPipelineSpecialCases:
    """Test special behaviors and edge cases."""
    
    def test_rename_columns_shorthand_mapping(self):
        """rename_columns accepts shorthand dict mapping."""
        df = pd.DataFrame({"old": [1, 2, 3]})
        frame = ar.from_pandas(df)
        result = ar.pipeline(frame, [("rename_columns", {"old": "new"})])
        result_df = ar.to_pandas(result)
        
        assert "new" in result_df.columns
        assert "old" not in result_df.columns
    
    def test_multiple_steps_chain_in_sequence(self):
        """Multiple steps execute in order and chain correctly."""
        df = pd.DataFrame({
            "a": [1, 1, 2],
            "b": ["x", "x", "y"],
            "c": [10, 10, 20]
        })
        frame = ar.from_pandas(df)
        result = ar.pipeline(frame, [
            ("drop_duplicates",),
            ("drop_columns", {"subset": ["c"]})
        ])
        result_df = ar.to_pandas(result)
        assert result_df.shape[0] == 2
        assert "c" not in result_df.columns
        assert "a" in result_df.columns
