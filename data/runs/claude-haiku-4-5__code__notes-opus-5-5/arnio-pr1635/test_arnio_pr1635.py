import pytest
import pandas as pd
import re
import logging

import arnio as ar
from arnio.exceptions import PipelineStepError, UnknownStepError


class TestPipelineTypeValidation:
    """Test parameter type validation."""
    
    def test_frame_must_be_arframe(self):
        """Non-ArFrame raises TypeError."""
        with pytest.raises(TypeError, match=re.escape("frame must be an ArFrame")):
            ar.pipeline(pd.DataFrame({"a": [1]}), [])
    
    def test_return_metadata_must_be_bool(self):
        """Non-bool return_metadata raises TypeError."""
        df = ar.from_pandas(pd.DataFrame({"a": [1]}))
        with pytest.raises(TypeError, match=re.escape("return_metadata must be a bool")):
            ar.pipeline(df, [], return_metadata=1)
    
    def test_dry_run_must_be_bool(self):
        """Non-bool dry_run raises TypeError."""
        df = ar.from_pandas(pd.DataFrame({"a": [1]}))
        with pytest.raises(TypeError, match=re.escape("dry_run must be a bool")):
            ar.pipeline(df, [], dry_run="yes")
    
    def test_verbose_must_be_bool(self):
        """Non-bool verbose raises TypeError."""
        df = ar.from_pandas(pd.DataFrame({"a": [1]}))
        with pytest.raises(TypeError, match=re.escape("verbose must be a bool")):
            ar.pipeline(df, [], verbose=None)
    
    def test_track_lineage_must_be_bool(self):
        """Non-bool track_lineage raises TypeError."""
        df = ar.from_pandas(pd.DataFrame({"a": [1]}))
        with pytest.raises(TypeError, match=re.escape("track_lineage must be a bool")):
            ar.pipeline(df, [], track_lineage=[])


class TestPipelineStepsValidation:
    """Test steps list and step format validation."""
    
    def test_steps_must_be_list(self):
        """steps must be list, not str/None/dict."""
        df = ar.from_pandas(pd.DataFrame({"a": [1]}))
        with pytest.raises(TypeError, match=re.escape("steps must be a list of step tuples")):
            ar.pipeline(df, "drop_nulls")
    
    def test_step_format_must_be_tuple(self):
        """Step must be tuple of (name,) or (name, kwargs)."""
        df = ar.from_pandas(pd.DataFrame({"a": [1]}))
        with pytest.raises(ValueError, match=re.escape("Invalid step format:")):
            ar.pipeline(df, [["drop_nulls"]])
    
    def test_step_name_must_be_string(self):
        """Step name must be string."""
        df = ar.from_pandas(pd.DataFrame({"a": [1]}))
        with pytest.raises(ValueError, match=re.escape("Invalid pipeline step name: 1")):
            ar.pipeline(df, [(1,)])
    
    def test_step_kwargs_must_be_dict(self):
        """Step kwargs must be dict."""
        df = ar.from_pandas(pd.DataFrame({"a": [1]}))
        with pytest.raises(ValueError, match=re.escape("Invalid step kwargs")):
            ar.pipeline(df, [("drop_nulls", [1])])
    
    def test_unknown_step_raises_error(self):
        """Unknown step name raises UnknownStepError."""
        df = ar.from_pandas(pd.DataFrame({"a": [1]}))
        with pytest.raises(UnknownStepError):
            ar.pipeline(df, [("unknown_step_xyz_123",)])


class TestPipelineLineageAndMetadata:
    """Test lineage tracking and metadata return."""
    
    def test_track_lineage_and_return_metadata_mutually_exclusive(self):
        """Cannot use both track_lineage and return_metadata together."""
        df = ar.from_pandas(pd.DataFrame({"a": [1]}))
        with pytest.raises(ValueError, match=re.escape("track_lineage=True and return_metadata=True cannot be used together")):
            ar.pipeline(df, [], track_lineage=True, return_metadata=True)
    
    def test_track_lineage_returns_lineage_report(self):
        """track_lineage=True returns (ArFrame, LineageReport)."""
        df = ar.from_pandas(pd.DataFrame({"a": [1, 1, 2]}))
        result, lineage = ar.pipeline(df, [("drop_duplicates",)], track_lineage=True)
        assert hasattr(lineage, 'dropped_by_step')
        assert hasattr(lineage, 'total_dropped')
        assert isinstance(lineage.dropped_by_step, dict)
    
    def test_track_lineage_rejects_sentinel_column_in_input(self):
        """lineage=True raises ValueError if input already has sentinel column."""
        df = ar.from_pandas(pd.DataFrame({"__arnio_lineage_id__": [0, 1]}))
        with pytest.raises(ValueError, match=re.escape("__arnio_lineage_id__")):
            ar.pipeline(df, [("drop_duplicates",)], track_lineage=True)
    
    def test_return_metadata_returns_dict(self):
        """return_metadata=True returns (ArFrame, dict) with required keys."""
        df = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))
        result, meta = ar.pipeline(df, [("drop_duplicates",)], return_metadata=True)
        assert isinstance(meta, dict)
        assert set(meta.keys()) >= {"applied_steps", "row_counts", "step_timings", "execution_summary"}
        assert isinstance(meta["row_counts"], list)
        assert isinstance(meta["step_timings"], list)


class TestPipelineBasicBehavior:
    """Test basic pipeline execution."""
    
    def test_empty_pipeline_returns_input_frame_unchanged(self):
        """Empty pipeline returns the input frame object itself."""
        df = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
        result = ar.pipeline(df, [])
        assert result is df
    
    def test_dry_run_returns_input_frame_data(self):
        """dry_run=True executes steps on working frame but returns input frame."""
        df = ar.from_pandas(pd.DataFrame({"a": [1, 1, 2]}))
        result = ar.pipeline(df, [("drop_duplicates",)], dry_run=True)
        # Result should match input data
        result_df = ar.to_pandas(result)
        input_df = ar.to_pandas(df)
        assert result_df.shape == input_df.shape


class TestPipelineCustomSteps:
    """Test custom step error handling."""
    
    def test_custom_step_exception_wrapped_in_pipeline_step_error(self):
        """Custom step exceptions are wrapped in PipelineStepError."""
        def failing_step(df):
            raise ValueError("test error message")
        
        ar.register_step("failing_step", failing_step)
        try:
            df = ar.from_pandas(pd.DataFrame({"a": [1]}))
            with pytest.raises(PipelineStepError):
                ar.pipeline(df, [("failing_step",)])
        finally:
            ar.unregister_step("failing_step")
    
    def test_custom_step_cannot_return_none(self):
        """Custom step returning None raises TypeError with clear message."""
        def none_step(df):
            return None
        
        ar.register_step("none_step", none_step)
        try:
            df = ar.from_pandas(pd.DataFrame({"a": [1]}))
            with pytest.raises(TypeError, match=re.escape("returned None")):
                ar.pipeline(df, [("none_step",)])
        finally:
            ar.unregister_step("none_step")
    
    def test_custom_step_must_return_dataframe(self):
        """Custom step returning non-DataFrame raises TypeError."""
        def list_step(df):
            return [1, 2, 3]
        
        ar.register_step("list_step", list_step)
        try:
            df = ar.from_pandas(pd.DataFrame({"a": [1]}))
            with pytest.raises(TypeError, match=re.escape("instead of a pandas DataFrame")):
                ar.pipeline(df, [("list_step",)])
        finally:
            ar.unregister_step("list_step")
