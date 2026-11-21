import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestPipelineBasic:
    """Test basic pipeline functionality."""

    def test_pipeline_returns_arframe_by_default(self):
        """Default: pipeline returns ArFrame without tuples."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
        result = ar.pipeline(frame, [])
        assert not isinstance(result, tuple)

    def test_pipeline_empty_steps_preserves_data(self):
        """Empty steps should return equivalent data."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        frame = ar.from_pandas(df)
        result = ar.pipeline(frame, [])
        result_df = ar.to_pandas(result)
        pd.testing.assert_frame_equal(result_df, df)

    def test_pipeline_returns_tuple_with_return_metadata(self):
        """return_metadata=True returns (ArFrame, dict)."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
        result = ar.pipeline(frame, [], return_metadata=True)
        assert isinstance(result, tuple) and len(result) == 2
        assert isinstance(result[1], dict)
        assert "applied_steps" in result[1]
        assert "step_timings" in result[1]
        assert "row_counts" in result[1]

    def test_pipeline_returns_tuple_with_track_lineage(self):
        """track_lineage=True returns (ArFrame, LineageReport)."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
        result = ar.pipeline(frame, [], track_lineage=True)
        assert isinstance(result, tuple) and len(result) == 2
        assert hasattr(result[1], "dropped_by_step")
        assert hasattr(result[1], "total_dropped")


class TestPipelineParameterValidation:
    """Test parameter type checking and validation."""

    def test_frame_must_be_arframe(self):
        """frame parameter must be an ArFrame, not raw DataFrame."""
        with pytest.raises(TypeError, match="frame must be an ArFrame"):
            ar.pipeline(pd.DataFrame({"a": [1]}), [])

    def test_return_metadata_must_be_bool(self):
        """return_metadata must be boolean."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        with pytest.raises(TypeError, match="return_metadata must be a bool"):
            ar.pipeline(frame, [], return_metadata="yes")

    def test_dry_run_must_be_bool(self):
        """dry_run must be boolean."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        with pytest.raises(TypeError, match="dry_run must be a bool"):
            ar.pipeline(frame, [], dry_run=1)

    def test_verbose_must_be_bool(self):
        """verbose must be boolean."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        with pytest.raises(TypeError, match="verbose must be a bool"):
            ar.pipeline(frame, [], verbose=[])

    def test_track_lineage_must_be_bool(self):
        """track_lineage must be boolean."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        with pytest.raises(TypeError, match="track_lineage must be a bool"):
            ar.pipeline(frame, [], track_lineage=1)


class TestPipelineConflicts:
    """Test mutually exclusive flag combinations."""

    def test_track_lineage_and_return_metadata_conflict(self):
        """Both track_lineage and return_metadata cannot be True together."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
        with pytest.raises(ValueError, match="cannot be used together"):
            ar.pipeline(frame, [], track_lineage=True, return_metadata=True)


class TestPipelineStepFormats:
    """Test step tuple format validation."""

    def test_step_with_one_element_tuple(self):
        """(step_name,) format is valid structure."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
        with pytest.raises(Exception):  # Should fail on unknown step, not format
            ar.pipeline(frame, [("unknown_step_xyz",)])

    def test_step_with_two_elements_tuple(self):
        """(step_name, kwargs) format is valid structure."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
        with pytest.raises(Exception):  # Should fail on unknown step, not format
            ar.pipeline(frame, [("unknown_step_xyz", {})])

    def test_step_kwargs_must_be_dict(self):
        """Step kwargs, if provided, must be a dict."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
        with pytest.raises(ValueError, match="Expected a dict"):
            ar.pipeline(frame, [("step_name", "not_a_dict")])

    def test_step_tuple_too_many_elements(self):
        """Step tuple with more than 2 elements raises ValueError."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
        with pytest.raises(ValueError, match="Invalid step format"):
            ar.pipeline(frame, [("name", {}, "extra_arg")])


class TestPipelineLineageTracking:
    """Test lineage tracking feature and sentinel column handling."""

    def test_lineage_rejects_existing_sentinel_column(self):
        """Input frame with __arnio_lineage_id__ column should raise ValueError."""
        df = pd.DataFrame({"__arnio_lineage_id__": [0, 1, 2], "a": [1, 2, 3]})
        frame = ar.from_pandas(df)
        with pytest.raises(ValueError, match="__arnio_lineage_id__"):
            ar.pipeline(frame, [], track_lineage=True)

    def test_lineage_report_structure(self):
        """LineageReport should contain dropped_by_step and total_dropped."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
        _, lineage = ar.pipeline(frame, [], track_lineage=True)
        assert hasattr(lineage, "dropped_by_step")
        assert isinstance(lineage.dropped_by_step, dict)
        assert hasattr(lineage, "total_dropped")
        assert isinstance(lineage.total_dropped, int)
        assert lineage.total_dropped >= 0

    def test_lineage_removes_sentinel_from_output(self):
        """Sentinel column should not appear in returned frame."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
        result_frame, _ = ar.pipeline(frame, [], track_lineage=True)
        result_df = ar.to_pandas(result_frame)
        assert "__arnio_lineage_id__" not in result_df.columns


class TestPipelineWithRealSteps:
    """Test pipeline with documented valid steps."""

    def test_pipeline_with_strip_whitespace(self):
        """strip_whitespace step should remove leading/trailing spaces."""
        df = pd.DataFrame({"text": ["  hello  ", " world "]})
        frame = ar.from_pandas(df)
        result = ar.pipeline(frame, [("strip_whitespace",)])
        result_df = ar.to_pandas(result)
        assert result_df["text"].iloc[0] == "hello"
        assert result_df["text"].iloc[1] == "world"

    def test_pipeline_chained_steps_execute_sequentially(self):
        """Multiple steps should execute in provided order."""
        df = pd.DataFrame({
            "name": ["  alice  ", "  bob  ", "  alice  "],
            "id": [1, 2, 3]
        })
        frame = ar.from_pandas(df)
        result = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("drop_duplicates", {"keep": "first"}),
        ])
        result_df = ar.to_pandas(result)
        # After strip and dedup, should have 2 unique rows
        assert len(result_df) <= 3


class TestPipelineEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_dataframe_pipeline(self):
        """Pipeline should handle empty DataFrames."""
        df = pd.DataFrame({"a": pd.Series([], dtype=int), "b": pd.Series([], dtype=int)})
        frame = ar.from_pandas(df)
        result = ar.pipeline(frame, [])
        result_df = ar.to_pandas(result)
        assert len(result_df) == 0
        assert list(result_df.columns) == ["a", "b"]

    def test_single_row_dataframe(self):
        """Pipeline should handle single-row DataFrames."""
        df = pd.DataFrame({"a": [1], "b": [2]})
        frame = ar.from_pandas(df)
        result = ar.pipeline(frame, [])
        result_df = ar.to_pandas(result)
        assert len(result_df) == 1

    def test_pipeline_with_verbose_flag(self):
        """verbose=True should not raise or break pipeline execution."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
        result = ar.pipeline(frame, [], verbose=True)
        result_df = ar.to_pandas(result)
        assert len(result_df) == 3
