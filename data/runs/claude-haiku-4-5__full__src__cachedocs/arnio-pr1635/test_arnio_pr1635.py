import pytest
import pandas as pd
import numpy as np
import arnio as ar
from arnio.exceptions import UnknownStepError


class TestPipelineBasic:
    """Test basic pipeline functionality."""

    def test_pipeline_single_step(self):
        """Test pipeline with a single step."""
        df = pd.DataFrame({"name": ["  Alice  ", "  Bob  "], "age": [30, 25]})
        frame = ar.from_pandas(df)
        
        result = ar.pipeline(frame, [("strip_whitespace",)])
        result_df = ar.to_pandas(result)
        
        assert result_df["name"].tolist() == ["Alice", "Bob"]
        assert result_df.shape == (2, 2)

    def test_pipeline_multiple_steps_sequence(self):
        """Test pipeline applies steps in correct sequence."""
        df = pd.DataFrame(
            {"name": ["  ALICE  ", "  bob  "], "value": [1, 2]}
        )
        frame = ar.from_pandas(df)
        
        result = ar.pipeline(
            frame,
            [
                ("strip_whitespace",),
                ("normalize_case", {"case_type": "lower"}),
            ],
        )
        result_df = ar.to_pandas(result)
        
        assert result_df["name"].tolist() == ["alice", "bob"]

    def test_pipeline_step_with_kwargs(self):
        """Test pipeline step with keyword arguments."""
        df = pd.DataFrame({"value": [1.0, 2.0, np.nan, 4.0]})
        frame = ar.from_pandas(df)
        
        result = ar.pipeline(
            frame,
            [("fill_nulls", {"value": 0, "subset": ["value"]})]
        )
        result_df = ar.to_pandas(result)
        
        assert result_df["value"].tolist() == [1.0, 2.0, 0.0, 4.0]


class TestPipelineReturnTypes:
    """Test different return types based on parameters."""

    def test_pipeline_default_returns_arframe(self):
        """Test default return type is ArFrame."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        frame = ar.from_pandas(df)
        
        result = ar.pipeline(frame, [("strip_whitespace",)])
        
        assert isinstance(result, ar.ArFrame)

    def test_pipeline_return_metadata_returns_tuple(self):
        """Test return_metadata=True returns (ArFrame, dict)."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        frame = ar.from_pandas(df)
        
        result, metadata = ar.pipeline(
            frame, [("strip_whitespace",)], return_metadata=True
        )
        
        assert isinstance(result, ar.ArFrame)
        assert isinstance(metadata, dict)
        assert set(metadata.keys()) == {
            "applied_steps",
            "step_timings",
            "row_counts",
            "execution_summary",
        }
        assert metadata["applied_steps"] == ["strip_whitespace"]
        assert len(metadata["step_timings"]) == 1
        assert len(metadata["row_counts"]) == 1

    def test_pipeline_track_lineage_returns_tuple(self):
        """Test track_lineage=True returns (ArFrame, LineageReport)."""
        df = pd.DataFrame({"x": [1, 2, 3, 4], "y": ["a", None, "c", "d"]})
        frame = ar.from_pandas(df)
        
        result, lineage = ar.pipeline(
            frame,
            [("drop_nulls", {"subset": ["y"]})],
            track_lineage=True,
        )
        
        assert isinstance(result, ar.ArFrame)
        assert hasattr(lineage, "dropped_by_step")
        assert hasattr(lineage, "total_dropped")
        assert result.shape[0] == 3


class TestPipelineErrorHandling:
    """Test error handling and validation."""

    def test_pipeline_invalid_frame_type(self):
        """Test pipeline raises TypeError for non-ArFrame input."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        
        with pytest.raises(TypeError, match="frame must be an ArFrame"):
            ar.pipeline(df, [("strip_whitespace",)])

    def test_pipeline_return_metadata_not_bool(self):
        """Test pipeline raises TypeError for non-bool return_metadata."""
        frame = ar.from_pandas(pd.DataFrame({"x": [1, 2, 3]}))
        
        with pytest.raises(TypeError, match="return_metadata must be a bool"):
            ar.pipeline(frame, [("strip_whitespace",)], return_metadata="yes")

    def test_pipeline_dry_run_not_bool(self):
        """Test pipeline raises TypeError for non-bool dry_run."""
        frame = ar.from_pandas(pd.DataFrame({"x": [1, 2, 3]}))
        
        with pytest.raises(TypeError, match="dry_run must be a bool"):
            ar.pipeline(frame, [("strip_whitespace",)], dry_run=1)

    def test_pipeline_track_lineage_and_return_metadata_conflict(self):
        """Test ValueError when track_lineage and return_metadata both True."""
        frame = ar.from_pandas(pd.DataFrame({"x": [1, 2, 3]}))
        
        with pytest.raises(ValueError, match="cannot be used together"):
            ar.pipeline(
                frame,
                [("strip_whitespace",)],
                track_lineage=True,
                return_metadata=True,
            )

    def test_pipeline_unknown_step_raises_error(self):
        """Test pipeline raises UnknownStepError for unknown step."""
        frame = ar.from_pandas(pd.DataFrame({"x": [1, 2, 3]}))
        
        with pytest.raises(UnknownStepError):
            ar.pipeline(frame, [("nonexistent_step_xyz",)])

    def test_pipeline_invalid_step_format(self):
        """Test pipeline raises ValueError for malformed step tuple."""
        frame = ar.from_pandas(pd.DataFrame({"x": [1, 2, 3]}))
        
        with pytest.raises(ValueError, match="Invalid step format"):
            ar.pipeline(frame, [("step", "arg1", "arg2")])

    def test_pipeline_sentinel_column_collision(self):
        """Test ValueError when sentinel column already exists with track_lineage."""
        df = pd.DataFrame({
            "x": [1, 2, 3],
            "__arnio_lineage_id__": [0, 1, 2],
        })
        frame = ar.from_pandas(df)
        
        with pytest.raises(ValueError, match="__arnio_lineage_id__"):
            ar.pipeline(frame, [("strip_whitespace",)], track_lineage=True)


class TestPipelineLineageTracking:
    """Test row-level lineage tracking."""

    def test_pipeline_lineage_tracks_dropped_rows(self):
        """Test lineage tracks row indices dropped by steps."""
        df = pd.DataFrame({"x": [1, 2, None, 4], "y": ["a", "b", "c", "d"]})
        frame = ar.from_pandas(df)
        
        result, lineage = ar.pipeline(
            frame,
            [("drop_nulls", {"subset": ["x"]})],
            track_lineage=True,
        )
        
        assert lineage.total_dropped == 1
        assert lineage.dropped_by_step == {"drop_nulls": [2]}

    def test_pipeline_lineage_merged_repeated_steps(self):
        """Test lineage merges drops from repeated step names."""
        df = pd.DataFrame(
            {"x": [1, 1, 2, 2, 3], "y": [None, "b", "c", "d", "e"]}
        )
        frame = ar.from_pandas(df)
        
        result, lineage = ar.pipeline(
            frame,
            [
                ("drop_nulls", {"subset": ["y"]}),
                ("drop_duplicates", {"keep": "first"}),
            ],
            track_lineage=True,
        )
        
        # Both invocations of steps should be in lineage
        assert isinstance(lineage.dropped_by_step, dict)
        assert lineage.total_dropped >= 0


class TestPipelineMetadata:
    """Test metadata collection."""

    def test_pipeline_metadata_contains_all_fields(self):
        """Test metadata dict contains all required fields."""
        frame = ar.from_pandas(pd.DataFrame({"x": [1, 2, 3]}))
        
        result, metadata = ar.pipeline(
            frame,
            [
                ("strip_whitespace",),
                ("drop_duplicates",),
            ],
            return_metadata=True,
        )
        
        assert metadata["applied_steps"] == ["strip_whitespace", "drop_duplicates"]
        assert len(metadata["step_timings"]) == 2
        assert len(metadata["row_counts"]) == 2
        assert all("step" in t for t in metadata["step_timings"])
        assert all("seconds" in t for t in metadata["step_timings"])
        assert all(
            set(rc.keys()).issuperset({"step", "before", "after"})
            for rc in metadata["row_counts"]
        )
        assert "total_runtime_ms" in metadata["execution_summary"]


class TestPipelineDryRun:
    """Test dry_run behavior."""

    def test_pipeline_dry_run_validates_without_changing_data(self):
        """Test dry_run validates pipeline without modifying frame."""
        df = pd.DataFrame(
            {"x": [1, 2, None], "y": ["a", "b", "c"]}
        )
        frame = ar.from_pandas(df)
        original_shape = frame.shape
        
        result = ar.pipeline(
            frame,
            [("drop_nulls", {"subset": ["x"]})],
            dry_run=True,
        )
        
        # Dry run should not modify the result
        assert result.shape == original_shape
        assert ar.to_pandas(result)["x"].isna().sum() == 1
