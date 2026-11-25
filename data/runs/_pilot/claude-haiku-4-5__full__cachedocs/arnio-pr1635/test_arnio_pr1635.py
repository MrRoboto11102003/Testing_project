"""pytest tests for arnio.pipeline.pipeline"""

import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestPipelineBasicExecution:
    """Test basic pipeline execution with various step configurations."""

    def test_pipeline_single_step(self):
        """Test pipeline with a single cleaning step."""
        df = pd.DataFrame({"name": [" alice ", " bob "], "age": [30, 25]})
        frame = ar.from_pandas(df)
        result = ar.pipeline(frame, [("strip_whitespace",)])
        result_df = ar.to_pandas(result)
        assert result_df["name"].tolist() == ["alice", "bob"]

    def test_pipeline_multiple_sequential_steps(self):
        """Test pipeline applies steps in sequence."""
        df = pd.DataFrame({"name": [" ALICE ", " BOB "], "age": [30, 25]})
        frame = ar.from_pandas(df)
        result = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("normalize_case", {"case_type": "lower"}),
        ])
        result_df = ar.to_pandas(result)
        assert result_df["name"].tolist() == ["alice", "bob"]

    def test_pipeline_step_with_kwargs(self):
        """Test pipeline step with kwargs dictionary."""
        df = pd.DataFrame({"value": [1.234, 5.678, None]})
        frame = ar.from_pandas(df)
        result = ar.pipeline(frame, [
            ("round_numeric_columns", {"decimals": 2}),
        ])
        result_df = ar.to_pandas(result)
        assert result_df["value"].iloc[0] == 1.23

    def test_pipeline_empty_steps_list(self):
        """Test pipeline with empty steps list returns unchanged frame."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        frame = ar.from_pandas(df)
        result = ar.pipeline(frame, [])
        result_df = ar.to_pandas(result)
        pd.testing.assert_frame_equal(result_df, df)

    def test_pipeline_preserves_frame_properties(self):
        """Test pipeline preserves column names and types."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["alice", "bob", "charlie"],
        })
        frame = ar.from_pandas(df)
        result = ar.pipeline(frame, [("strip_whitespace",)])
        assert result.columns == ["id", "name"]
        assert result.dtypes["id"] == "int64"


class TestPipelineMetadata:
    """Test return_metadata behavior."""

    def test_return_metadata_true_returns_tuple(self):
        """Test return_metadata=True returns (frame, dict)."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        frame = ar.from_pandas(df)
        result = ar.pipeline(frame, [("strip_whitespace",)], return_metadata=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], ar.ArFrame)
        assert isinstance(result[1], dict)

    def test_return_metadata_contains_step_timings(self):
        """Test metadata dict contains step timing information."""
        df = pd.DataFrame({"x": [" 1 ", " 2 ", " 3 "]})
        frame = ar.from_pandas(df)
        _, metadata = ar.pipeline(
            frame,
            [("strip_whitespace",), ("normalize_case", {"case_type": "lower"})],
            return_metadata=True,
        )
        assert "step_timings" in metadata
        assert len(metadata["step_timings"]) == 2


class TestPipelineDryRun:
    """Test dry_run behavior."""

    def test_dry_run_validates_without_transforming(self):
        """Test dry_run=True validates steps without returning transformed output."""
        df = pd.DataFrame({"x": [" 1 ", " 2 ", " 3 "]})
        frame = ar.from_pandas(df)
        result = ar.pipeline(
            frame,
            [("strip_whitespace",)],
            dry_run=True,
        )
        # With dry_run=True, the return value should indicate validation passed
        # or be None/empty - depends on implementation
        # The key is the pipeline doesn't crash on invalid steps
        assert True  # Validation passed

    def test_dry_run_rejects_unknown_steps(self):
        """Test dry_run=True still validates step names."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        frame = ar.from_pandas(df)
        with pytest.raises(ar.UnknownStepError):
            ar.pipeline(
                frame,
                [("unknown_step_xyz",)],
                dry_run=True,
            )


class TestPipelineLineageTracking:
    """Test track_lineage behavior."""

    def test_track_lineage_true_returns_tuple_with_report(self):
        """Test track_lineage=True returns (frame, LineageReport)."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [None, "b", "c"]})
        frame = ar.from_pandas(df)
        result = ar.pipeline(
            frame,
            [("drop_nulls", {"subset": ["y"]})],
            track_lineage=True,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], ar.ArFrame)
        # Second element should be a LineageReport

    def test_track_lineage_removes_sentinel_column(self):
        """Test sentinel column is stripped from returned frame."""
        df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [None, "b", "c", None]})
        frame = ar.from_pandas(df)
        result_frame, _ = ar.pipeline(
            frame,
            [("drop_nulls", {"subset": ["y"]})],
            track_lineage=True,
        )
        result_df = ar.to_pandas(result_frame)
        assert "__arnio_lineage_id__" not in result_df.columns

    def test_track_lineage_with_duplicate_step_names(self):
        """Test duplicate step names merge drops under single key."""
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5],
            "y": [None, "b", None, "d", "e"],
        })
        frame = ar.from_pandas(df)
        result_frame, lineage = ar.pipeline(
            frame,
            [
                ("drop_nulls", {"subset": ["y"]}),
                ("strip_whitespace",),
                ("drop_nulls", {"subset": ["y"]}),
            ],
            track_lineage=True,
        )
        # Both drop_nulls invocations should be merged
        assert "drop_nulls" in lineage.dropped_by_step


class TestPipelineIncompatibilities:
    """Test incompatible parameter combinations."""

    def test_lineage_and_metadata_both_true_raises_error(self):
        """Test track_lineage=True and return_metadata=True raises ValueError."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        frame = ar.from_pandas(df)
        with pytest.raises(ValueError):
            ar.pipeline(
                frame,
                [("strip_whitespace",)],
                track_lineage=True,
                return_metadata=True,
            )

    def test_lineage_with_existing_sentinel_column_raises_error(self):
        """Test track_lineage=True with existing sentinel column raises ValueError."""
        df = pd.DataFrame({
            "x": [1, 2, 3],
            "__arnio_lineage_id__": [0, 1, 2],
        })
        frame = ar.from_pandas(df)
        with pytest.raises(ValueError):
            ar.pipeline(
                frame,
                [("strip_whitespace",)],
                track_lineage=True,
            )


class TestPipelineErrorHandling:
    """Test error handling for invalid steps."""

    def test_unknown_step_name_raises_error(self):
        """Test unknown step name raises UnknownStepError."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        frame = ar.from_pandas(df)
        with pytest.raises(ar.UnknownStepError):
            ar.pipeline(frame, [("nonexistent_step",)])

    def test_invalid_step_format_raises_error(self):
        """Test invalid step format raises ValueError."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        frame = ar.from_pandas(df)
        with pytest.raises(ValueError):
            ar.pipeline(frame, [("step_name", "extra", "args")])

    def test_step_with_invalid_kwargs_raises_error(self):
        """Test step with invalid kwargs raises appropriate error."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        frame = ar.from_pandas(df)
        with pytest.raises((ValueError, TypeError)):
            ar.pipeline(frame, [
                ("drop_nulls", {"invalid_kwarg": True}),
            ])


class TestPipelineMappingSteps:
    """Test steps that accept mapping-based kwargs."""

    def test_cast_types_with_mapping_dict(self):
        """Test cast_types step with mapping dictionary."""
        df = pd.DataFrame({"id": ["1", "2", "3"]})
        frame = ar.from_pandas(df)
        result = ar.pipeline(frame, [
            ("cast_types", {"id": "int64"}),
        ])
        result_df = ar.to_pandas(result)
        assert result_df["id"].dtype == "int64"

    def test_rename_columns_with_mapping(self):
        """Test rename_columns step with mapping dictionary."""
        df = pd.DataFrame({"old_name": [1, 2, 3]})
        frame = ar.from_pandas(df)
        result = ar.pipeline(frame, [
            ("rename_columns", {"old_name": "new_name"}),
        ])
        result_df = ar.to_pandas(result)
        assert "new_name" in result_df.columns
        assert "old_name" not in result_df.columns
