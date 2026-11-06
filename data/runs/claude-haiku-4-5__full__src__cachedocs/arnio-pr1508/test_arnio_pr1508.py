import pytest
import pandas as pd
import numpy as np
import arnio as ar
from arnio.exceptions import UnknownStepError


class TestPipelineBasic:
    """Test basic pipeline execution"""

    def test_pipeline_single_step(self):
        """Test pipeline with a single cleaning step"""
        df = pd.DataFrame({"name": [" Alice ", " Bob "]})
        frame = ar.from_pandas(df)

        result = ar.pipeline(frame, [("strip_whitespace",)])

        result_df = ar.to_pandas(result)
        assert result_df["name"].tolist() == ["Alice", "Bob"]

    def test_pipeline_multiple_steps(self):
        """Test pipeline executes multiple steps sequentially"""
        df = pd.DataFrame({
            "name": [" ALICE ", " BOB "],
            "value": [1.0, np.nan]
        })
        frame = ar.from_pandas(df)

        result = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("normalize_case", {"case_type": "lower"}),
            ("fill_nulls", {"value": 0.0}),
        ])

        result_df = ar.to_pandas(result)
        assert result_df["name"].tolist() == ["alice", "bob"]
        assert result_df["value"].tolist() == [1.0, 0.0]

    def test_pipeline_empty_steps(self):
        """Test pipeline with empty steps list returns unchanged frame"""
        df = pd.DataFrame({"x": [1, 2, 3]})
        frame = ar.from_pandas(df)

        result = ar.pipeline(frame, [])

        result_df = ar.to_pandas(result)
        pd.testing.assert_frame_equal(result_df, df)


class TestPipelineReturnTypes:
    """Test return value types based on parameters"""

    def test_pipeline_default_returns_arframe(self):
        """Test default return type is ArFrame"""
        frame = ar.from_pandas(pd.DataFrame({"x": [1, 2]}))

        result = ar.pipeline(frame, [("strip_whitespace",)])

        assert isinstance(result, ar.ArFrame)

    def test_pipeline_return_metadata_returns_tuple(self):
        """Test return_metadata=True returns (ArFrame, dict) tuple"""
        frame = ar.from_pandas(pd.DataFrame({"x": [1, 2]}))

        result, metadata = ar.pipeline(
            frame,
            [("drop_nulls",)],
            return_metadata=True
        )

        assert isinstance(result, ar.ArFrame)
        assert isinstance(metadata, dict)
        assert "applied_steps" in metadata
        assert "step_timings" in metadata
        assert "row_counts" in metadata
        assert "execution_summary" in metadata

    def test_pipeline_track_lineage_returns_lineage_report(self):
        """Test track_lineage=True returns (ArFrame, LineageReport) tuple"""
        df = pd.DataFrame({"x": [1, 2, None, 4]})
        frame = ar.from_pandas(df)

        result, lineage = ar.pipeline(
            frame,
            [("drop_nulls",)],
            track_lineage=True
        )

        assert isinstance(result, ar.ArFrame)
        assert hasattr(lineage, "dropped_by_step")
        assert hasattr(lineage, "total_dropped")


class TestPipelineValidation:
    """Test parameter validation and error handling"""

    def test_pipeline_invalid_frame_type(self):
        """Test TypeError when frame is not ArFrame"""
        with pytest.raises(TypeError, match="frame must be an ArFrame"):
            ar.pipeline(pd.DataFrame({"x": [1]}), [])

    def test_pipeline_conflicting_return_and_lineage(self):
        """Test ValueError when both return_metadata and track_lineage are True"""
        frame = ar.from_pandas(pd.DataFrame({"x": [1]}))

        with pytest.raises(ValueError, match="cannot be used together"):
            ar.pipeline(
                frame,
                [],
                return_metadata=True,
                track_lineage=True
            )

    def test_pipeline_invalid_step_format_too_many_elements(self):
        """Test ValueError for step tuple with too many elements"""
        frame = ar.from_pandas(pd.DataFrame({"x": [1]}))

        with pytest.raises(ValueError, match="Invalid step format"):
            ar.pipeline(frame, [("drop_nulls", {}, "extra")])

    def test_pipeline_invalid_step_kwargs_type(self):
        """Test ValueError when step kwargs is not a dict"""
        frame = ar.from_pandas(pd.DataFrame({"x": [1]}))

        with pytest.raises(ValueError, match="Expected a dict"):
            ar.pipeline(frame, [("drop_nulls", "invalid")])

    def test_pipeline_unknown_step_name(self):
        """Test UnknownStepError for unregistered step name"""
        frame = ar.from_pandas(pd.DataFrame({"x": [1]}))

        with pytest.raises(UnknownStepError):
            ar.pipeline(frame, [("nonexistent_step_xyz",)])

    def test_pipeline_lineage_sentinel_column_collision(self):
        """Test ValueError when input frame has __arnio_lineage_id__ column"""
        df = pd.DataFrame({
            "x": [1, 2],
            "__arnio_lineage_id__": [0, 1]
        })
        frame = ar.from_pandas(df)

        with pytest.raises(ValueError, match="__arnio_lineage_id__"):
            ar.pipeline(frame, [("drop_nulls",)], track_lineage=True)


class TestPipelineSpecialModes:
    """Test dry_run and track_lineage modes"""

    def test_pipeline_dry_run_prevents_modification(self):
        """Test dry_run=True validates without modifying data"""
        df = pd.DataFrame({"x": [1, 2, None]})
        frame = ar.from_pandas(df)

        result = ar.pipeline(
            frame,
            [("drop_nulls",)],
            dry_run=True
        )

        result_df = ar.to_pandas(result)
        pd.testing.assert_frame_equal(result_df, df)

    def test_pipeline_lineage_tracks_dropped_rows(self):
        """Test lineage tracking records dropped row indices by step"""
        df = pd.DataFrame({
            "x": [1, 2, None, 4, None, 6],
            "y": ["a", "b", "c", "d", "e", "f"]
        })
        frame = ar.from_pandas(df)

        _, lineage = ar.pipeline(
            frame,
            [("drop_nulls", {"subset": ["x"]})],
            track_lineage=True
        )

        assert "drop_nulls" in lineage.dropped_by_step
        assert lineage.total_dropped == len(lineage.dropped_by_step["drop_nulls"])
        assert lineage.total_dropped > 0

    def test_pipeline_lineage_sentinel_not_in_output(self):
        """Test lineage tracking sentinel column removed from returned frame"""
        frame = ar.from_pandas(pd.DataFrame({"x": [1, 2, 3]}))

        result, _ = ar.pipeline(
            frame,
            [("strip_whitespace",)],
            track_lineage=True
        )

        result_df = ar.to_pandas(result)
        assert "__arnio_lineage_id__" not in result_df.columns

    def test_pipeline_lineage_accumulates_drops_same_step_name(self):
        """Test lineage merges drops when step name appears multiple times"""
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5],
            "y": [10, 20, np.nan, 40, 50]
        })
        frame = ar.from_pandas(df)

        _, lineage = ar.pipeline(
            frame,
            [
                ("drop_nulls", {"subset": ["y"]}),
                ("strip_whitespace",),
                ("drop_nulls", {"subset": ["y"]}),  # Same step name again
            ],
            track_lineage=True
        )

        # Both drop_nulls invocations should be merged under single key
        assert "drop_nulls" in lineage.dropped_by_step
        assert lineage.dropped_by_step["drop_nulls"] == sorted(
            lineage.dropped_by_step["drop_nulls"]
        )
