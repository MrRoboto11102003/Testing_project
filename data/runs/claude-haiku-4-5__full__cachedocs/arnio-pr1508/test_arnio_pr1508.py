import pytest
import pandas as pd
import numpy as np
import arnio as ar


@pytest.fixture
def sample_frame():
    """Create a sample ArFrame for testing."""
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": [" alice ", " bob ", " charlie ", " bob ", " alice "],
        "age": [25.0, 30.0, np.nan, 35.0, 25.0],
        "score": [85.5, 90.0, 88.5, np.nan, 85.5],
    })
    return ar.from_pandas(df)


@pytest.fixture
def empty_frame():
    """Create an empty ArFrame."""
    df = pd.DataFrame({
        "col1": pd.array([], dtype="object"),
        "col2": pd.array([], dtype="object")
    })
    return ar.from_pandas(df)


class TestPipelineBasic:
    """Test basic pipeline execution."""

    def test_single_step(self, sample_frame):
        """Pipeline applies a single step correctly."""
        result = ar.pipeline(sample_frame, [("strip_whitespace",)])
        assert isinstance(result, ar.ArFrame)
        df = ar.to_pandas(result)
        assert df["name"].iloc[0] == "alice"

    def test_multiple_steps_sequential(self, sample_frame):
        """Pipeline applies multiple steps in correct sequence."""
        result = ar.pipeline(sample_frame, [
            ("strip_whitespace",),
            ("drop_duplicates",),
        ])
        assert isinstance(result, ar.ArFrame)
        df = ar.to_pandas(result)
        original_len = len(ar.to_pandas(sample_frame))
        assert len(df) <= original_len

    def test_empty_steps_list(self, sample_frame):
        """Pipeline with empty steps list returns unchanged frame."""
        original_df = ar.to_pandas(sample_frame)
        result = ar.pipeline(sample_frame, [])
        result_df = ar.to_pandas(result)
        pd.testing.assert_frame_equal(result_df, original_df)

    def test_step_with_kwargs(self, sample_frame):
        """Pipeline steps correctly receive kwargs dictionary."""
        result = ar.pipeline(sample_frame, [
            ("fill_nulls", {"value": 0.0, "subset": ["age"]}),
        ])
        df = ar.to_pandas(result)
        assert not df["age"].isna().any()


class TestPipelineMetadata:
    """Test return_metadata parameter."""

    def test_return_metadata_tuple_structure(self, sample_frame):
        """return_metadata=True returns (ArFrame, dict) tuple."""
        result = ar.pipeline(
            sample_frame,
            [("strip_whitespace",)],
            return_metadata=True
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        frame, metadata = result
        assert isinstance(frame, ar.ArFrame)
        assert isinstance(metadata, dict)

    def test_metadata_step_timings(self, sample_frame):
        """Metadata includes step_timings list."""
        result = ar.pipeline(
            sample_frame,
            [("strip_whitespace",), ("drop_duplicates",)],
            return_metadata=True
        )
        frame, metadata = result
        assert "step_timings" in metadata
        assert len(metadata["step_timings"]) == 2

    def test_metadata_applied_steps(self, sample_frame):
        """Metadata tracks applied_steps in order."""
        result = ar.pipeline(
            sample_frame,
            [("strip_whitespace",), ("drop_duplicates",)],
            return_metadata=True
        )
        frame, metadata = result
        assert "applied_steps" in metadata
        assert metadata["applied_steps"][0] == "strip_whitespace"
        assert metadata["applied_steps"][1] == "drop_duplicates"

    def test_metadata_row_counts(self, sample_frame):
        """Metadata includes row_counts tracking."""
        result = ar.pipeline(
            sample_frame,
            [("drop_nulls",)],
            return_metadata=True
        )
        frame, metadata = result
        assert "row_counts" in metadata
        assert len(metadata["row_counts"]) > 0


class TestPipelineLineage:
    """Test track_lineage parameter."""

    def test_track_lineage_returns_report(self, sample_frame):
        """track_lineage=True returns (ArFrame, LineageReport) tuple."""
        result = ar.pipeline(
            sample_frame,
            [("drop_nulls",)],
            track_lineage=True
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        frame, lineage = result
        assert isinstance(frame, ar.ArFrame)
        assert hasattr(lineage, "dropped_by_step")

    def test_track_lineage_dropped_rows(self, sample_frame):
        """LineageReport tracks which rows were dropped by each step."""
        result = ar.pipeline(
            sample_frame,
            [("drop_nulls",)],
            track_lineage=True
        )
        frame, lineage = result
        assert isinstance(lineage.dropped_by_step, dict)
        assert "drop_nulls" in lineage.dropped_by_step

    def test_lineage_sentinel_column_conflict(self):
        """ValueError raised if frame already has __arnio_lineage_id__ column."""
        df = pd.DataFrame({
            "__arnio_lineage_id__": [1, 2, 3],
            "value": ["a", "b", "c"],
        })
        frame = ar.from_pandas(df)
        with pytest.raises(ValueError):
            ar.pipeline(frame, [("strip_whitespace",)], track_lineage=True)

    def test_lineage_and_metadata_mutually_exclusive(self, sample_frame):
        """ValueError raised when both track_lineage and return_metadata are True."""
        with pytest.raises(ValueError):
            ar.pipeline(
                sample_frame,
                [("drop_nulls",)],
                track_lineage=True,
                return_metadata=True
            )


class TestPipelineErrorHandling:
    """Test error handling."""

    def test_unknown_step_raises_error(self, sample_frame):
        """UnknownStepError raised for unregistered step names."""
        with pytest.raises(ar.UnknownStepError):
            ar.pipeline(sample_frame, [("unknown_step_xyz",)])

    def test_invalid_frame_type_raises_error(self):
        """TypeError raised if frame is not ArFrame."""
        with pytest.raises(TypeError):
            ar.pipeline(pd.DataFrame({"a": [1, 2]}), [("strip_whitespace",)])

    def test_invalid_steps_type_raises_error(self, sample_frame):
        """TypeError raised if steps is not a list."""
        with pytest.raises(TypeError):
            ar.pipeline(sample_frame, "not_a_list")

    def test_missing_column_in_step_raises_error(self, sample_frame):
        """Error raised when step references non-existent column."""
        with pytest.raises(Exception):  # ValueError or KeyError
            ar.pipeline(sample_frame, [
                ("drop_nulls", {"subset": ["nonexistent_col"]}),
            ])


class TestPipelineOptions:
    """Test optional parameters."""

    def test_dry_run_validates_without_error(self, sample_frame):
        """dry_run=True validates pipeline without executing."""
        result = ar.pipeline(
            sample_frame,
            [("strip_whitespace",)],
            dry_run=True
        )
        assert isinstance(result, ar.ArFrame)

    def test_dry_run_validates_unknown_steps(self, sample_frame):
        """dry_run=True still validates step names."""
        with pytest.raises(ar.UnknownStepError):
            ar.pipeline(
                sample_frame,
                [("unknown_step",)],
                dry_run=True
            )

    def test_verbose_flag_accepted(self, sample_frame):
        """verbose=True executes without error."""
        result = ar.pipeline(
            sample_frame,
            [("strip_whitespace",)],
            verbose=True
        )
        assert isinstance(result, ar.ArFrame)


class TestPipelineEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_frame_through_pipeline(self, empty_frame):
        """Pipeline works correctly on empty frames."""
        result = ar.pipeline(empty_frame, [("strip_whitespace",)])
        assert isinstance(result, ar.ArFrame)
        assert len(ar.to_pandas(result)) == 0

    def test_repeated_steps_in_pipeline(self, sample_frame):
        """Same step can appear multiple times in pipeline."""
        result = ar.pipeline(sample_frame, [
            ("drop_nulls", {"subset": ["age"]}),
            ("drop_nulls", {"subset": ["score"]}),
        ])
        assert isinstance(result, ar.ArFrame)
        df = ar.to_pandas(result)
        assert not df["age"].isna().any()
        assert not df["score"].isna().any()

    def test_always_returns_arframe(self, sample_frame):
        """Pipeline returns ArFrame for all basic operations."""
        result = ar.pipeline(sample_frame, [("drop_duplicates",)])
        assert isinstance(result, ar.ArFrame)
        assert hasattr(result, "columns")
        assert hasattr(result, "shape")
