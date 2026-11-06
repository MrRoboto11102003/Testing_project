import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestPipelineBasic:
    """Test basic pipeline execution."""

    @pytest.fixture
    def sample_frame(self):
        """Create a sample ArFrame for testing."""
        df = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
            "city": ["NYC", "LA", "Chicago"]
        })
        return ar.from_pandas(df)

    def test_pipeline_returns_arframe_by_default(self, sample_frame):
        """Pipeline returns ArFrame by default."""
        result = ar.pipeline(sample_frame, [("strip_whitespace",)])
        assert isinstance(result, ar.ArFrame)
        assert not isinstance(result, tuple)

    def test_pipeline_single_step(self, sample_frame):
        """Pipeline executes a single step correctly."""
        result = ar.pipeline(sample_frame, [("strip_whitespace",)])
        df = ar.to_pandas(result)
        assert df.shape == (3, 3)

    def test_pipeline_multiple_steps(self, sample_frame):
        """Pipeline executes multiple steps in sequence."""
        result = ar.pipeline(sample_frame, [
            ("strip_whitespace",),
            ("normalize_case", {"case_type": "lower"}),
        ])
        df = ar.to_pandas(result)
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == 3

    def test_pipeline_with_step_kwargs(self, sample_frame):
        """Pipeline passes kwargs to steps correctly."""
        result = ar.pipeline(sample_frame, [
            ("normalize_case", {"case_type": "upper"}),
        ])
        assert isinstance(result, ar.ArFrame)


class TestPipelineMetadata:
    """Test pipeline metadata collection and return."""

    @pytest.fixture
    def sample_frame(self):
        df = pd.DataFrame({
            "id": [1, 2, 2, 3],
            "name": ["Alice", "Bob", "Bob", "Charlie"]
        })
        return ar.from_pandas(df)

    def test_return_metadata_returns_tuple(self, sample_frame):
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

    def test_metadata_contains_required_keys(self, sample_frame):
        """Metadata dict contains required keys."""
        result, metadata = ar.pipeline(
            sample_frame,
            [("strip_whitespace",), ("drop_duplicates",)],
            return_metadata=True
        )
        assert "step_timings" in metadata
        assert "applied_steps" in metadata
        assert "row_counts" in metadata
        assert "execution_summary" in metadata

    def test_metadata_applied_steps_list(self, sample_frame):
        """Metadata applied_steps records each executed step."""
        result, metadata = ar.pipeline(
            sample_frame,
            [("strip_whitespace",), ("drop_duplicates",)],
            return_metadata=True
        )
        applied = metadata["applied_steps"]
        assert len(applied) >= 2

    def test_metadata_row_counts_tracking(self, sample_frame):
        """Metadata tracks row count changes through pipeline."""
        result, metadata = ar.pipeline(
            sample_frame,
            [("drop_duplicates",)],
            return_metadata=True
        )
        row_counts = metadata["row_counts"]
        assert row_counts is not None


class TestPipelineValidation:
    """Test pipeline validation and error handling."""

    @pytest.fixture
    def sample_frame(self):
        df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
        return ar.from_pandas(df)

    def test_unknown_step_raises_error(self, sample_frame):
        """Unknown step name raises UnknownStepError."""
        with pytest.raises(ar.UnknownStepError):
            ar.pipeline(sample_frame, [("nonexistent_step",)])

    def test_invalid_step_raises_error_before_execution(self, sample_frame):
        """Invalid steps are caught before any execution."""
        with pytest.raises((ar.UnknownStepError, ValueError, TypeError)):
            ar.pipeline(sample_frame, [
                ("strip_whitespace",),
                ("invalid_step_name",),
            ])

    def test_malformed_step_tuple_raises_error(self, sample_frame):
        """Malformed step tuples raise error during validation."""
        with pytest.raises((ValueError, TypeError)):
            ar.pipeline(sample_frame, [
                ("strip_whitespace", {}, "extra", "args"),
            ])

    def test_invalid_step_kwargs_type_raises_error(self, sample_frame):
        """Non-dict kwargs in step tuple raises error."""
        with pytest.raises((ValueError, TypeError)):
            ar.pipeline(sample_frame, [
                ("strip_whitespace", "not_a_dict"),
            ])


class TestPipelineDryRun:
    """Test dry_run mode validation."""

    @pytest.fixture
    def sample_frame(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        return ar.from_pandas(df)

    def test_dry_run_validates_without_returning_data(self, sample_frame):
        """dry_run=True validates pipeline without processing data."""
        result = ar.pipeline(
            sample_frame,
            [("strip_whitespace",)],
            dry_run=True
        )
        # dry_run should not throw but may return None or empty result
        assert result is None or isinstance(result, ar.ArFrame)

    def test_dry_run_catches_invalid_steps(self, sample_frame):
        """dry_run=True still validates step names."""
        with pytest.raises(ar.UnknownStepError):
            ar.pipeline(
                sample_frame,
                [("invalid_step",)],
                dry_run=True
            )

    def test_dry_run_with_return_metadata(self, sample_frame):
        """dry_run=True works with return_metadata=True."""
        result = ar.pipeline(
            sample_frame,
            [("strip_whitespace",)],
            dry_run=True,
            return_metadata=True
        )
        # May return tuple or single value depending on implementation
        assert result is not None


class TestPipelineVerbose:
    """Test verbose mode."""

    @pytest.fixture
    def sample_frame(self):
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        return ar.from_pandas(df)

    def test_verbose_mode_executes_pipeline(self, sample_frame):
        """verbose=True doesn't affect execution, only logging."""
        result = ar.pipeline(
            sample_frame,
            [("strip_whitespace",)],
            verbose=True
        )
        assert isinstance(result, ar.ArFrame)

    def test_verbose_with_multiple_steps(self, sample_frame):
        """verbose=True works with multiple steps."""
        result = ar.pipeline(
            sample_frame,
            [
                ("strip_whitespace",),
                ("drop_duplicates",),
            ],
            verbose=True
        )
        assert isinstance(result, ar.ArFrame)


class TestPipelineLineage:
    """Test track_lineage mode."""

    @pytest.fixture
    def sample_frame(self):
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        return ar.from_pandas(df)

    def test_track_lineage_returns_tuple(self, sample_frame):
        """track_lineage=True returns (ArFrame, LineageReport) tuple."""
        result = ar.pipeline(
            sample_frame,
            [("strip_whitespace",)],
            track_lineage=True
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        frame, lineage = result
        assert isinstance(frame, ar.ArFrame)

    def test_lineage_with_multiple_steps(self, sample_frame):
        """Lineage tracking captures multiple steps."""
        result = ar.pipeline(
            sample_frame,
            [
                ("strip_whitespace",),
                ("drop_duplicates",),
            ],
            track_lineage=True
        )
        frame, lineage = result
        assert isinstance(frame, ar.ArFrame)
        assert lineage is not None


class TestPipelineExecutionOrder:
    """Test that steps execute in correct order."""

    def test_steps_execute_in_specified_order(self):
        """Steps execute in the order specified."""
        df = pd.DataFrame({
            "name": ["  Alice  ", "  Bob  ", "  Bob  "],
            "id": [1, 2, 2]
        })
        frame = ar.from_pandas(df)

        # Strip whitespace, then drop duplicates
        result = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("drop_duplicates",),
        ])

        result_df = ar.to_pandas(result)
        # Should have 2 rows after deduplication
        assert result_df.shape[0] == 2

    def test_repeated_same_step(self):
        """Pipeline can apply the same step multiple times."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        frame = ar.from_pandas(df)

        result = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("strip_whitespace",),
        ])
        assert isinstance(result, ar.ArFrame)


class TestPipelineEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_steps_list(self):
        """Pipeline with empty steps list returns unchanged frame."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        frame = ar.from_pandas(df)
        result = ar.pipeline(frame, [])
        assert isinstance(result, ar.ArFrame)
        assert ar.to_pandas(result).shape == (3, 1)

    def test_single_row_frame(self):
        """Pipeline works with single-row frame."""
        df = pd.DataFrame({"a": [1], "b": ["test"]})
        frame = ar.from_pandas(df)
        result = ar.pipeline(frame, [("strip_whitespace",)])
        assert ar.to_pandas(result).shape[0] == 1

    def test_empty_frame(self):
        """Pipeline works with empty frame (zero rows)."""
        df = pd.DataFrame({"a": pd.Series([], dtype="int64"), "b": pd.Series([], dtype="object")})
        frame = ar.from_pandas(df)
        result = ar.pipeline(frame, [("strip_whitespace",)])
        assert ar.to_pandas(result).shape[0] == 0

    def test_many_sequential_steps(self):
        """Pipeline with many steps executes correctly."""
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        frame = ar.from_pandas(df)

        steps = [("strip_whitespace",)] * 10
        result = ar.pipeline(frame, steps)
        assert isinstance(result, ar.ArFrame)
        assert ar.to_pandas(result).shape[0] == 2


class TestPipelineWithDifferentSteps:
    """Test pipeline with various built-in steps."""

    def test_pipeline_with_drop_nulls(self):
        """Pipeline executes drop_nulls step."""
        df = pd.DataFrame({
            "id": [1, 2, None, 4],
            "name": ["Alice", "Bob", "Charlie", "Dave"]
        })
        frame = ar.from_pandas(df)
        result = ar.pipeline(frame, [("drop_nulls",)])
        result_df = ar.to_pandas(result)
        assert result_df.shape[0] < 4

    def test_pipeline_with_drop_duplicates(self):
        """Pipeline executes drop_duplicates step."""
        df = pd.DataFrame({
            "id": [1, 2, 2, 3],
            "name": ["A", "B", "B", "C"]
        })
        frame = ar.from_pandas(df)
        result = ar.pipeline(frame, [
            ("drop_duplicates", {"keep": "first"})
        ])
        result_df = ar.to_pandas(result)
        assert result_df.shape[0] == 3

    def test_pipeline_with_fill_nulls(self):
        """Pipeline executes fill_nulls step."""
        df = pd.DataFrame({
            "value": [1.0, None, 3.0]
        })
        frame = ar.from_pandas(df)
        result = ar.pipeline(frame, [
            ("fill_nulls", {"value": 0.0})
        ])
        result_df = ar.to_pandas(result)
        assert result_df.shape[0] == 3
