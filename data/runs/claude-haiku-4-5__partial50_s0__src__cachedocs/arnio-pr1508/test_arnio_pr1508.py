import pytest
import pandas as pd
import arnio as ar
from arnio.exceptions import UnknownStepError


class TestPipelineBasic:
    """Test basic pipeline functionality."""

    @pytest.fixture
    def sample_frame(self):
        """Create a sample ArFrame for testing."""
        df = pd.DataFrame({
            "name": ["  Alice  ", "  Bob  ", "  Bob  "],
            "age": [25, 30, 30],
        })
        return ar.from_pandas(df)

    def test_pipeline_empty_steps(self, sample_frame):
        """Pipeline with empty steps list returns unchanged frame."""
        result = ar.pipeline(sample_frame, [])
        assert result.shape == sample_frame.shape
        assert result.columns == sample_frame.columns

    def test_pipeline_single_step(self, sample_frame):
        """Pipeline with single step applies transformation."""
        result = ar.pipeline(sample_frame, [("strip_whitespace",)])
        result_df = ar.to_pandas(result)
        assert result_df["name"].iloc[0] == "Alice"
        assert result_df["name"].iloc[1] == "Bob"

    def test_pipeline_multiple_steps(self, sample_frame):
        """Pipeline applies multiple steps sequentially."""
        result = ar.pipeline(sample_frame, [
            ("strip_whitespace",),
            ("drop_duplicates",),
        ])
        result_df = ar.to_pandas(result)
        assert result_df.shape[0] == 2  # One duplicate removed

    def test_pipeline_step_with_kwargs(self, sample_frame):
        """Pipeline correctly handles steps with keyword arguments."""
        result = ar.pipeline(sample_frame, [
            ("drop_duplicates", {"keep": "first"}),
        ])
        result_df = ar.to_pandas(result)
        assert result_df.shape[0] == 2


class TestPipelineMetadata:
    """Test return_metadata parameter and return value structure."""

    @pytest.fixture
    def sample_frame(self):
        df = pd.DataFrame({
            "name": ["Alice", "Bob", "Bob"],
            "age": [25, 30, 30],
        })
        return ar.from_pandas(df)

    def test_return_metadata_structure(self, sample_frame):
        """Metadata dict contains all expected keys."""
        result, metadata = ar.pipeline(
            sample_frame,
            [("drop_duplicates",)],
            return_metadata=True,
        )
        assert isinstance(metadata, dict)
        assert "applied_steps" in metadata
        assert "row_counts" in metadata
        assert "step_timings" in metadata
        assert "execution_summary" in metadata

    def test_return_metadata_applied_steps_order(self, sample_frame):
        """Metadata applied_steps list preserves step order."""
        result, metadata = ar.pipeline(
            sample_frame,
            [
                ("strip_whitespace",),
                ("drop_duplicates",),
            ],
            return_metadata=True,
        )
        assert metadata["applied_steps"] == ["strip_whitespace", "drop_duplicates"]

    def test_return_metadata_row_counts_tracking(self, sample_frame):
        """Metadata row_counts tracks before/after per step."""
        result, metadata = ar.pipeline(
            sample_frame,
            [("drop_duplicates",)],
            return_metadata=True,
        )
        assert len(metadata["row_counts"]) == 1
        assert metadata["row_counts"][0]["before"] == 3
        assert metadata["row_counts"][0]["after"] == 2

    def test_return_metadata_step_timings(self, sample_frame):
        """Metadata step_timings includes timing data."""
        result, metadata = ar.pipeline(
            sample_frame,
            [("drop_duplicates",)],
            return_metadata=True,
        )
        assert len(metadata["step_timings"]) == 1
        assert "step" in metadata["step_timings"][0]
        assert "seconds" in metadata["step_timings"][0]
        assert metadata["step_timings"][0]["dry_run"] is False


class TestPipelineDryRun:
    """Test dry_run parameter behavior."""

    @pytest.fixture
    def sample_frame(self):
        df = pd.DataFrame({
            "name": ["Alice", "Bob", "Bob"],
            "age": [25, 30, 30],
        })
        return ar.from_pandas(df)

    def test_dry_run_no_data_modification(self, sample_frame):
        """Dry run validates pipeline without modifying output."""
        original_shape = sample_frame.shape
        result = ar.pipeline(
            sample_frame,
            [("drop_duplicates",)],
            dry_run=True,
        )
        # Dry run should not remove duplicates
        assert result.shape[0] == original_shape[0]

    def test_dry_run_with_metadata_flag(self, sample_frame):
        """Dry run metadata marks steps with dry_run=True."""
        result, metadata = ar.pipeline(
            sample_frame,
            [("drop_duplicates",)],
            dry_run=True,
            return_metadata=True,
        )
        assert all(item["dry_run"] is True for item in metadata["step_timings"])


class TestPipelineLineageTracking:
    """Test track_lineage parameter and return value."""

    @pytest.fixture
    def sample_frame(self):
        df = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie", "Bob"],
            "age": [25, 30, 35, 30],
        })
        return ar.from_pandas(df)

    def test_track_lineage_return_type(self, sample_frame):
        """track_lineage=True returns (ArFrame, LineageReport) tuple."""
        result, lineage = ar.pipeline(
            sample_frame,
            [("drop_duplicates",)],
            track_lineage=True,
        )
        assert hasattr(lineage, "dropped_by_step")
        assert hasattr(lineage, "total_dropped")

    def test_track_lineage_sentinel_removed_from_output(self, sample_frame):
        """Lineage sentinel column is stripped before returning result."""
        result, lineage = ar.pipeline(
            sample_frame,
            [("strip_whitespace",)],
            track_lineage=True,
        )
        result_df = ar.to_pandas(result)
        assert "__arnio_lineage_id__" not in result_df.columns

    def test_track_lineage_sentinel_collision_error(self):
        """ValueError raised if input frame contains sentinel column."""
        df = pd.DataFrame({
            "__arnio_lineage_id__": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
        })
        frame = ar.from_pandas(df)
        with pytest.raises(ValueError, match="__arnio_lineage_id__"):
            ar.pipeline(frame, [("drop_duplicates",)], track_lineage=True)


class TestPipelineParameterValidation:
    """Test parameter type and constraint validation."""

    @pytest.fixture
    def sample_frame(self):
        df = pd.DataFrame({"name": ["Alice"], "age": [25]})
        return ar.from_pandas(df)

    def test_frame_must_be_arframe(self, sample_frame):
        """Non-ArFrame input raises TypeError."""
        df = pd.DataFrame({"name": ["Alice"]})
        with pytest.raises(TypeError, match="frame must be an ArFrame"):
            ar.pipeline(df, [])

    def test_return_metadata_must_be_bool(self, sample_frame):
        """Non-bool return_metadata raises TypeError."""
        with pytest.raises(TypeError, match="return_metadata"):
            ar.pipeline(sample_frame, [], return_metadata="yes")

    def test_dry_run_must_be_bool(self, sample_frame):
        """Non-bool dry_run raises TypeError."""
        with pytest.raises(TypeError, match="dry_run"):
            ar.pipeline(sample_frame, [], dry_run=1)

    def test_verbose_must_be_bool(self, sample_frame):
        """Non-bool verbose raises TypeError."""
        with pytest.raises(TypeError, match="verbose"):
            ar.pipeline(sample_frame, [], verbose=[])

    def test_track_lineage_must_be_bool(self, sample_frame):
        """Non-bool track_lineage raises TypeError."""
        with pytest.raises(TypeError, match="track_lineage"):
            ar.pipeline(sample_frame, [], track_lineage="yes")

    def test_track_lineage_return_metadata_mutual_exclusion(self, sample_frame):
        """track_lineage=True and return_metadata=True raises ValueError."""
        with pytest.raises(ValueError, match="cannot be used together"):
            ar.pipeline(
                sample_frame,
                [],
                track_lineage=True,
                return_metadata=True,
            )


class TestPipelineStepValidation:
    """Test pipeline step format and name validation."""

    @pytest.fixture
    def sample_frame(self):
        df = pd.DataFrame({"name": ["Alice"]})
        return ar.from_pandas(df)

    def test_unknown_step_name_error(self, sample_frame):
        """Unknown step name raises UnknownStepError."""
        with pytest.raises(UnknownStepError):
            ar.pipeline(sample_frame, [("nonexistent_step_xyz_123",)])

    def test_invalid_step_tuple_too_many_elements(self, sample_frame):
        """Step tuple with >2 elements raises ValueError."""
        with pytest.raises(ValueError, match="Invalid step format"):
            ar.pipeline(sample_frame, [("drop_duplicates", {}, "extra")])

    def test_invalid_step_kwargs_not_dict(self, sample_frame):
        """Step kwargs must be dict, raises ValueError otherwise."""
        with pytest.raises(ValueError, match="Invalid step kwargs"):
            ar.pipeline(sample_frame, [("drop_duplicates", "not_a_dict")])


class TestPipelineCustomSteps:
    """Test custom step registration and execution in pipelines."""

    @pytest.fixture
    def sample_frame(self):
        df = pd.DataFrame({"value": [1, 2, 3]})
        return ar.from_pandas(df)

    def test_custom_step_executes_in_pipeline(self, sample_frame):
        """Registered custom step executes correctly in pipeline."""
        def custom_double(df):
            df = df.copy()
            df["value"] = df["value"] * 2
            return df

        ar.register_step("test_double", custom_double)
        try:
            result = ar.pipeline(sample_frame, [("test_double",)])
            result_df = ar.to_pandas(result)
            assert result_df["value"].tolist() == [2, 4, 6]
        finally:
            ar.reset_steps()

    def test_custom_step_returning_none_raises_error(self, sample_frame):
        """Custom step returning None raises TypeError."""
        def bad_step(df):
            return None

        ar.register_step("bad_step_none", bad_step)
        try:
            with pytest.raises(TypeError, match="returned None"):
                ar.pipeline(sample_frame, [("bad_step_none",)])
        finally:
            ar.reset_steps()

    def test_custom_step_returning_non_dataframe_raises_error(self, sample_frame):
        """Custom step returning non-DataFrame raises TypeError."""
        def bad_step(df):
            return [1, 2, 3]

        ar.register_step("bad_step_list", bad_step)
        try:
            with pytest.raises(TypeError, match="returned.*instead of"):
                ar.pipeline(sample_frame, [("bad_step_list",)])
        finally:
            ar.reset_steps()
