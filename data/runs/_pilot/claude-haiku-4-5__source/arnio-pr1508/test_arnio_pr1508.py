import pytest
import pandas as pd
import arnio as ar


class TestPipelineBasicFunctionality:
    """Test basic pipeline execution and return types."""
    
    @pytest.fixture
    def sample_frame(self):
        """Create a sample ArFrame for testing."""
        data = {
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
            "score": [90.5, 85.0, 95.5]
        }
        return ar.from_dict(data)
    
    def test_pipeline_returns_arframe_by_default(self, sample_frame):
        """Test that pipeline returns an ArFrame when no options are set."""
        result = ar.pipeline(sample_frame, [])
        assert isinstance(result, ar.ArFrame)
    
    def test_pipeline_with_single_step(self, sample_frame):
        """Test pipeline execution with a single step."""
        result = ar.pipeline(sample_frame, [("drop_nulls",)])
        assert isinstance(result, ar.ArFrame)
    
    def test_pipeline_with_multiple_steps(self, sample_frame):
        """Test pipeline execution with multiple sequential steps."""
        result = ar.pipeline(sample_frame, [
            ("drop_nulls",),
            ("drop_duplicates",),
        ])
        assert isinstance(result, ar.ArFrame)
    
    def test_pipeline_with_step_kwargs(self, sample_frame):
        """Test pipeline step with kwargs dictionary."""
        result = ar.pipeline(sample_frame, [
            ("drop_nulls", {"subset": ["name"]})
        ])
        assert isinstance(result, ar.ArFrame)


class TestPipelineParameterValidation:
    """Test type validation of all pipeline parameters."""
    
    @pytest.fixture
    def sample_frame(self):
        data = {"col": [1, 2, 3]}
        return ar.from_dict(data)
    
    def test_frame_parameter_must_be_arframe(self, sample_frame):
        """Test that non-ArFrame frame parameter raises TypeError."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        with pytest.raises(TypeError, match="frame must be an ArFrame"):
            ar.pipeline(df, [])
    
    def test_return_metadata_must_be_bool(self, sample_frame):
        """Test that return_metadata must be boolean."""
        with pytest.raises(TypeError, match="return_metadata must be a bool"):
            ar.pipeline(sample_frame, [], return_metadata="true")
    
    def test_dry_run_must_be_bool(self, sample_frame):
        """Test that dry_run must be boolean."""
        with pytest.raises(TypeError, match="dry_run must be a bool"):
            ar.pipeline(sample_frame, [], dry_run=1)
    
    def test_verbose_must_be_bool(self, sample_frame):
        """Test that verbose must be boolean."""
        with pytest.raises(TypeError, match="verbose must be a bool"):
            ar.pipeline(sample_frame, [], verbose="yes")
    
    def test_track_lineage_must_be_bool(self, sample_frame):
        """Test that track_lineage must be boolean."""
        with pytest.raises(TypeError, match="track_lineage must be a bool"):
            ar.pipeline(sample_frame, [], track_lineage=None)


class TestPipelineReturnMetadata:
    """Test return_metadata parameter and metadata dict structure."""
    
    @pytest.fixture
    def sample_frame(self):
        data = {"col": [1, 2, 3]}
        return ar.from_dict(data)
    
    def test_return_metadata_true_returns_tuple(self, sample_frame):
        """Test that return_metadata=True returns tuple of (ArFrame, dict)."""
        result = ar.pipeline(
            sample_frame,
            [("drop_nulls",)],
            return_metadata=True
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        frame, metadata = result
        assert isinstance(frame, ar.ArFrame)
        assert isinstance(metadata, dict)
    
    def test_metadata_dict_contains_required_keys(self, sample_frame):
        """Test that metadata dict contains all required keys."""
        _, metadata = ar.pipeline(
            sample_frame,
            [("drop_nulls",)],
            return_metadata=True
        )
        assert "applied_steps" in metadata
        assert "step_timings" in metadata
        assert "row_counts" in metadata
        assert "execution_summary" in metadata
    
    def test_metadata_step_timings_structure(self, sample_frame):
        """Test that step_timings list contains timing information."""
        _, metadata = ar.pipeline(
            sample_frame,
            [("drop_nulls",)],
            return_metadata=True
        )
        timings = metadata["step_timings"]
        assert len(timings) == 1
        assert "step" in timings[0]
        assert "seconds" in timings[0]
        assert isinstance(timings[0]["seconds"], (int, float))
    
    def test_metadata_row_counts_structure(self, sample_frame):
        """Test that row_counts list tracks before/after row counts."""
        _, metadata = ar.pipeline(
            sample_frame,
            [("drop_nulls",)],
            return_metadata=True
        )
        counts = metadata["row_counts"]
        assert len(counts) >= 1
        assert "before" in counts[0]
        assert "after" in counts[0]
        assert isinstance(counts[0]["before"], int)
        assert isinstance(counts[0]["after"], int)


class TestPipelineDryRun:
    """Test dry_run parameter."""
    
    @pytest.fixture
    def sample_frame(self):
        data = {"col": [1, None, 3]}
        return ar.from_dict(data)
    
    def test_dry_run_validates_without_executing(self, sample_frame):
        """Test that dry_run=True validates pipeline without modifying data."""
        result = ar.pipeline(
            sample_frame,
            [("drop_nulls",)],
            dry_run=True
        )
        assert isinstance(result, ar.ArFrame)


class TestPipelineLineageTracking:
    """Test track_lineage parameter."""
    
    @pytest.fixture
    def sample_frame(self):
        data = {"name": ["Alice", "Bob", "Charlie"], "val": [1, 2, 3]}
        return ar.from_dict(data)
    
    def test_track_lineage_returns_frame_and_report(self, sample_frame):
        """Test that track_lineage=True returns (ArFrame, LineageReport)."""
        result = ar.pipeline(
            sample_frame,
            [("drop_nulls",)],
            track_lineage=True
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        frame, lineage_report = result
        assert isinstance(frame, ar.ArFrame)
    
    def test_track_lineage_mutually_exclusive_with_return_metadata(self, sample_frame):
        """Test that track_lineage and return_metadata cannot both be True."""
        with pytest.raises(ValueError, match="track_lineage=True and return_metadata=True cannot be used together"):
            ar.pipeline(
                sample_frame,
                [("drop_nulls",)],
                track_lineage=True,
                return_metadata=True
            )
    
    def test_track_lineage_detects_sentinel_column_collision(self):
        """Test that existing __arnio_lineage_id__ column raises ValueError."""
        data = {"__arnio_lineage_id__": [0, 1, 2], "col": [1, 2, 3]}
        frame = ar.from_dict(data)
        
        with pytest.raises(ValueError, match="__arnio_lineage_id__"):
            ar.pipeline(
                frame,
                [("drop_nulls",)],
                track_lineage=True
            )


class TestPipelineStepValidation:
    """Test step tuple format and content validation."""
    
    @pytest.fixture
    def sample_frame(self):
        data = {"col": [1, 2, 3]}
        return ar.from_dict(data)
    
    def test_unknown_step_raises_error(self, sample_frame):
        """Test that unknown step name raises an error."""
        with pytest.raises(Exception):
            ar.pipeline(sample_frame, [("nonexistent_step_xyz",)])
    
    def test_invalid_step_tuple_too_many_elements(self, sample_frame):
        """Test that step tuple with more than 2 elements raises ValueError."""
        with pytest.raises(ValueError, match="Invalid step format"):
            ar.pipeline(sample_frame, [("drop_nulls", {}, "extra")])
    
    def test_invalid_step_kwargs_not_dict(self, sample_frame):
        """Test that non-dict kwargs raises ValueError."""
        with pytest.raises(ValueError, match="Invalid step kwargs|Expected a dict"):
            ar.pipeline(sample_frame, [("drop_nulls", "not_a_dict")])


class TestPipelineVerboseMode:
    """Test verbose parameter."""
    
    @pytest.fixture
    def sample_frame(self):
        data = {"col": [1, 2, 3]}
        return ar.from_dict(data)
    
    def test_verbose_mode_does_not_raise(self, sample_frame):
        """Test that verbose=True enables logging without raising errors."""
        result = ar.pipeline(
            sample_frame,
            [("drop_nulls",)],
            verbose=True
        )
        assert isinstance(result, ar.ArFrame)


class TestPipelineComplexScenarios:
    """Test complex pipeline scenarios."""
    
    def test_pipeline_with_whitespace_and_duplicates(self):
        """Test pipeline handling whitespace stripping and duplicate removal."""
        data = {
            "name": ["  Alice  ", "  Alice  ", "  Bob  "],
            "score": [90, 90, 85]
        }
        frame = ar.from_dict(data)
        
        result = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("drop_duplicates",),
        ])
        
        assert isinstance(result, ar.ArFrame)
        assert result.shape[0] <= frame.shape[0]
    
    def test_pipeline_metadata_and_verbose_together(self):
        """Test that return_metadata and verbose can be used together."""
        data = {"col": [1, 2, 3]}
        frame = ar.from_dict(data)
        
        result = ar.pipeline(
            frame,
            [("drop_nulls",)],
            return_metadata=True,
            verbose=True
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        frame_result, metadata = result
        assert isinstance(frame_result, ar.ArFrame)
        assert isinstance(metadata, dict)
