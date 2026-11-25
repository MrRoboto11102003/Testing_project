import pytest
import pandas as pd
import arnio as ar


class TestPipelineBasic:
    """Test basic pipeline functionality."""
    
    @pytest.fixture
    def sample_frame(self):
        """Create sample ArFrame for testing."""
        data = {
            "name": ["Alice", "Bob", "Alice", "Charlie"],
            "age": [25, 30, 25, None],
            "score": [85.5, None, 85.5, 95.0]
        }
        return ar.from_dict(data)
    
    def test_pipeline_single_step_no_kwargs(self, sample_frame):
        """Pipeline executes single step without arguments."""
        result = ar.pipeline(sample_frame, [("drop_duplicates",)])
        assert isinstance(result, ar.ArFrame)
    
    def test_pipeline_single_step_with_kwargs(self, sample_frame):
        """Pipeline executes single step with kwargs."""
        result = ar.pipeline(sample_frame, [
            ("fill_nulls", {"value": 0.0, "subset": ["score"]})
        ])
        assert isinstance(result, ar.ArFrame)
    
    def test_pipeline_multiple_steps_sequential(self, sample_frame):
        """Pipeline executes multiple steps in sequence."""
        result = ar.pipeline(sample_frame, [
            ("strip_whitespace",),
            ("fill_nulls", {"value": 0.0, "subset": ["score"]}),
            ("drop_nulls", {"subset": ["age"]}),
            ("drop_duplicates",),
        ])
        assert isinstance(result, ar.ArFrame)
    
    def test_pipeline_empty_steps_list(self, sample_frame):
        """Pipeline handles empty steps list."""
        result = ar.pipeline(sample_frame, [])
        assert isinstance(result, ar.ArFrame)
    
    def test_pipeline_result_to_pandas(self, sample_frame):
        """Pipeline result converts to pandas DataFrame."""
        result = ar.pipeline(sample_frame, [("drop_duplicates",)])
        df = ar.to_pandas(result)
        assert isinstance(df, pd.DataFrame)


class TestPipelineMetadata:
    """Test metadata return functionality."""
    
    @pytest.fixture
    def sample_frame(self):
        return ar.from_dict({"a": [1, 2, 1], "b": [3, 4, 3]})
    
    def test_pipeline_return_metadata_tuple(self, sample_frame):
        """return_metadata=True returns (frame, metadata_dict)."""
        result, metadata = ar.pipeline(
            sample_frame,
            [("drop_duplicates",)],
            return_metadata=True
        )
        assert isinstance(result, ar.ArFrame)
        assert isinstance(metadata, dict)
    
    def test_pipeline_metadata_contains_required_keys(self, sample_frame):
        """Metadata dict contains step_timings, applied_steps, row_counts, execution_summary."""
        result, metadata = ar.pipeline(
            sample_frame,
            [("strip_whitespace",), ("drop_duplicates",)],
            return_metadata=True
        )
        assert "step_timings" in metadata
        assert "applied_steps" in metadata
        assert "row_counts" in metadata
        assert "execution_summary" in metadata
    
    def test_pipeline_metadata_applied_steps_count(self, sample_frame):
        """Applied steps list length matches number of pipeline steps."""
        result, metadata = ar.pipeline(
            sample_frame,
            [("strip_whitespace",), ("drop_duplicates",), ("drop_nulls",)],
            return_metadata=True
        )
        assert len(metadata["applied_steps"]) == 3


class TestPipelineLineage:
    """Test lineage tracking functionality."""
    
    @pytest.fixture
    def sample_frame(self):
        return ar.from_dict({
            "name": ["Alice", "Bob", None, "Alice"],
            "value": [1, 2, 3, 1]
        })
    
    def test_pipeline_track_lineage_returns_report(self, sample_frame):
        """track_lineage=True returns (frame, LineageReport)."""
        result, lineage = ar.pipeline(
            sample_frame,
            [("drop_nulls",)],
            track_lineage=True
        )
        assert isinstance(result, ar.ArFrame)
        assert hasattr(lineage, "dropped_by_step")
        assert hasattr(lineage, "total_dropped")
    
    def test_pipeline_lineage_merges_duplicate_steps(self, sample_frame):
        """Same step name invoked twice merges drops under single key."""
        result, lineage = ar.pipeline(
            sample_frame,
            [("drop_duplicates",), ("drop_duplicates",)],
            track_lineage=True
        )
        assert "drop_duplicates" in lineage.dropped_by_step
        assert isinstance(lineage.dropped_by_step["drop_duplicates"], list)
    
    def test_pipeline_lineage_sentinel_column_conflict(self):
        """ValueError raised if input frame contains __arnio_lineage_id__ column."""
        frame = ar.from_dict({
            "__arnio_lineage_id__": [1, 2],
            "data": ["x", "y"]
        })
        with pytest.raises(ValueError):
            ar.pipeline(frame, [("drop_nulls",)], track_lineage=True)


class TestPipelineErrors:
    """Test error handling and validation."""
    
    @pytest.fixture
    def sample_frame(self):
        return ar.from_dict({"a": [1, 2], "b": [3, 4]})
    
    def test_pipeline_unknown_step_raises_error(self, sample_frame):
        """Unknown step name raises UnknownStepError."""
        with pytest.raises(Exception):
            ar.pipeline(sample_frame, [("nonexistent_step",)])
    
    def test_pipeline_malformed_step_tuple_raises_error(self, sample_frame):
        """Step tuple with more than 2 elements raises ValueError."""
        with pytest.raises(ValueError):
            ar.pipeline(sample_frame, [("step_name", {}, "extra_arg")])
    
    def test_pipeline_conflicting_metadata_and_lineage_flags(self, sample_frame):
        """ValueError raised when both return_metadata=True and track_lineage=True."""
        with pytest.raises(ValueError):
            ar.pipeline(
                sample_frame,
                [("drop_nulls",)],
                return_metadata=True,
                track_lineage=True
            )
    
    def test_pipeline_invalid_steps_type_raises_error(self, sample_frame):
        """Non-list steps parameter raises TypeError."""
        with pytest.raises(TypeError):
            ar.pipeline(sample_frame, "not_a_list")
    
    def test_pipeline_invalid_frame_type_raises_error(self):
        """Non-ArFrame frame parameter raises TypeError."""
        with pytest.raises(TypeError):
            ar.pipeline({"not": "frame"}, [("step_name",)])


class TestPipelineDryRun:
    """Test dry_run validation mode."""
    
    @pytest.fixture
    def sample_frame(self):
        return ar.from_dict({"a": [1, 2], "b": [3, 4]})
    
    def test_pipeline_dry_run_validates_without_transform(self, sample_frame):
        """dry_run=True validates structure without returning transformed output."""
        result = ar.pipeline(
            sample_frame,
            [("drop_nulls",)],
            dry_run=True
        )
        assert result is None
    
    def test_pipeline_dry_run_catches_invalid_steps(self, sample_frame):
        """dry_run=True catches invalid step names before execution."""
        with pytest.raises(Exception):
            ar.pipeline(
                sample_frame,
                [("invalid_step",)],
                dry_run=True
            )
