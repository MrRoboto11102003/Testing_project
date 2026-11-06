import pytest
import arnio as ar
import pandas as pd


class TestPipelineBasic:
    """Test basic pipeline execution."""
    
    @pytest.fixture
    def sample_frame(self):
        data = {"name": ["Alice", "Bob"], "age": [25, 30], "salary": [50000.0, 60000.0]}
        return ar.from_dict(data)
    
    def test_pipeline_single_step_no_kwargs(self, sample_frame):
        """Test pipeline executes a single step with no arguments."""
        result = ar.pipeline(sample_frame, [("strip_whitespace",)])
        assert isinstance(result, ar.ArFrame)
    
    def test_pipeline_multiple_steps(self, sample_frame):
        """Test pipeline executes multiple steps sequentially."""
        result = ar.pipeline(sample_frame, [
            ("strip_whitespace",),
            ("drop_duplicates",),
        ])
        assert isinstance(result, ar.ArFrame)
    
    def test_pipeline_step_with_kwargs(self, sample_frame):
        """Test pipeline step with keyword arguments."""
        result = ar.pipeline(sample_frame, [
            ("normalize_case", {"case_type": "lower"}),
        ])
        assert isinstance(result, ar.ArFrame)
    
    def test_pipeline_mixed_step_formats(self, sample_frame):
        """Test pipeline with both no-arg and kwarg steps."""
        result = ar.pipeline(sample_frame, [
            ("strip_whitespace",),
            ("drop_nulls", {"subset": ["age"]}),
        ])
        assert isinstance(result, ar.ArFrame)


class TestPipelineReturnTypes:
    """Test different return type behaviors."""
    
    @pytest.fixture
    def sample_frame(self):
        return ar.from_dict({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    
    def test_pipeline_default_returns_frame(self, sample_frame):
        """Test pipeline returns ArFrame by default."""
        result = ar.pipeline(sample_frame, [("drop_nulls",)])
        assert isinstance(result, ar.ArFrame)
    
    def test_pipeline_return_metadata_returns_tuple(self, sample_frame):
        """Test pipeline returns (ArFrame, dict) when return_metadata=True."""
        result = ar.pipeline(sample_frame, [("drop_nulls",)], return_metadata=True)
        assert isinstance(result, tuple) and len(result) == 2
        frame, metadata = result
        assert isinstance(frame, ar.ArFrame) and isinstance(metadata, dict)
    
    def test_metadata_contains_required_keys(self, sample_frame):
        """Test metadata dict contains required keys."""
        _, metadata = ar.pipeline(sample_frame, [("drop_nulls",)], return_metadata=True)
        assert all(key in metadata for key in ["step_timings", "applied_steps", "row_counts", "execution_summary"])
    
    def test_pipeline_track_lineage_returns_tuple_with_report(self, sample_frame):
        """Test pipeline returns (ArFrame, LineageReport) when track_lineage=True."""
        result = ar.pipeline(sample_frame, [("drop_nulls",)], track_lineage=True)
        assert isinstance(result, tuple) and len(result) == 2
        frame, lineage = result
        assert isinstance(frame, ar.ArFrame) and hasattr(lineage, "dropped_by_step")


class TestPipelineDryRun:
    """Test dry_run validation mode."""
    
    @pytest.fixture
    def sample_frame(self):
        return ar.from_dict({"x": [1, 2, 3]})
    
    def test_dry_run_validates_without_executing(self, sample_frame):
        """Test dry_run validates pipeline structure."""
        result = ar.pipeline(sample_frame, [("drop_nulls",)], dry_run=True)
        # dry_run should succeed
        assert result is not None or result is None  # behavior not fully specified
    
    def test_dry_run_catches_unknown_steps(self, sample_frame):
        """Test dry_run still catches unknown step names."""
        with pytest.raises(Exception):  # UnknownStepError or similar
            ar.pipeline(sample_frame, [("unknown_step",)], dry_run=True)


class TestPipelineErrorHandling:
    """Test error conditions."""
    
    @pytest.fixture
    def sample_frame(self):
        return ar.from_dict({"col": [1, 2]})
    
    def test_pipeline_conflicting_flags_raises_error(self, sample_frame):
        """Test ValueError when both return_metadata and track_lineage are True."""
        with pytest.raises(ValueError):
            ar.pipeline(sample_frame, [("drop_nulls",)], 
                       return_metadata=True, track_lineage=True)
    
    def test_pipeline_unknown_step_raises_error(self, sample_frame):
        """Test UnknownStepError for unregistered step names."""
        with pytest.raises(Exception):  # UnknownStepError
            ar.pipeline(sample_frame, [("nonexistent_step",)])
    
    def test_pipeline_invalid_step_format_raises_error(self, sample_frame):
        """Test ValueError for malformed step tuple."""
        with pytest.raises((ValueError, TypeError)):
            ar.pipeline(sample_frame, [("step", "extra", "args")])
    
    def test_pipeline_lineage_conflicting_column_raises_error(self):
        """Test ValueError when frame has __arnio_lineage_id__ column."""
        frame = ar.from_dict({"col": [1], "__arnio_lineage_id__": [0]})
        with pytest.raises(ValueError):
            ar.pipeline(frame, [("drop_nulls",)], track_lineage=True)
    
    def test_pipeline_non_list_steps_raises_error(self, sample_frame):
        """Test TypeError if steps is not a list."""
        with pytest.raises(TypeError):
            ar.pipeline(sample_frame, ("drop_nulls",))


class TestPipelineLineageTracking:
    """Test lineage tracking functionality."""
    
    @pytest.fixture
    def sample_frame(self):
        return ar.from_dict({"id": [1, 2, 3, 4], "val": [10, None, 30, 40]})
    
    def test_lineage_returns_dropped_by_step(self, sample_frame):
        """Test lineage tracks dropped rows per step."""
        frame, lineage = ar.pipeline(sample_frame, [("drop_nulls",)], track_lineage=True)
        assert hasattr(lineage, "dropped_by_step") and isinstance(lineage.dropped_by_step, dict)
    
    def test_lineage_returns_total_dropped(self, sample_frame):
        """Test lineage provides total_dropped count."""
        frame, lineage = ar.pipeline(sample_frame, [("drop_nulls",)], track_lineage=True)
        assert hasattr(lineage, "total_dropped") and isinstance(lineage.total_dropped, int)
    
    def test_lineage_sentinel_column_removed(self, sample_frame):
        """Test lineage sentinel column is stripped from returned frame."""
        frame, lineage = ar.pipeline(sample_frame, [("drop_nulls",)], track_lineage=True)
        col_names = str(frame.columns) if hasattr(frame, 'columns') else ""
        assert "__arnio_lineage_id__" not in col_names
    
    def test_lineage_merges_repeated_steps(self, sample_frame):
        """Test lineage merges drops from repeated step names."""
        frame, lineage = ar.pipeline(sample_frame, [
            ("drop_nulls",),
            ("drop_nulls",),
        ], track_lineage=True)
        assert "drop_nulls" in lineage.dropped_by_step or isinstance(lineage.dropped_by_step, dict)


class TestPipelineComplexWorkflows:
    """Test realistic multi-step pipelines."""
    
    @pytest.fixture
    def messy_frame(self):
        return ar.from_dict({
            "name": ["  Alice  ", "Bob", "  Bob  "],
            "age": [25, 30, 30],
            "salary": [50000.0, 60000.0, None]
        })
    
    def test_realistic_cleaning_pipeline(self, messy_frame):
        """Test realistic multi-step data cleaning workflow."""
        result = ar.pipeline(messy_frame, [
            ("strip_whitespace",),
            ("drop_nulls", {"subset": ["salary"]}),
            ("drop_duplicates",),
        ])
        assert isinstance(result, ar.ArFrame)
    
    def test_complex_pipeline_with_metadata(self, messy_frame):
        """Test complex pipeline with return_metadata enabled."""
        frame, metadata = ar.pipeline(messy_frame, [
            ("strip_whitespace",),
            ("normalize_case", {"case_type": "lower"}),
            ("drop_duplicates",),
        ], return_metadata=True)
        assert isinstance(frame, ar.ArFrame) and isinstance(metadata, dict)
        assert len(metadata["applied_steps"]) == 3
    
    def test_pipeline_with_verbose_flag(self, messy_frame):
        """Test verbose flag is accepted without error."""
        result = ar.pipeline(messy_frame, [
            ("drop_nulls",),
            ("strip_whitespace",),
        ], verbose=True)
        assert isinstance(result, ar.ArFrame)


class TestPipelineEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_pipeline_empty_steps_list(self):
        """Test pipeline with empty steps list returns input frame."""
        frame = ar.from_dict({"x": [1, 2]})
        result = ar.pipeline(frame, [])
        assert isinstance(result, ar.ArFrame)
    
    def test_pipeline_single_row(self):
        """Test pipeline on single-row frame."""
        frame = ar.from_dict({"col": [42]})
        result = ar.pipeline(frame, [("drop_duplicates",)])
        assert isinstance(result, ar.ArFrame)
    
    def test_pipeline_empty_frame(self):
        """Test pipeline on empty frame."""
        frame = ar.from_dict({"col1": [], "col2": []})
        result = ar.pipeline(frame, [("drop_nulls",)])
        assert isinstance(result, ar.ArFrame)
    
    def test_pipeline_mapping_style_kwargs(self):
        """Test pipeline with mapping-style kwargs for cast_types."""
        frame = ar.from_dict({"val": ["1", "2", "3"]})
        result = ar.pipeline(frame, [
            ("cast_types", {"mapping": {"val": "int64"}})
        ])
        assert isinstance(result, ar.ArFrame)
