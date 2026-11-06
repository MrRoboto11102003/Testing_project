import pytest
import pandas as pd
import numpy as np
import arnio as ar
from arnio.exceptions import UnknownStepError


class TestPipeline:
    """Test suite for arnio.pipeline.pipeline function"""
    
    @pytest.fixture
    def sample_frame(self):
        """Create a sample ArFrame for testing"""
        df = pd.DataFrame({
            "name": ["  Alice  ", "  Bob  ", "  Charlie  "],
            "age": [25, 30, 25],
            "city": ["Paris", "London", "Paris"],
            "score": [95.5, 88.3, 95.5]
        })
        return ar.from_pandas(df)
    
    @pytest.fixture
    def empty_frame(self):
        """Create an empty ArFrame"""
        df = pd.DataFrame({"col1": [], "col2": [], "col3": []})
        return ar.from_pandas(df)
    
    def test_pipeline_single_step(self, sample_frame):
        """Test pipeline execution with a single step"""
        result = ar.pipeline(sample_frame, [("strip_whitespace",)])
        assert isinstance(result, ar.ArFrame)
        assert result.shape == sample_frame.shape
    
    def test_pipeline_multiple_steps_sequence(self, sample_frame):
        """Test pipeline executes multiple steps in sequence"""
        result = ar.pipeline(sample_frame, [
            ("strip_whitespace",),
            ("drop_duplicates", {"keep": "first"}),
        ])
        assert isinstance(result, ar.ArFrame)
        assert result.shape[0] <= sample_frame.shape[0]
    
    def test_pipeline_empty_steps_list(self, sample_frame):
        """Test pipeline with empty steps list returns unchanged frame"""
        result = ar.pipeline(sample_frame, [])
        assert isinstance(result, ar.ArFrame)
        assert result.shape == sample_frame.shape
    
    def test_pipeline_early_validation_unknown_step(self, sample_frame):
        """Test that unknown step names fail before execution begins"""
        with pytest.raises(UnknownStepError):
            ar.pipeline(sample_frame, [
                ("strip_whitespace",),
                ("nonexistent_step",),
            ])
    
    def test_pipeline_early_validation_malformed_tuple(self, sample_frame):
        """Test that malformed step tuples fail early before execution"""
        with pytest.raises((ValueError, TypeError, UnknownStepError)):
            ar.pipeline(sample_frame, [
                ("strip_whitespace", "extra", "positional", "args"),
            ])
    
    def test_pipeline_step_with_kwargs(self, sample_frame):
        """Test pipeline step with keyword arguments"""
        result = ar.pipeline(sample_frame, [
            ("normalize_case", {"case_type": "lower"}),
            ("drop_duplicates", {"keep": "first"}),
        ])
        assert isinstance(result, ar.ArFrame)
    
    def test_pipeline_return_metadata_default(self, sample_frame):
        """Test pipeline returns ArFrame by default (return_metadata=False)"""
        result = ar.pipeline(sample_frame, [
            ("strip_whitespace",)
        ])
        assert isinstance(result, ar.ArFrame)
        assert not isinstance(result, tuple)
    
    def test_pipeline_return_metadata_true(self, sample_frame):
        """Test pipeline returns tuple when return_metadata=True"""
        result = ar.pipeline(sample_frame, [
            ("strip_whitespace",)
        ], return_metadata=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], ar.ArFrame)
        assert isinstance(result[1], dict)
    
    def test_pipeline_track_lineage(self, sample_frame):
        """Test pipeline with track_lineage=True returns frame and lineage report"""
        result = ar.pipeline(sample_frame, [
            ("strip_whitespace",),
            ("drop_duplicates",),
        ], track_lineage=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], ar.ArFrame)
    
    def test_pipeline_verbose_mode(self, sample_frame):
        """Test pipeline execution with verbose=True"""
        result = ar.pipeline(sample_frame, [
            ("strip_whitespace",),
            ("normalize_case", {"case_type": "lower"}),
        ], verbose=True)
        assert isinstance(result, ar.ArFrame)
    
    def test_pipeline_with_empty_frame(self, empty_frame):
        """Test pipeline works correctly with empty ArFrame"""
        result = ar.pipeline(empty_frame, [
            ("strip_whitespace",),
            ("drop_nulls",),
        ])
        assert isinstance(result, ar.ArFrame)
        assert result.shape[0] == 0
    
    def test_pipeline_complex_workflow(self, sample_frame):
        """Test complex multi-step pipeline with mixed step types"""
        result = ar.pipeline(sample_frame, [
            ("strip_whitespace",),
            ("normalize_case", {"case_type": "title"}),
            ("drop_duplicates", {"keep": "first"}),
        ])
        assert isinstance(result, ar.ArFrame)
        assert result.shape[0] <= sample_frame.shape[0]
    
    def test_pipeline_fill_nulls_step(self):
        """Test pipeline with fill_nulls step and parameters"""
        df = pd.DataFrame({
            "value": [1.0, None, 3.0],
            "category": ["A", "B", "C"]
        })
        frame = ar.from_pandas(df)
        result = ar.pipeline(frame, [
            ("fill_nulls", {"value": 0}),
        ])
        assert isinstance(result, ar.ArFrame)
        assert result.shape == frame.shape
    
    def test_pipeline_combination_metadata_and_verbose(self, sample_frame):
        """Test pipeline with both return_metadata and verbose flags"""
        result = ar.pipeline(sample_frame, [
            ("strip_whitespace",),
            ("drop_duplicates",),
        ], return_metadata=True, verbose=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], ar.ArFrame)
        assert isinstance(result[1], dict)
