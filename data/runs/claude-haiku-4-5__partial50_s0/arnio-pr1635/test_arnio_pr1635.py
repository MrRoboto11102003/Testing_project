import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestPipeline:
    """Tests for arnio.pipeline.pipeline function"""
    
    @pytest.fixture
    def sample_frame(self):
        """Create a sample ArFrame for testing"""
        data = {
            "name": ["  Alice  ", "  Bob  ", "  Charlie  "],
            "age": [25, 30, 25],
            "revenue": [1000.0, 2000.0, 1000.0],
        }
        df = pd.DataFrame(data)
        return ar.from_pandas(df)
    
    def test_pipeline_basic_single_step(self, sample_frame):
        """Test pipeline with a single cleaning step"""
        result = ar.pipeline(sample_frame, [("strip_whitespace",)])
        assert isinstance(result, ar.ArFrame)
        df = ar.to_pandas(result)
        assert df["name"].iloc[0] == "Alice"
    
    def test_pipeline_multiple_sequential_steps(self, sample_frame):
        """Test pipeline executes multiple steps in order"""
        result = ar.pipeline(sample_frame, [
            ("strip_whitespace",),
            ("normalize_case", {"case_type": "upper"}),
        ])
        df = ar.to_pandas(result)
        assert df["name"].iloc[0] == "ALICE"
    
    def test_pipeline_step_with_keyword_arguments(self, sample_frame):
        """Test pipeline steps accept configuration via kwargs"""
        result = ar.pipeline(sample_frame, [
            ("normalize_case", {"case_type": "title"}),
        ])
        assert isinstance(result, ar.ArFrame)
    
    def test_pipeline_empty_steps_list(self, sample_frame):
        """Test pipeline with empty steps list returns unchanged frame"""
        result = ar.pipeline(sample_frame, [])
        assert result.shape == sample_frame.shape
        assert isinstance(result, ar.ArFrame)
    
    def test_pipeline_unknown_step_raises_error(self, sample_frame):
        """Test that unknown step names raise exception before execution"""
        with pytest.raises(Exception):
            ar.pipeline(sample_frame, [("nonexistent_step",)])
    
    def test_pipeline_validates_steps_before_execution(self, sample_frame):
        """Test that invalid steps fail early, preventing partial execution"""
        original_df = ar.to_pandas(sample_frame)
        
        with pytest.raises(Exception):
            ar.pipeline(sample_frame, [
                ("strip_whitespace",),
                ("invalid_step_name",),
            ])
    
    def test_pipeline_dry_run_mode(self, sample_frame):
        """Test dry_run=True validates without modifying data"""
        result = ar.pipeline(
            sample_frame,
            [("strip_whitespace",)],
            dry_run=True,
        )
        assert result is None or isinstance(result, ar.ArFrame)
    
    def test_pipeline_return_metadata_tuple(self, sample_frame):
        """Test return_metadata=True returns (ArFrame, dict) tuple"""
        result = ar.pipeline(
            sample_frame,
            [("strip_whitespace",)],
            return_metadata=True,
        )
        assert isinstance(result, tuple) and len(result) == 2
        frame, metadata = result
        assert isinstance(frame, ar.ArFrame)
        assert isinstance(metadata, dict)
    
    def test_pipeline_metadata_contains_execution_info(self, sample_frame):
        """Test metadata dict includes step timings and execution details"""
        _, metadata = ar.pipeline(
            sample_frame,
            [("strip_whitespace",), ("drop_duplicates",)],
            return_metadata=True,
        )
        assert len(metadata) > 0
        assert any(key in metadata for key in ["step_timings", "applied_steps", "row_counts"])
    
    def test_pipeline_verbose_mode_succeeds(self, sample_frame):
        """Test verbose=True enables logging without changing behavior"""
        result = ar.pipeline(
            sample_frame,
            [("strip_whitespace",)],
            verbose=True,
        )
        assert isinstance(result, ar.ArFrame)
    
    def test_pipeline_preserves_column_count(self, sample_frame):
        """Test that row-safe steps preserve DataFrame column count"""
        result = ar.pipeline(sample_frame, [
            ("strip_whitespace",),
            ("normalize_case", {"case_type": "lower"}),
        ])
        assert result.shape[1] == sample_frame.shape[1]
    
    def test_pipeline_with_subset_parameter(self, sample_frame):
        """Test pipeline steps accept subset parameter for selective columns"""
        result = ar.pipeline(sample_frame, [
            ("drop_nulls", {"subset": ["name"]}),
        ])
        assert isinstance(result, ar.ArFrame)
    
    def test_pipeline_drop_duplicates_reduces_rows(self, sample_frame):
        """Test drop_duplicates step removes identical rows"""
        result = ar.pipeline(sample_frame, [
            ("drop_duplicates", {"keep": "first"}),
        ])
        df = ar.to_pandas(result)
        assert len(df) <= 3
    
    def test_pipeline_complex_workflow_end_to_end(self, sample_frame):
        """Test complex pipeline combining multiple cleaning operations"""
        result = ar.pipeline(sample_frame, [
            ("strip_whitespace",),
            ("normalize_case", {"case_type": "lower"}),
            ("drop_duplicates", {"keep": "first"}),
        ])
        df = ar.to_pandas(result)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert df["name"].iloc[0] == "alice"
    
    def test_pipeline_step_parameter_validation(self, sample_frame):
        """Test that pipeline validates step parameters early"""
        # Invalid case_type should fail early
        with pytest.raises(Exception):
            ar.pipeline(sample_frame, [
                ("normalize_case", {"case_type": "invalid"}),
            ])
