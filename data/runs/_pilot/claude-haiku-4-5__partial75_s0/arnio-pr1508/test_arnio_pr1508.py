import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestPipeline:
    """Test arnio.pipeline.pipeline function"""
    
    @pytest.fixture
    def simple_frame(self):
        """Create a simple test frame"""
        df = pd.DataFrame({
            "name": ["alice", "bob", "charlie"],
            "age": ["25", "30", "35"],
            "score": ["95.5", "88.0", "92.3"]
        })
        return ar.from_pandas(df)
    
    @pytest.fixture
    def dirty_frame(self):
        """Create a frame with whitespace and duplicates"""
        df = pd.DataFrame({
            "name": [" alice ", "  bob  ", " alice "],
            "age": ["25", "30", "25"],
            "city": [" NY ", " LA ", " NY "]
        })
        return ar.from_pandas(df)
    
    @pytest.fixture
    def frame_with_nulls(self):
        """Create a frame with null values"""
        df = pd.DataFrame({
            "id": ["1", "2", "3", "4"],
            "name": ["alice", None, "charlie", "dave"],
            "score": ["95", "88", None, "92"]
        })
        return ar.from_pandas(df)
    
    def test_single_step_no_kwargs(self, simple_frame):
        """Test pipeline with single step and no kwargs"""
        result = ar.pipeline(simple_frame, [("strip_whitespace",)])
        assert isinstance(result, ar.ArFrame)
        assert result.shape == simple_frame.shape
    
    def test_multiple_steps(self, dirty_frame):
        """Test pipeline with multiple steps"""
        steps = [
            ("strip_whitespace",),
            ("drop_duplicates",)
        ]
        result = ar.pipeline(dirty_frame, steps)
        assert isinstance(result, ar.ArFrame)
        assert result.shape[1] == dirty_frame.shape[1]
    
    def test_step_with_kwargs(self, simple_frame):
        """Test pipeline step with keyword arguments"""
        steps = [("normalize_case", {"case_type": "lower"})]
        result = ar.pipeline(simple_frame, steps)
        assert isinstance(result, ar.ArFrame)
    
    def test_empty_steps_list(self, simple_frame):
        """Test pipeline with empty steps list"""
        result = ar.pipeline(simple_frame, [])
        assert isinstance(result, ar.ArFrame)
        assert result.shape == simple_frame.shape
    
    def test_unknown_step_raises_error(self, simple_frame):
        """Test that unknown step names raise UnknownStepError before execution"""
        with pytest.raises(ar.UnknownStepError):
            ar.pipeline(simple_frame, [("nonexistent_step",)])
    
    def test_malformed_step_tuple_raises_early(self, simple_frame):
        """Test that malformed step tuples fail early before execution"""
        with pytest.raises((ValueError, TypeError, ar.UnknownStepError)):
            ar.pipeline(simple_frame, [("strip_whitespace", "oops", "extra")])
    
    def test_invalid_kwargs_type(self, simple_frame):
        """Test that non-dict kwargs raises error"""
        with pytest.raises((ValueError, TypeError)):
            ar.pipeline(simple_frame, [("strip_whitespace", "not_a_dict")])
    
    def test_dry_run_validates_without_execution(self, simple_frame):
        """Test dry_run=True validates pipeline without returning transformed output"""
        result = ar.pipeline(
            simple_frame,
            [("strip_whitespace",), ("drop_duplicates",)],
            dry_run=True
        )
        assert result is None
    
    def test_dry_run_validates_step_names(self, simple_frame):
        """Test dry_run=True still validates unknown step names"""
        with pytest.raises(ar.UnknownStepError):
            ar.pipeline(
                simple_frame,
                [("strip_whitespace",), ("bad_step",)],
                dry_run=True
            )
    
    def test_return_metadata_provides_dict(self, dirty_frame):
        """Test return_metadata=True returns (frame, metadata) tuple"""
        result, metadata = ar.pipeline(
            dirty_frame,
            [("strip_whitespace",), ("drop_duplicates",)],
            return_metadata=True
        )
        
        assert isinstance(result, ar.ArFrame)
        assert isinstance(metadata, dict)
        assert "step_timings" in metadata
        assert "applied_steps" in metadata
        assert "row_counts" in metadata
        assert "execution_summary" in metadata
        assert isinstance(metadata["step_timings"], (list, dict))
    
    def test_verbose_flag_does_not_break(self, simple_frame):
        """Test verbose=True enables logging without breaking pipeline"""
        result = ar.pipeline(
            simple_frame,
            [("strip_whitespace",)],
            verbose=True
        )
        assert isinstance(result, ar.ArFrame)
    
    def test_drop_nulls_reduces_rows(self, frame_with_nulls):
        """Test drop_nulls step reduces row count with nulls present"""
        original_rows = frame_with_nulls.shape[0]
        result = ar.pipeline(frame_with_nulls, [("drop_nulls",)])
        assert result.shape[0] < original_rows
        assert result.shape[1] == frame_with_nulls.shape[1]
    
    def test_drop_duplicates_removes_duplicates(self, dirty_frame):
        """Test drop_duplicates removes identical rows"""
        original_rows = dirty_frame.shape[0]
        result = ar.pipeline(dirty_frame, [
            ("strip_whitespace",),
            ("drop_duplicates",)
        ])
        assert result.shape[0] < original_rows
    
    def test_fill_nulls_preserves_row_count(self, frame_with_nulls):
        """Test fill_nulls preserves row count while filling missing values"""
        original_rows = frame_with_nulls.shape[0]
        result = ar.pipeline(
            frame_with_nulls,
            [("fill_nulls", {"value": "0", "subset": ["score"]})]
        )
        assert result.shape[0] == original_rows
    
    def test_chained_steps_return_arframe(self, simple_frame):
        """Test that all steps in chain preserve ArFrame type"""
        steps = [
            ("strip_whitespace",),
            ("normalize_case", {"case_type": "lower"}),
            ("drop_duplicates",),
        ]
        result = ar.pipeline(simple_frame, steps)
        assert isinstance(result, ar.ArFrame)
        assert result.shape[1] == simple_frame.shape[1]
    
    def test_pipeline_with_subset_kwarg(self, frame_with_nulls):
        """Test pipeline step with subset parameter"""
        result = ar.pipeline(
            frame_with_nulls,
            [("drop_nulls", {"subset": ["name"]})]
        )
        assert isinstance(result, ar.ArFrame)
        assert result.shape[0] < frame_with_nulls.shape[0]
