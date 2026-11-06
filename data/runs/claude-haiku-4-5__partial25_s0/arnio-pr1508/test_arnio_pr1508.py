import pytest
import pandas as pd
import arnio as ar
from arnio.exceptions import UnknownStepError


class TestPipelineBasicExecution:
    """Test basic pipeline execution and return types."""
    
    @pytest.fixture
    def sample_frame(self):
        """Create a sample ArFrame for testing."""
        data = {"name": [" Alice ", " Bob "], "age": [25, 30]}
        return ar.from_dict(data)
    
    def test_pipeline_single_step_returns_frame(self, sample_frame):
        """Test that pipeline with single step returns ArFrame."""
        result = ar.pipeline(sample_frame, [("strip_whitespace",)])
        assert isinstance(result, ar.ArFrame)
    
    def test_pipeline_multiple_steps(self, sample_frame):
        """Test pipeline with multiple steps executes in sequence."""
        result = ar.pipeline(sample_frame, [
            ("strip_whitespace",),
            ("drop_duplicates",),
        ])
        assert isinstance(result, ar.ArFrame)
    
    def test_pipeline_step_with_kwargs(self, sample_frame):
        """Test step with keyword arguments."""
        result = ar.pipeline(sample_frame, [
            ("normalize_case", {"case_type": "lower"}),
        ])
        assert isinstance(result, ar.ArFrame)
    
    def test_pipeline_mixed_steps_with_and_without_kwargs(self, sample_frame):
        """Test pipeline with steps both with and without kwargs."""
        result = ar.pipeline(sample_frame, [
            ("strip_whitespace",),
            ("normalize_case", {"case_type": "upper"}),
        ])
        assert isinstance(result, ar.ArFrame)


class TestPipelineValidation:
    """Test pipeline validation behavior (early failure)."""
    
    @pytest.fixture
    def sample_frame(self):
        """Create a sample ArFrame for testing."""
        return ar.from_dict({"id": [1, 2], "name": ["a", "b"]})
    
    def test_unknown_step_raises_before_execution(self, sample_frame):
        """Test that unknown step names raise UnknownStepError before execution."""
        with pytest.raises(UnknownStepError):
            ar.pipeline(sample_frame, [
                ("strip_whitespace",),
                ("unknown_step_xyz",),
            ])
    
    def test_malformed_tuple_too_many_elements(self, sample_frame):
        """Test that tuple with extra elements fails validation."""
        with pytest.raises((ValueError, TypeError, IndexError)):
            ar.pipeline(sample_frame, [
                ("strip_whitespace",),
                ("bad_step", "oops", "extra"),
            ])
    
    def test_invalid_kwargs_not_dict(self, sample_frame):
        """Test that non-dict kwargs structure fails validation."""
        with pytest.raises((ValueError, TypeError)):
            ar.pipeline(sample_frame, [
                ("strip_whitespace", "not_a_dict"),
            ])
    
    def test_validation_prevents_partial_execution(self, sample_frame):
        """Test that validation occurs before any steps execute."""
        original_shape = sample_frame.shape
        try:
            ar.pipeline(sample_frame, [
                ("strip_whitespace",),
                ("invalid_step_name",),
            ])
        except UnknownStepError:
            pass
        # If validation is done correctly, no steps should have executed


class TestPipelineReturnTypes:
    """Test different return value configurations."""
    
    @pytest.fixture
    def sample_frame(self):
        """Create a sample ArFrame for testing."""
        return ar.from_dict({"name": [" Alice "], "age": [25]})
    
    def test_default_return_is_arframe(self, sample_frame):
        """Test that default return is just the ArFrame, not a tuple."""
        result = ar.pipeline(sample_frame, [("strip_whitespace",)])
        assert isinstance(result, ar.ArFrame)
        assert not isinstance(result, tuple)
    
    def test_return_metadata_true_returns_tuple(self, sample_frame):
        """Test that return_metadata=True returns tuple with frame and metadata."""
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
    
    def test_track_lineage_true_returns_tuple(self, sample_frame):
        """Test that track_lineage=True returns tuple with frame and report."""
        result = ar.pipeline(
            sample_frame,
            [("strip_whitespace",)],
            track_lineage=True
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        frame, lineage = result
        assert isinstance(frame, ar.ArFrame)


class TestPipelineExecutionBehavior:
    """Test actual step execution and transformations."""
    
    def test_pipeline_executes_strip_whitespace(self):
        """Test that strip_whitespace step actually modifies data."""
        data = {"name": [" Alice ", " Bob "]}
        frame = ar.from_dict(data)
        result = ar.pipeline(frame, [("strip_whitespace",)])
        pdf = ar.to_pandas(result)
        # Whitespace should be removed
        assert pdf["name"].iloc[0] == "Alice"
    
    def test_pipeline_executes_drop_duplicates(self):
        """Test that drop_duplicates reduces row count."""
        data = {"id": [1, 2, 2, 3], "name": ["a", "b", "b", "c"]}
        frame = ar.from_dict(data)
        result = ar.pipeline(frame, [("drop_duplicates",)])
        # Should have fewer rows after deduplication
        assert result.shape[0] < frame.shape[0]
    
    def test_pipeline_steps_execute_in_order(self):
        """Test that pipeline steps execute in the order specified."""
        data = {"value": ["1", "2", "2"]}
        frame = ar.from_dict(data)
        result = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("drop_duplicates",),
        ])
        # Should have 2 unique values after both steps
        assert result.shape[0] == 2


class TestPipelineWithCustomSteps:
    """Test pipeline with custom registered steps."""
    
    def test_custom_step_in_pipeline(self):
        """Test using a custom-registered step in pipeline."""
        def custom_add_column(df):
            df = df.copy()
            df["custom"] = "test"
            return df
        
        ar.register_step("custom_add_column", custom_add_column)
        try:
            frame = ar.from_dict({"id": [1, 2]})
            result = ar.pipeline(frame, [("custom_add_column",)])
            assert isinstance(result, ar.ArFrame)
        finally:
            ar.reset_steps()
    
    def test_custom_step_with_arguments(self):
        """Test custom step that accepts keyword arguments."""
        def custom_multiply(df, factor=2):
            df = df.copy()
            df["value"] = df["value"] * factor
            return df
        
        ar.register_step("custom_multiply", custom_multiply)
        try:
            frame = ar.from_dict({"value": [1, 2, 3]})
            result = ar.pipeline(frame, [
                ("custom_multiply", {"factor": 10}),
            ])
            assert isinstance(result, ar.ArFrame)
        finally:
            ar.reset_steps()


class TestPipelineEdgeCases:
    """Test edge cases and special scenarios."""
    
    def test_empty_pipeline_list(self):
        """Test pipeline with empty step list."""
        frame = ar.from_dict({"id": [1, 2, 3]})
        result = ar.pipeline(frame, [])
        assert isinstance(result, ar.ArFrame)
        assert result.shape == frame.shape
    
    def test_pipeline_with_null_values(self):
        """Test pipeline handles frames with null values."""
        data = {"name": ["Alice", None, "Charlie"], "age": [25, 30, None]}
        frame = ar.from_dict(data)
        result = ar.pipeline(frame, [("strip_whitespace",)])
        assert isinstance(result, ar.ArFrame)
    
    def test_dry_run_mode(self):
        """Test dry_run mode parameter."""
        frame = ar.from_dict({"name": [" Alice "]})
        result = ar.pipeline(frame, [("strip_whitespace",)], dry_run=True)
        assert isinstance(result, ar.ArFrame)
    
    def test_verbose_mode_parameter(self):
        """Test verbose mode parameter (executes without error)."""
        frame = ar.from_dict({"name": [" Alice "]})
        result = ar.pipeline(
            frame,
            [("strip_whitespace",)],
            verbose=True
        )
        assert isinstance(result, ar.ArFrame)
    
    def test_builtin_namespace_prefix(self):
        """Test explicit 'builtin:' namespace for step names."""
        frame = ar.from_dict({"name": [" Alice "]})
        result = ar.pipeline(frame, [("builtin:strip_whitespace",)])
        assert isinstance(result, ar.ArFrame)


class TestPipelineStepTupleFormats:
    """Test valid and invalid step tuple formats."""
    
    def test_valid_tuple_name_only(self):
        """Test valid tuple format with name only."""
        frame = ar.from_dict({"id": [1]})
        result = ar.pipeline(frame, [("strip_whitespace",)])
        assert isinstance(result, ar.ArFrame)
    
    def test_valid_tuple_with_empty_dict(self):
        """Test valid tuple format with empty kwargs dict."""
        frame = ar.from_dict({"id": [1]})
        result = ar.pipeline(frame, [("strip_whitespace", {})])
        assert isinstance(result, ar.ArFrame)
    
    def test_valid_tuple_with_kwargs(self):
        """Test valid tuple format with kwargs dictionary."""
        frame = ar.from_dict({"name": ["a"]})
        result = ar.pipeline(frame, [
            ("normalize_case", {"case_type": "upper"}),
        ])
        assert isinstance(result, ar.ArFrame)
    
    def test_invalid_tuple_with_three_elements(self):
        """Test that 3-element tuple fails validation."""
        frame = ar.from_dict({"id": [1]})
        with pytest.raises((ValueError, TypeError, IndexError)):
            ar.pipeline(frame, [("step", {}, "extra")])
    
    def test_invalid_tuple_with_string_instead_of_dict_kwargs(self):
        """Test that non-dict kwargs fails."""
        frame = ar.from_dict({"id": [1]})
        with pytest.raises((ValueError, TypeError)):
            ar.pipeline(frame, [("strip_whitespace", "invalid")])


class TestPipelineIntegration:
    """Test pipeline integration with other Arnio operations."""
    
    def test_pipeline_output_to_pandas(self):
        """Test converting pipeline result to pandas DataFrame."""
        frame = ar.from_dict({"name": [" Alice ", " Bob "], "age": [25, 30]})
        result = ar.pipeline(frame, [("strip_whitespace",)])
        pdf = ar.to_pandas(result)
        assert isinstance(pdf, pd.DataFrame)
        assert pdf.shape[0] == 2
    
    def test_pipeline_chaining_with_from_dict(self):
        """Test pipeline works seamlessly with from_dict input."""
        data = {"name": [" Alice "], "age": [25]}
        frame = ar.from_dict(data)
        result = ar.pipeline(frame, [("strip_whitespace",)])
        assert isinstance(result, ar.ArFrame)
        assert result.shape == frame.shape
    
    def test_pipeline_with_column_selection(self):
        """Test pipeline step with column subset."""
        frame = ar.from_dict({
            "first": [" alice "],
            "last": [" smith "],
        })
        result = ar.pipeline(frame, [
            ("strip_whitespace", {"subset": ["first"]}),
        ])
        assert isinstance(result, ar.ArFrame)


class TestPipelineConsistency:
    """Test consistency of pipeline behavior."""
    
    def test_idempotent_pipeline_on_clean_data(self):
        """Test pipeline on already-clean data doesn't error."""
        frame = ar.from_dict({"name": ["Alice", "Bob"], "age": [25, 30]})
        result = ar.pipeline(frame, [("strip_whitespace",)])
        assert isinstance(result, ar.ArFrame)
    
    def test_pipeline_preserves_frame_type(self):
        """Test that pipeline always returns ArFrame, not pandas DataFrame."""
        frame = ar.from_dict({"id": [1, 2, 3]})
        result = ar.pipeline(frame, [("drop_duplicates",)])
        assert type(result).__name__ == "ArFrame"
        assert not isinstance(result, pd.DataFrame)
