import pytest
import pandas as pd
import arnio as ar
from arnio.pipeline import register_step, unregister_step
from arnio.exceptions import UnknownStepError


def custom_drop_na(frame):
    """Custom cleaning step that drops NA values."""
    return frame.dropna()


def custom_fill_zero(frame):
    """Custom cleaning step that fills NA with 0."""
    return frame.fillna(0)


class TestUnregisterStepBasic:
    """Test basic unregister_step functionality."""

    def test_register_and_unregister_custom_step(self):
        """Test successfully registering and unregistering a custom step."""
        step_name = "test_custom_drop_na"
        register_step(step_name, custom_drop_na)
        unregister_step(step_name)

        # Verify step is no longer available
        with pytest.raises(UnknownStepError):
            unregister_step(step_name)

    def test_unregister_then_register_same_name(self):
        """Test that a step name can be reused after unregistering."""
        step_name = "reusable_step"
        register_step(step_name, custom_drop_na)
        unregister_step(step_name)

        # Re-register with different function
        register_step(step_name, custom_fill_zero)
        unregister_step(step_name)


class TestUnregisterStepParameterValidation:
    """Test parameter validation for unregister_step."""

    def test_unregister_non_string_name_raises_type_error(self):
        """Test that non-string name parameter raises TypeError."""
        with pytest.raises(TypeError, match="parameter 'name' must be a string"):
            unregister_step(123)

        with pytest.raises(TypeError, match="parameter 'name' must be a string"):
            unregister_step(None)

        with pytest.raises(TypeError, match="parameter 'name' must be a string"):
            unregister_step(['not', 'string'])

    def test_unregister_empty_string_raises_value_error(self):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError, match="non-empty string"):
            unregister_step("")

    def test_unregister_nonexistent_step_raises_unknown_step_error(self):
        """Test that unregistering non-existent step raises UnknownStepError."""
        with pytest.raises(UnknownStepError):
            unregister_step("nonexistent_step_xyz_123_does_not_exist")


class TestUnregisterStepBuiltinProtection:
    """Test that built-in steps cannot be unregistered."""

    def test_cannot_unregister_builtin_drop_na(self):
        """Test that built-in 'drop_na' step cannot be unregistered."""
        with pytest.raises(ValueError, match="Cannot unregister built-in step"):
            unregister_step("drop_na")

    def test_cannot_unregister_builtin_fill_na(self):
        """Test that built-in 'fill_na' step cannot be unregistered."""
        with pytest.raises(ValueError, match="Cannot unregister built-in step"):
            unregister_step("fill_na")

    def test_builtin_protection_error_message_content(self):
        """Test that built-in protection error message is informative."""
        with pytest.raises(ValueError) as exc_info:
            unregister_step("drop_na")

        error_msg = str(exc_info.value)
        assert "built-in" in error_msg.lower()
        assert "custom" in error_msg.lower()


class TestUnregisterStepWithPipelines:
    """Test unregister_step with real pipeline workflows."""

    def test_unregistered_step_not_usable_in_pipeline(self, tmp_path):
        """Test that after unregistering, step cannot be used in pipelines."""
        csv_path = tmp_path / "test_data.csv"
        df = pd.DataFrame({
            'A': [1, 2, None, 4, 5],
            'B': [10, None, 30, 40, 50]
        })
        df.to_csv(csv_path, index=False)

        step_name = "pipeline_test_step"
        register_step(step_name, custom_drop_na)

        # Verify pipeline works before unregister
        pipeline = ar.Pipeline([(step_name, {})])
        result1 = ar.load(str(csv_path)).transform(pipeline)
        assert len(result1) == 3  # Only rows without NaN

        # Unregister the step
        unregister_step(step_name)

        # Pipeline should fail after unregister
        with pytest.raises(UnknownStepError):
            ar.load(str(csv_path)).transform(ar.Pipeline([(step_name, {})]))

    def test_other_custom_steps_work_after_unregister(self, tmp_path):
        """Test that other steps still work after one is unregistered."""
        csv_path = tmp_path / "data.csv"
        df = pd.DataFrame({
            'value': [1, None, 3],
            'group': ['A', 'B', 'C']
        })
        df.to_csv(csv_path, index=False)

        step1 = "step_to_remove"
        step2 = "step_to_keep"

        register_step(step1, custom_drop_na)
        register_step(step2, custom_fill_zero)

        # Unregister first step
        unregister_step(step1)

        # Second step should still work
        pipeline = ar.Pipeline([(step2, {})])
        result = ar.load(str(csv_path)).transform(pipeline)
        assert result.isna().sum().sum() == 0  # Filled with 0

    def test_pipeline_serialization_with_registered_step(self, tmp_path):
        """Test that pipeline works correctly before and after unregister."""
        csv_path = tmp_path / "source.csv"
        df = pd.DataFrame({'col1': [1, None, 3], 'col2': [4, 5, None]})
        df.to_csv(csv_path, index=False)

        step_name = "serial_test_step"
        register_step(step_name, custom_drop_na)

        frame1 = ar.load(str(csv_path))
        pipeline = ar.Pipeline([(step_name, {})])
        result1 = frame1.transform(pipeline)

        assert len(result1) == 1  # Only one row with no NaN

        unregister_step(step_name)

        frame2 = ar.load(str(csv_path))
        with pytest.raises(UnknownStepError):
            frame2.transform(pipeline)


class TestUnregisterStepEndToEnd:
    """End-to-end workflow tests."""

    def test_complete_workflow_register_use_unregister_reregister(self, tmp_path):
        """Complete workflow: register, use, unregister, re-register."""
        csv_path = tmp_path / "workflow.csv"
        df = pd.DataFrame({'x': [1, None, 3, 4], 'y': [5, 6, None, 8]})
        df.to_csv(csv_path, index=False)

        step_name = "workflow_step"

        # Phase 1: Register and use
        register_step(step_name, custom_drop_na)
        result1 = ar.load(str(csv_path)).transform(ar.Pipeline([(step_name, {})]))
        assert len(result1) == 2

        # Phase 2: Unregister
        unregister_step(step_name)
        with pytest.raises(UnknownStepError):
            ar.load(str(csv_path)).transform(ar.Pipeline([(step_name, {})]))

        # Phase 3: Re-register with different function
        register_step(step_name, custom_fill_zero)
        result2 = ar.load(str(csv_path)).transform(ar.Pipeline([(step_name, {})]))
        assert result2.isna().sum().sum() == 0

        unregister_step(step_name)

    def test_case_sensitive_step_names(self):
        """Test that step names are case-sensitive."""
        lower_name = "mystep"
        upper_name = "MYSTEP"

        register_step(lower_name, custom_drop_na)

        # Different case should not be found
        with pytest.raises(UnknownStepError):
            unregister_step(upper_name)

        # Correct case should work
        unregister_step(lower_name)

    def test_step_name_with_special_characters(self):
        """Test step names with special characters."""
        step_name = "my-step_v2.0"

        register_step(step_name, custom_drop_na)
        unregister_step(step_name)

        # Should not exist after unregister
        with pytest.raises(UnknownStepError):
            unregister_step(step_name)

    def test_multiple_custom_steps_lifecycle(self, tmp_path):
        """Test managing multiple custom steps through their lifecycle."""
        csv_path = tmp_path / "multi.csv"
        df = pd.DataFrame({
            'a': [1, None, 3],
            'b': [None, 5, 6],
            'c': [7, 8, None]
        })
        df.to_csv(csv_path, index=False)

        steps = ["step_a", "step_b", "step_c"]

        # Register all steps
        for step in steps:
            register_step(step, custom_drop_na)

        # Use pipeline with all steps
        pipeline = ar.Pipeline([(s, {}) for s in steps])
        result1 = ar.load(str(csv_path)).transform(pipeline)
        assert len(result1) == 0  # All rows have NaN

        # Unregister first and third steps
        unregister_step(steps[0])
        unregister_step(steps[2])

        # Middle step should still work
        pipeline2 = ar.Pipeline([(steps[1], {})])
        result2 = ar.load(str(csv_path)).transform(pipeline2)
        assert len(result2) == 0

        # Unregister remaining step
        unregister_step(steps[1])

        with pytest.raises(UnknownStepError):
            ar.load(str(csv_path)).transform(ar.Pipeline([(steps[1], {})]))


class TestUnregisterStepErrorMessages:
    """Test error message content and informativeness."""

    def test_unknown_step_error_message(self):
        """Test that UnknownStepError provides useful information."""
        with pytest.raises(UnknownStepError) as exc_info:
            unregister_step("not_a_real_step_xyz")

        error_msg = str(exc_info.value)
        # Error should indicate the step name
        assert "not_a_real_step_xyz" in error_msg or "Unknown" in error_msg

    def test_type_error_message_clarity(self):
        """Test that TypeError message clearly indicates the problem."""
        with pytest.raises(TypeError) as exc_info:
            unregister_step(42)

        error_msg = str(exc_info.value)
        assert "name" in error_msg
        assert "string" in error_msg
