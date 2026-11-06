import pytest
import pandas as pd
import arnio as ar


class TestUnregisterStep:
    """Test suite for arnio.pipeline.unregister_step."""

    def teardown_method(self):
        """Clean up custom steps after each test."""
        ar.reset_steps()

    def test_unregister_custom_step_success(self):
        """Successfully unregister a custom step."""
        def my_step(df):
            return df
        
        ar.register_step("test_step", my_step)
        assert "test_step" in ar.list_steps()
        
        ar.unregister_step("test_step")
        assert "test_step" not in ar.list_steps()

    def test_unregister_name_not_string_integer(self):
        """TypeError raised when name is an integer."""
        with pytest.raises(TypeError, match="parameter 'name' must be a string"):
            ar.unregister_step(42)

    def test_unregister_name_not_string_none(self):
        """TypeError raised when name is None."""
        with pytest.raises(TypeError, match="parameter 'name' must be a string"):
            ar.unregister_step(None)

    def test_unregister_name_not_string_list(self):
        """TypeError raised when name is a list."""
        with pytest.raises(TypeError, match="parameter 'name' must be a string"):
            ar.unregister_step(["my_step"])

    def test_unregister_empty_string_name(self):
        """ValueError raised when name is empty string."""
        with pytest.raises(ValueError, match="parameter 'name' must be a non-empty string"):
            ar.unregister_step("")

    def test_unregister_unknown_step(self):
        """UnknownStepError raised when step not in registry."""
        with pytest.raises(ar.UnknownStepError):
            ar.unregister_step("does_not_exist_xyz")

    def test_unregister_builtin_step_protection(self):
        """ValueError raised when trying to unregister a built-in step."""
        with pytest.raises(ValueError, match="Cannot unregister built-in step"):
            ar.unregister_step("drop_nulls")

    def test_unregister_step_removed_from_list(self):
        """Unregistered step is removed from ar.list_steps()."""
        def step(df):
            return df
        
        ar.register_step("removable", step)
        assert "removable" in ar.list_steps()
        
        ar.unregister_step("removable")
        assert "removable" not in ar.list_steps()

    def test_unregister_step_cannot_be_used_in_pipeline(self):
        """Step cannot be used in pipeline after unregistration."""
        def step(df):
            df = df.copy()
            df["col"] = 1
            return df
        
        ar.register_step("pipeline_step", step)
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        
        ar.unregister_step("pipeline_step")
        
        with pytest.raises(ar.UnknownStepError):
            ar.pipeline(frame, [("pipeline_step",)])

    def test_unregister_twice_fails(self):
        """UnknownStepError when unregistering already-unregistered step."""
        def step(df):
            return df
        
        ar.register_step("once_only", step)
        ar.unregister_step("once_only")
        
        with pytest.raises(ar.UnknownStepError):
            ar.unregister_step("once_only")

    def test_unregister_case_sensitive(self):
        """Step names are case-sensitive for unregistration."""
        def step(df):
            return df
        
        ar.register_step("CaseSensitive", step)
        
        with pytest.raises(ar.UnknownStepError):
            ar.unregister_step("casesensitive")
        
        ar.unregister_step("CaseSensitive")

    def test_unregister_does_not_affect_other_steps(self):
        """Unregistering one step doesn't affect other registered steps."""
        def step_a(df):
            return df
        
        def step_b(df):
            return df
        
        ar.register_step("step_a", step_a)
        ar.register_step("step_b", step_b)
        
        ar.unregister_step("step_a")
        
        assert "step_b" in ar.list_steps()
        
        # step_b should still be usable
        frame = ar.from_pandas(pd.DataFrame({"x": [1]}))
        ar.pipeline(frame, [("step_b",)])  # Should not raise

    def test_unregister_and_reregister(self):
        """Step can be re-registered after unregistration."""
        def step_v1(df):
            df = df.copy()
            df["version"] = 1
            return df
        
        ar.register_step("reusable", step_v1)
        ar.unregister_step("reusable")
        
        def step_v2(df):
            df = df.copy()
            df["version"] = 2
            return df
        
        ar.register_step("reusable", step_v2, overwrite=True)
        
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        result = ar.pipeline(frame, [("reusable",)])
        assert ar.to_pandas(result)["version"].iloc[0] == 2

    def test_unregister_step_with_namespace_prefix(self):
        """Can register and unregister steps with namespace prefixes."""
        def step(df):
            return df
        
        ar.register_step("custom:my_step", step)
        assert "custom:my_step" in ar.list_steps()
        
        ar.unregister_step("custom:my_step")
        assert "custom:my_step" not in ar.list_steps()

    def test_unregister_multiple_custom_steps_independently(self):
        """Can unregister multiple custom steps independently."""
        def step_a(df):
            return df
        
        def step_b(df):
            return df
        
        def step_c(df):
            return df
        
        ar.register_step("step_a", step_a)
        ar.register_step("step_b", step_b)
        ar.register_step("step_c", step_c)
        
        ar.unregister_step("step_b")
        
        assert "step_a" in ar.list_steps()
        assert "step_b" not in ar.list_steps()
        assert "step_c" in ar.list_steps()
