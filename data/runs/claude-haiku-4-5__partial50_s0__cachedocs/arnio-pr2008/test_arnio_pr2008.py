import pytest
import pandas as pd
import arnio as ar


class TestUnregisterStep:
    """Tests for arnio.pipeline.unregister_step function."""

    def setup_method(self):
        """Reset the step registry before each test."""
        ar.reset_steps()

    def teardown_method(self):
        """Clean up after each test."""
        ar.reset_steps()

    def test_unregister_custom_step_removes_from_registry(self):
        """Unregistering a custom step removes it from list_steps()."""
        def custom_func(df):
            return df

        ar.register_step("my_custom_step", custom_func)
        assert "my_custom_step" in ar.list_steps()

        ar.unregister_step("my_custom_step")
        assert "my_custom_step" not in ar.list_steps()

    def test_unregister_step_makes_pipeline_fail(self):
        """After unregistering, the step cannot be used in a pipeline."""
        def add_column(df, value=0):
            df = df.copy()
            df["new_col"] = value
            return df

        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
        ar.register_step("add_column", add_column)

        # Verify it works before unregistering
        result = ar.to_pandas(ar.pipeline(frame, [("add_column", {"value": 5})]))
        assert "new_col" in result.columns

        # Unregister and verify pipeline fails
        ar.unregister_step("add_column")
        with pytest.raises(ar.UnknownStepError):
            ar.pipeline(frame, [("add_column", {"value": 5})])

    def test_unregister_nonexistent_custom_step_raises_error(self):
        """Unregistering a non-existent custom step raises an error."""
        with pytest.raises((ValueError, KeyError, ar.UnknownStepError)):
            ar.unregister_step("does_not_exist")

    def test_unregister_builtin_step_raises_error(self):
        """Built-in steps cannot be unregistered."""
        # Verify a built-in exists
        assert "drop_nulls" in ar.list_steps()

        # Attempting to unregister a built-in should raise an error
        with pytest.raises((ValueError, ar.UnknownStepError)):
            ar.unregister_step("drop_nulls")

        # Verify the built-in is still available
        assert "drop_nulls" in ar.list_steps()

    def test_unregister_namespaced_custom_step(self):
        """Unregistering a namespaced custom step works correctly."""
        def team_step(df):
            return df

        ar.register_step("team:custom_step", team_step)
        assert "team:custom_step" in ar.list_steps()

        ar.unregister_step("team:custom_step")
        assert "team:custom_step" not in ar.list_steps()

    def test_unregister_then_reregister_same_name(self):
        """After unregistering, a step name can be re-registered."""
        def first_impl(df):
            df = df.copy()
            df["version"] = 1
            return df

        def second_impl(df):
            df = df.copy()
            df["version"] = 2
            return df

        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))

        ar.register_step("versioned_step", first_impl)
        result1 = ar.to_pandas(ar.pipeline(frame, [("versioned_step",)]))
        assert result1["version"].iloc[0] == 1

        ar.unregister_step("versioned_step")

        # Re-register with different implementation
        ar.register_step("versioned_step", second_impl)
        result2 = ar.to_pandas(ar.pipeline(frame, [("versioned_step",)]))
        assert result2["version"].iloc[0] == 2

    def test_unregister_multiple_independent_steps(self):
        """Multiple custom steps can be unregistered independently."""
        def step1(df):
            return df

        def step2(df):
            return df

        def step3(df):
            return df

        ar.register_step("step1", step1)
        ar.register_step("step2", step2)
        ar.register_step("step3", step3)

        ar.unregister_step("step2")

        assert "step1" in ar.list_steps()
        assert "step2" not in ar.list_steps()
        assert "step3" in ar.list_steps()

    def test_unregister_preserves_other_custom_steps_and_builtins(self):
        """Unregistering one custom step doesn't affect others or built-ins."""
        def custom1(df):
            return df

        def custom2(df):
            return df

        builtins_before = {s for s in ar.list_steps() if not s.startswith("custom")}
        ar.register_step("custom1", custom1)
        ar.register_step("custom2", custom2)

        ar.unregister_step("custom1")

        assert "custom1" not in ar.list_steps()
        assert "custom2" in ar.list_steps()
        # Built-ins should still be present
        assert all(b in ar.list_steps() for b in builtins_before)

    def test_unregister_twice_raises_error(self):
        """Unregistering the same step twice raises an error."""
        def my_step(df):
            return df

        ar.register_step("my_step", my_step)
        ar.unregister_step("my_step")

        with pytest.raises((ValueError, KeyError, ar.UnknownStepError)):
            ar.unregister_step("my_step")

    def test_unregister_case_sensitive(self):
        """Step names are case-sensitive for unregistering."""
        def my_func(df):
            return df

        ar.register_step("MyStep", my_func)

        # Wrong case should fail
        with pytest.raises((ValueError, KeyError, ar.UnknownStepError)):
            ar.unregister_step("mystep")

        # Correct case should succeed
        ar.unregister_step("MyStep")
        assert "MyStep" not in ar.list_steps()

    def test_unregister_step_with_kwargs_param(self):
        """Unregistering a step that uses kwargs works correctly."""
        def flexible_step(df, **kwargs):
            return df

        ar.register_step("flexible", flexible_step)
        assert "flexible" in ar.list_steps()

        ar.unregister_step("flexible")
        assert "flexible" not in ar.list_steps()

    def test_unregister_step_with_context_param(self):
        """Unregistering a step with context parameter works correctly."""
        def context_aware(df, context=None):
            return df

        ar.register_step("context_aware", context_aware)
        ar.unregister_step("context_aware")

        assert "context_aware" not in ar.list_steps()

    def test_unregister_custom_step_complete_workflow(self):
        """Complete workflow: register, use, unregister, verify unavailable."""
        def multiply_values(df, factor=2):
            df = df.copy()
            df["a"] = df["a"] * factor
            return df

        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))

        # Register and use
        ar.register_step("multiply", multiply_values)
        result_before = ar.to_pandas(ar.pipeline(frame, [("multiply", {"factor": 3})]))
        assert result_before["a"].tolist() == [3, 6, 9]

        # Unregister
        ar.unregister_step("multiply")

        # Verify unavailable
        assert "multiply" not in ar.list_steps()
        with pytest.raises(ar.UnknownStepError):
            ar.pipeline(frame, [("multiply", {"factor": 3})])

    def test_unregister_empty_string_raises_error(self):
        """Attempting to unregister empty string raises an error."""
        with pytest.raises((ValueError, KeyError, ar.UnknownStepError)):
            ar.unregister_step("")
