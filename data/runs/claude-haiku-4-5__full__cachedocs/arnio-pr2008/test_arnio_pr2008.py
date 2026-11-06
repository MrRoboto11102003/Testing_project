import pytest
import pandas as pd
import arnio as ar
from arnio.exceptions import UnknownStepError


class TestUnregisterStep:
    """Test suite for ar.unregister_step()"""

    def setup_method(self):
        """Reset custom steps before each test to ensure clean state"""
        ar.reset_steps()

    def test_unregister_custom_step_removes_from_registry(self):
        """Test that unregister_step successfully removes a custom step"""
        def my_step(df):
            return df.copy()
        
        ar.register_step("test_step", my_step)
        ar.unregister_step("test_step")
        
        # Step should no longer be available
        with pytest.raises(UnknownStepError):
            ar.pipeline(
                ar.from_pandas(pd.DataFrame({"a": [1]})),
                [("test_step",)]
            )

    def test_unregister_nonexistent_step_raises_error(self):
        """Test that unregistering a non-existent step raises an error"""
        with pytest.raises((ValueError, KeyError, UnknownStepError)):
            ar.unregister_step("does_not_exist")

    def test_unregister_builtin_step_raises_error(self):
        """Test that unregistering a built-in step is not allowed"""
        with pytest.raises((ValueError, UnknownStepError)):
            ar.unregister_step("drop_nulls")

    def test_unregister_twice_raises_error(self):
        """Test that unregistering the same step twice raises an error"""
        def my_step(df):
            return df.copy()
        
        ar.register_step("temp_step", my_step)
        ar.unregister_step("temp_step")
        
        # Second unregister should fail
        with pytest.raises((ValueError, KeyError, UnknownStepError)):
            ar.unregister_step("temp_step")

    def test_unregister_then_reregister_succeeds(self):
        """Test that a step can be re-registered after unregistration"""
        def step_v1(df):
            return df.copy()
        
        def step_v2(df):
            df = df.copy()
            df["new_col"] = 999
            return df
        
        ar.register_step("reusable_step", step_v1)
        ar.unregister_step("reusable_step")
        ar.register_step("reusable_step", step_v2)
        
        # New version should work
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
        result = ar.pipeline(frame, [("reusable_step",)])
        df = ar.to_pandas(result)
        assert "new_col" in df.columns
        assert list(df["new_col"]) == [999, 999, 999]

    def test_unregister_namespaced_step(self):
        """Test unregistering a namespaced custom step"""
        def my_step(df):
            return df.copy()
        
        ar.register_step("myapp:clean_data", my_step)
        ar.unregister_step("myapp:clean_data")
        
        with pytest.raises(UnknownStepError):
            ar.pipeline(
                ar.from_pandas(pd.DataFrame({"a": [1]})),
                [("myapp:clean_data",)]
            )

    def test_unregister_case_sensitive(self):
        """Test that step name unregistration is case-sensitive"""
        def my_step(df):
            return df.copy()
        
        ar.register_step("CaseSensitive", my_step)
        
        # Wrong case should fail to unregister
        with pytest.raises((ValueError, KeyError, UnknownStepError)):
            ar.unregister_step("casesensitive")
        
        # Correct case should succeed
        ar.unregister_step("CaseSensitive")

    def test_unregister_does_not_affect_other_steps(self):
        """Test that unregistering one step doesn't affect others"""
        def step_a(df):
            return df.copy()
        
        def step_b(df):
            df = df.copy()
            df["col"] = df["col"] * 2
            return df
        
        ar.register_step("step_a", step_a)
        ar.register_step("step_b", step_b)
        ar.unregister_step("step_a")
        
        # step_b should still be available and functional
        frame = ar.from_pandas(pd.DataFrame({"col": [1, 2, 3]}))
        result = ar.pipeline(frame, [("step_b",)])
        df = ar.to_pandas(result)
        assert list(df["col"]) == [2, 4, 6]

    def test_unregister_returns_none(self):
        """Test that unregister_step returns None"""
        def my_step(df):
            return df.copy()
        
        ar.register_step("my_step", my_step)
        result = ar.unregister_step("my_step")
        
        assert result is None

    def test_unregister_with_invalid_argument_types_raises_error(self):
        """Test that unregister_step rejects non-string step names"""
        # None
        with pytest.raises((TypeError, AttributeError)):
            ar.unregister_step(None)
        
        # Integer
        with pytest.raises((TypeError, AttributeError)):
            ar.unregister_step(123)
        
        # List
        with pytest.raises((TypeError, AttributeError)):
            ar.unregister_step(["my_step"])

    def test_unregister_empty_string_raises_error(self):
        """Test that unregister_step with empty string raises an error"""
        with pytest.raises((ValueError, KeyError, UnknownStepError)):
            ar.unregister_step("")

    def test_unregister_with_builtin_prefix_raises_error(self):
        """Test that builtin: prefix doesn't bypass unregister protection"""
        with pytest.raises((ValueError, UnknownStepError)):
            ar.unregister_step("builtin:drop_nulls")

    def test_unregister_clears_step_from_list_steps(self):
        """Test that unregistered step no longer appears in list_steps()"""
        def my_step(df):
            return df.copy()
        
        ar.register_step("list_test_step", my_step)
        assert "list_test_step" in ar.list_steps()
        
        ar.unregister_step("list_test_step")
        assert "list_test_step" not in ar.list_steps()

    def test_unregister_multiple_custom_steps_independently(self):
        """Test unregistering multiple custom steps in any order"""
        def step1(df):
            return df.copy()
        
        def step2(df):
            return df.copy()
        
        def step3(df):
            return df.copy()
        
        ar.register_step("step1", step1)
        ar.register_step("step2", step2)
        ar.register_step("step3", step3)
        
        # Unregister in different order
        ar.unregister_step("step2")
        ar.unregister_step("step1")
        
        # step3 should still work
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        ar.pipeline(frame, [("step3",)])  # Should not raise
        
        # step1 and step2 should not work
        with pytest.raises(UnknownStepError):
            ar.pipeline(frame, [("step1",)])
        
        with pytest.raises(UnknownStepError):
            ar.pipeline(frame, [("step2",)])
