import re
import pytest
import pandas as pd
import arnio as ar
from arnio.pipeline import register_step, unregister_step, list_steps, reset_steps, pipeline
from arnio.exceptions import UnknownStepError


@pytest.fixture(autouse=True)
def cleanup_registry():
    """Reset the registry after each test to ensure test isolation."""
    yield
    reset_steps()


class TestUnregisterStepBasic:
    """Test basic unregistration of custom steps."""

    def test_unregister_custom_step_succeeds(self):
        """Unregistering a registered custom step returns None."""
        register_step("zz_test_step", lambda df: df)
        assert "zz_test_step" in list_steps()

        result = unregister_step("zz_test_step")

        assert result is None
        assert "zz_test_step" not in list_steps()

    def test_unregister_custom_step_pipeline_fails_after(self):
        """After unregistration, pipeline raises UnknownStepError for that step."""
        register_step("zz_test_step", lambda df: df)
        unregister_step("zz_test_step")

        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))

        with pytest.raises(UnknownStepError):
            pipeline(frame, [("zz_test_step",)])

    def test_unregister_twice_raises_unknown_step_error(self):
        """Unregistering the same custom step twice raises UnknownStepError."""
        register_step("zz_test_step", lambda df: df)
        unregister_step("zz_test_step")

        with pytest.raises(UnknownStepError):
            unregister_step("zz_test_step")

    def test_unregister_and_reregister(self):
        """After unregistration, the same name can be registered again."""
        register_step("zz_test_step", lambda df: df)
        unregister_step("zz_test_step")

        register_step("zz_test_step", lambda df: df.copy())

        assert "zz_test_step" in list_steps()


class TestUnregisterStepTypeValidation:
    """Test type validation for the name parameter."""

    def test_unregister_name_not_string_int(self):
        """Passing an int raises TypeError with correct message."""
        with pytest.raises(TypeError, match=re.escape("parameter 'name' must be a string, not int")):
            unregister_step(123)

    def test_unregister_name_not_string_none(self):
        """Passing None raises TypeError with correct message."""
        with pytest.raises(TypeError, match=re.escape("parameter 'name' must be a string, not NoneType")):
            unregister_step(None)

    def test_unregister_name_not_string_bytes(self):
        """Passing bytes raises TypeError with correct message."""
        with pytest.raises(TypeError, match=re.escape("parameter 'name' must be a string, not bytes")):
            unregister_step(b"test")

    def test_unregister_name_not_string_list(self):
        """Passing a list raises TypeError with correct message."""
        with pytest.raises(TypeError, match=re.escape("parameter 'name' must be a string, not list")):
            unregister_step(["test"])


class TestUnregisterStepValueValidation:
    """Test value validation for the name parameter."""

    def test_unregister_empty_string(self):
        """Empty string raises ValueError."""
        with pytest.raises(ValueError, match=re.escape("parameter 'name' must be a non-empty string")):
            unregister_step("")

    def test_unregister_nonexistent_step(self):
        """Attempting to unregister a step that doesn't exist raises UnknownStepError."""
        with pytest.raises(UnknownStepError):
            unregister_step("does_not_exist_step_zzzz")

    def test_unregister_whitespace_only(self):
        """Whitespace-only name raises UnknownStepError (not ValueError)."""
        with pytest.raises(UnknownStepError):
            unregister_step("   ")


class TestUnregisterStepProtectedBuiltins:
    """Test that protected Python built-in steps cannot be unregistered."""

    @pytest.mark.parametrize("builtin_name", [
        "standardize_missing_tokens",
        "coalesce_columns",
        "normalize_whitespace",
        "filter_rows",
        "drop_columns_matching",
        "rename_columns_matching",
        "safe_divide_columns",
        "replace_values",
    ])
    def test_unregister_protected_builtin_raises_value_error(self, builtin_name):
        """Attempting to unregister a protected built-in raises ValueError."""
        expected_msg = f"Cannot unregister built-in step '{builtin_name}'. Only custom user-registered steps can be unregistered."
        with pytest.raises(ValueError, match=re.escape(expected_msg)):
            unregister_step(builtin_name)

    def test_protected_builtin_still_in_list_after_failed_unregister(self):
        """Protected built-in remains in list_steps after failed unregister attempt."""
        assert "filter_rows" in list_steps()

        with pytest.raises(ValueError):
            unregister_step("filter_rows")

        assert "filter_rows" in list_steps()


class TestUnregisterStepCppSteps:
    """Test behavior with C++ built-in steps."""

    def test_unregister_cpp_step_raises_unknown_step_error(self):
        """Attempting to unregister a C++ step raises UnknownStepError, not ValueError."""
        with pytest.raises(UnknownStepError):
            unregister_step("drop_nulls")

    @pytest.mark.parametrize("cpp_step", [
        "drop_columns",
        "strip_whitespace",
        "fill_nulls",
        "cast_types",
        "drop_duplicates",
    ])
    def test_unregister_other_cpp_steps(self, cpp_step):
        """Other C++ steps also raise UnknownStepError."""
        with pytest.raises(UnknownStepError):
            unregister_step(cpp_step)


class TestUnregisterStepMultipleCustomSteps:
    """Test behavior with multiple custom steps."""

    def test_unregister_one_custom_step_leaves_others(self):
        """Unregistering one custom step leaves others registered."""
        register_step("zz_step_1", lambda df: df)
        register_step("zz_step_2", lambda df: df)
        register_step("zz_step_3", lambda df: df)

        unregister_step("zz_step_2")

        assert "zz_step_1" in list_steps()
        assert "zz_step_2" not in list_steps()
        assert "zz_step_3" in list_steps()

    def test_unregister_all_custom_steps_sequentially(self):
        """Can unregister multiple custom steps one by one."""
        register_step("zz_step_1", lambda df: df)
        register_step("zz_step_2", lambda df: df)

        unregister_step("zz_step_1")
        unregister_step("zz_step_2")

        assert "zz_step_1" not in list_steps()
        assert "zz_step_2" not in list_steps()


class TestUnregisterStepEdgeCases:
    """Test edge cases and special scenarios."""

    def test_unregister_step_returns_none(self):
        """Verify that unregister_step returns None explicitly."""
        register_step("zz_test", lambda df: df)

        result = unregister_step("zz_test")

        assert result is None

    def test_case_sensitive_step_names(self):
        """Step names are case-sensitive."""
        register_step("zz_Test_Step", lambda df: df)

        # Different case should not be found
        with pytest.raises(UnknownStepError):
            unregister_step("zz_test_step")

        # Original case should work
        unregister_step("zz_Test_Step")
        assert "zz_Test_Step" not in list_steps()

    def test_unregister_step_with_special_characters(self):
        """Custom steps with special characters work correctly."""
        register_step("zz_step:with:colons", lambda df: df)

        assert "zz_step:with:colons" in list_steps()

        unregister_step("zz_step:with:colons")

        assert "zz_step:with:colons" not in list_steps()
