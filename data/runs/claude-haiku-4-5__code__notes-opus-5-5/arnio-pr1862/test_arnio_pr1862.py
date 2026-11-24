import pytest
import pandas as pd
import re
import importlib
import arnio as ar
from arnio.pipeline import register_step, reset_steps, list_steps

# Import module for access to private registry if needed
pm = importlib.import_module("arnio.pipeline")

BUILTIN_PYTHON_STEPS = {
    "standardize_missing_tokens",
    "coalesce_columns",
    "normalize_whitespace",
    "filter_rows",
    "drop_columns_matching",
    "rename_columns_matching",
    "safe_divide_columns",
    "replace_values"
}


@pytest.fixture(autouse=True)
def cleanup_registry():
    """Reset registry before and after each test to prevent state leakage"""
    reset_steps()
    yield
    reset_steps()


class TestRegisterStep:

    def test_register_step_basic_registration(self):
        """Test successful registration and use of a custom step in pipeline"""
        def my_step(df):
            return df.assign(new_col=1)

        register_step("test_step_basic", my_step)
        assert "test_step_basic" in list_steps()

        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))
        result = ar.pipeline(frame, [("test_step_basic",)])
        result_df = ar.to_pandas(result)
        assert "new_col" in result_df.columns
        assert result_df["new_col"].tolist() == [1, 1]

    def test_register_step_overwrite_true(self):
        """Test that overwrite=True allows replacing an existing custom step"""
        def step_v1(df):
            return df.assign(version=1)

        def step_v2(df):
            return df.assign(version=2)

        register_step("test_step_overwrite", step_v1)
        register_step("test_step_overwrite", step_v2, overwrite=True)

        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        result = ar.pipeline(frame, [("test_step_overwrite",)])
        result_df = ar.to_pandas(result)
        assert result_df["version"].iloc[0] == 2

    def test_register_step_empty_name(self):
        """Test that empty string name raises ValueError"""
        def my_step(df):
            return df

        with pytest.raises(ValueError, match=re.escape("parameter 'name' must be a non-empty string")):
            register_step("", my_step)

    def test_register_step_whitespace_only_name(self):
        """Test that whitespace-only name raises ValueError"""
        def my_step(df):
            return df

        with pytest.raises(ValueError, match=re.escape("parameter 'name' must be a non-empty string")):
            register_step("   ", my_step)

    def test_register_step_non_callable_fn(self):
        """Test that non-callable fn raises TypeError"""
        with pytest.raises(TypeError, match=re.escape("parameter 'fn' must be a callable object")):
            register_step("test_step_callable", None)

    def test_register_step_non_bool_overwrite(self):
        """Test that non-bool overwrite parameter raises TypeError"""
        def my_step(df):
            return df

        with pytest.raises(TypeError, match=re.escape("parameter 'overwrite' must be a bool, not int")):
            register_step("test_step_bool", my_step, overwrite=1)

    def test_register_step_reserved_builtin_prefix(self):
        """Test that 'builtin:' prefix is reserved and raises ValueError"""
        def my_step(df):
            return df

        with pytest.raises(ValueError, match=re.escape("'builtin:' is reserved for built-in pipeline steps")):
            register_step("builtin:custom", my_step)

    def test_register_step_cpp_name_conflict(self):
        """Test that C++ built-in step names cannot be registered"""
        def my_step(df):
            return df

        with pytest.raises(ValueError, match=re.escape("conflicts with built-in C++ step")):
            register_step("drop_nulls", my_step)

    def test_register_step_python_builtin_name_conflict(self):
        """Test that built-in Python step names cannot be registered"""
        def my_step(df):
            return df

        with pytest.raises(ValueError, match=re.escape("conflicts with built-in Python step")):
            register_step("filter_rows", my_step)

    def test_register_step_duplicate_without_overwrite(self):
        """Test that registering duplicate custom step without overwrite raises ValueError"""
        def my_step(df):
            return df

        register_step("test_step_duplicate", my_step)

        with pytest.raises(ValueError, match=re.escape("already registered as a custom Python step")):
            register_step("test_step_duplicate", my_step)


class TestResetSteps:

    def test_reset_steps_removes_custom_steps(self):
        """Test that reset_steps removes all custom registered steps"""
        def my_step(df):
            return df

        register_step("test_reset_custom_1", my_step)
        register_step("test_reset_custom_2", my_step)

        reset_steps()

        assert "test_reset_custom_1" not in list_steps()
        assert "test_reset_custom_2" not in list_steps()

    def test_reset_steps_restores_builtin_python_steps(self):
        """Test that reset_steps restores all built-in Python steps"""
        def my_step(df):
            return df

        register_step("test_reset_builtin", my_step)
        reset_steps()

        steps = set(list_steps())
        assert BUILTIN_PYTHON_STEPS <= steps, f"Missing built-in steps: {BUILTIN_PYTHON_STEPS - steps}"

    def test_reset_steps_returns_none(self):
        """Test that reset_steps returns None"""
        result = reset_steps()
        assert result is None

    def test_reset_steps_idempotent(self):
        """Test that calling reset_steps multiple times has the same effect"""
        def my_step(df):
            return df

        register_step("test_reset_idempotent", my_step)
        reset_steps()
        steps_after_first = set(list_steps())

        reset_steps()
        steps_after_second = set(list_steps())

        assert steps_after_first == steps_after_second

    def test_reset_steps_allows_reregistration(self):
        """Test that after reset, a custom step name can be registered again"""
        def my_step(df):
            return df

        register_step("test_reset_rereg", my_step)
        reset_steps()

        # This should not raise - reuses name after reset
        register_step("test_reset_rereg", my_step)
        assert "test_reset_rereg" in list_steps()
