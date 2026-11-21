import re
import pytest
import pandas as pd
import arnio as ar
from arnio.pipeline import register_step, reset_steps, list_steps


@pytest.fixture(autouse=True)
def cleanup_registry():
    """Reset registry before and after each test to avoid state leakage."""
    reset_steps()
    yield
    reset_steps()


class TestRegisterStep:
    """Tests for arnio.pipeline.register_step"""

    def test_register_simple_step(self):
        """Successfully register a simple custom step."""
        def my_step(df):
            return df
        
        register_step("my_test_step", my_step)
        assert "my_test_step" in list_steps()

    def test_name_empty_string(self):
        """Empty string name raises ValueError."""
        with pytest.raises(ValueError, match=re.escape("parameter 'name' must be a non-empty string")):
            register_step("", lambda df: df)

    def test_name_non_string(self):
        """Non-string name raises ValueError."""
        with pytest.raises(ValueError, match=re.escape("parameter 'name' must be a non-empty string")):
            register_step(123, lambda df: df)

    def test_fn_not_callable(self):
        """Non-callable fn raises TypeError."""
        with pytest.raises(TypeError, match=re.escape("parameter 'fn' must be a callable object")):
            register_step("my_step", None)

    def test_overwrite_not_bool(self):
        """Non-bool overwrite raises TypeError."""
        with pytest.raises(TypeError, match=re.escape("parameter 'overwrite' must be a bool, not int")):
            register_step("my_step", lambda df: df, overwrite=1)

    def test_reserved_builtin_prefix(self):
        """Name with 'builtin:' prefix raises ValueError."""
        with pytest.raises(ValueError, match=re.escape("'builtin:' is reserved")):
            register_step("builtin:foo", lambda df: df)

    def test_conflict_with_cpp_step(self):
        """Name conflicting with C++ step raises ValueError."""
        with pytest.raises(ValueError, match=re.escape("conflicts with built-in C++ step")):
            register_step("drop_nulls", lambda df: df)

    def test_conflict_with_python_step(self):
        """Name conflicting with built-in Python step raises ValueError."""
        with pytest.raises(ValueError, match=re.escape("conflicts with built-in Python step")):
            register_step("filter_rows", lambda df: df)

    def test_duplicate_without_overwrite(self):
        """Duplicate name without overwrite raises ValueError."""
        register_step("my_step", lambda df: df)
        with pytest.raises(ValueError, match=re.escape("Step 'my_step' is already registered")):
            register_step("my_step", lambda df: df)

    def test_overwrite_true_replaces_function(self):
        """With overwrite=True, function is replaced and new version is used."""
        def f1(df):
            return df
        
        def f2(df):
            return df.assign(replaced=True)
        
        register_step("my_step", f1)
        register_step("my_step", f2, overwrite=True)
        
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        result = ar.pipeline(frame, [("my_step",)])
        df_result = ar.to_pandas(result)
        assert "replaced" in df_result.columns

    def test_error_precedence_name_before_fn(self):
        """Name validation happens before fn validation."""
        with pytest.raises(ValueError, match=re.escape("parameter 'name'")):
            register_step(123, None)


class TestResetSteps:
    """Tests for arnio.pipeline.reset_steps"""

    def test_reset_removes_custom_steps(self):
        """reset_steps removes all custom registered steps."""
        register_step("custom1", lambda df: df)
        register_step("custom2", lambda df: df)
        
        reset_steps()
        
        assert "custom1" not in list_steps()
        assert "custom2" not in list_steps()

    def test_reset_restores_builtin_python_steps(self):
        """reset_steps restores all 8 built-in Python steps."""
        register_step("custom", lambda df: df)
        reset_steps()
        
        builtin_python = {
            "standardize_missing_tokens", "coalesce_columns", "normalize_whitespace",
            "filter_rows", "drop_columns_matching", "rename_columns_matching",
            "safe_divide_columns", "replace_values",
        }
        
        for step in builtin_python:
            assert step in list_steps()

    def test_reset_idempotent(self):
        """Calling reset_steps multiple times is safe."""
        register_step("custom", lambda df: df)
        
        reset_steps()
        steps1 = set(list_steps())
        
        reset_steps()
        steps2 = set(list_steps())
        
        assert steps1 == steps2

    def test_reset_then_reregister_same_name(self):
        """After reset, the same name can be registered again."""
        register_step("my_step", lambda df: df)
        reset_steps()
        
        # Should not raise
        register_step("my_step", lambda df: df)
        assert "my_step" in list_steps()
