import pytest
import pandas as pd
import re
from arnio.pipeline import register_step, unregister_step, list_steps, reset_steps, pipeline
from arnio.exceptions import UnknownStepError
import arnio as ar


@pytest.fixture(autouse=True)
def cleanup_registry():
    """Reset the registry before and after each test."""
    reset_steps()
    yield
    reset_steps()


def test_unregister_custom_step_returns_none():
    """Unregistering a custom step returns None."""
    register_step("zz_test_step", lambda df: df)
    result = unregister_step("zz_test_step")
    assert result is None


def test_unregister_custom_step_removes_from_registry():
    """After unregistering, the step is no longer in list_steps()."""
    register_step("zz_test_step", lambda df: df)
    assert "zz_test_step" in list_steps()
    unregister_step("zz_test_step")
    assert "zz_test_step" not in list_steps()


def test_unregister_custom_step_cannot_be_used_in_pipeline():
    """After unregistering, using the step in a pipeline raises UnknownStepError."""
    register_step("zz_test_step", lambda df: df)
    unregister_step("zz_test_step")
    df = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
    with pytest.raises(UnknownStepError):
        pipeline(df, [("zz_test_step",)])


def test_unregister_with_non_string_raises_typeerror():
    """Passing non-string types raises TypeError with correct message."""
    with pytest.raises(TypeError, match=re.escape("parameter 'name' must be a string, not int")):
        unregister_step(123)
    
    with pytest.raises(TypeError, match=re.escape("parameter 'name' must be a string, not NoneType")):
        unregister_step(None)
    
    with pytest.raises(TypeError, match=re.escape("parameter 'name' must be a string, not bytes")):
        unregister_step(b"test")


def test_unregister_with_empty_string_raises_valueerror():
    """Passing empty string raises ValueError with correct message."""
    with pytest.raises(ValueError, match=re.escape("parameter 'name' must be a non-empty string")):
        unregister_step("")


def test_unregister_protected_builtin_raises_valueerror():
    """Cannot unregister protected built-in steps."""
    with pytest.raises(ValueError, match=re.escape(
        "Cannot unregister built-in step 'filter_rows'. Only custom user-registered steps can be unregistered."
    )):
        unregister_step("filter_rows")
    
    with pytest.raises(ValueError, match=re.escape(
        "Cannot unregister built-in step 'coalesce_columns'. Only custom user-registered steps can be unregistered."
    )):
        unregister_step("coalesce_columns")


def test_unregister_nonexistent_step_raises_unknowsteparror():
    """Unregistering a non-existent step raises UnknownStepError."""
    with pytest.raises(UnknownStepError):
        unregister_step("zz_nonexistent_step")


def test_unregister_twice_raises_unknowsteparror():
    """Calling unregister twice raises UnknownStepError the second time."""
    register_step("zz_test_step", lambda df: df)
    unregister_step("zz_test_step")
    with pytest.raises(UnknownStepError):
        unregister_step("zz_test_step")


def test_unregister_cpp_step_raises_unknowsteparror():
    """Trying to unregister a C++ built-in step raises UnknownStepError, not ValueError."""
    with pytest.raises(UnknownStepError):
        unregister_step("drop_nulls")


def test_unregister_whitespace_only_raises_unknowsteparror():
    """A whitespace-only name raises UnknownStepError (passes validation but fails lookup)."""
    with pytest.raises(UnknownStepError):
        unregister_step("   ")


def test_registry_unchanged_after_typeerror():
    """Registry is unchanged after a TypeError."""
    register_step("zz_test_step", lambda df: df)
    with pytest.raises(TypeError):
        unregister_step(123)
    assert "zz_test_step" in list_steps()


def test_registry_unchanged_after_failed_unregister_builtin():
    """Registry is unchanged after trying to unregister a built-in."""
    with pytest.raises(ValueError):
        unregister_step("filter_rows")
    assert "filter_rows" in list_steps()


def test_can_reregister_after_unregister():
    """After unregistering, the step can be re-registered without overwrite=True."""
    register_step("zz_test_step", lambda df: df)
    unregister_step("zz_test_step")
    register_step("zz_test_step", lambda df: df.copy())
    assert "zz_test_step" in list_steps()


def test_unregister_does_not_affect_other_custom_steps():
    """Unregistering one step leaves others registered."""
    register_step("zz_step_1", lambda df: df)
    register_step("zz_step_2", lambda df: df)
    unregister_step("zz_step_1")
    assert "zz_step_1" not in list_steps()
    assert "zz_step_2" in list_steps()


def test_alias_of_cleaning_function_can_be_unregistered():
    """An alias of a cleaning function can be unregistered if not in builtin list."""
    from arnio.cleaning import filter_rows as cleanup_filter
    register_step("zz_my_alias", cleanup_filter)
    assert "zz_my_alias" in list_steps()
    unregister_step("zz_my_alias")
    assert "zz_my_alias" not in list_steps()
