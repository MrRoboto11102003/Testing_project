import pytest
import pandas as pd
import arnio as ar
from arnio.exceptions import UnknownStepError


@pytest.fixture(autouse=True)
def cleanup_steps():
    """Reset steps after each test to avoid state pollution."""
    yield
    ar.reset_steps()


def test_unregister_custom_step_basic():
    """Test that a custom step can be successfully unregistered."""
    def custom_func(df):
        return df
    
    ar.register_step("test_basic_unregister", custom_func)
    ar.unregister_step("test_basic_unregister")
    
    assert "test_basic_unregister" not in ar.list_steps()


def test_unregister_step_name_not_string_int():
    """Test that unregister_step rejects integer names."""
    with pytest.raises(TypeError, match="must be a string"):
        ar.unregister_step(123)


def test_unregister_step_name_not_string_none():
    """Test that unregister_step rejects None as name."""
    with pytest.raises(TypeError, match="must be a string"):
        ar.unregister_step(None)


def test_unregister_step_empty_string():
    """Test that unregister_step rejects empty string."""
    with pytest.raises(ValueError, match="must be a non-empty string"):
        ar.unregister_step("")


def test_unregister_step_not_found():
    """Test that unregister_step raises UnknownStepError for non-existent steps."""
    with pytest.raises(UnknownStepError):
        ar.unregister_step("nonexistent_step_xyz_abc")


def test_unregister_builtin_drop_nulls_fails():
    """Test that built-in steps cannot be unregistered."""
    with pytest.raises(ValueError, match="Cannot unregister built-in step"):
        ar.unregister_step("drop_nulls")


def test_unregister_step_removes_from_registry():
    """Test that unregistering actually removes the step from the registry."""
    def my_step(df):
        return df
    
    ar.register_step("verify_removal", my_step)
    assert "verify_removal" in ar.list_steps()
    
    ar.unregister_step("verify_removal")
    assert "verify_removal" not in ar.list_steps()


def test_unregister_and_reregister():
    """Test that a step can be re-registered after unregistering."""
    def step_func(df):
        return df
    
    ar.register_step("reregister_test", step_func)
    ar.unregister_step("reregister_test")
    
    ar.register_step("reregister_test", step_func, overwrite=True)
    assert "reregister_test" in ar.list_steps()


def test_unregister_multiple_custom_steps_isolated():
    """Test that unregistering one step doesn't affect others."""
    def step_func(df):
        return df
    
    ar.register_step("multi_step1", step_func)
    ar.register_step("multi_step2", step_func)
    
    ar.unregister_step("multi_step1")
    
    assert "multi_step1" not in ar.list_steps()
    assert "multi_step2" in ar.list_steps()


def test_unregister_namespaced_step():
    """Test that namespaced custom steps can be unregistered."""
    def namespaced_func(df):
        return df
    
    ar.register_step("team:custom_step", namespaced_func)
    ar.unregister_step("team:custom_step")
    
    assert "team:custom_step" not in ar.list_steps()


def test_unregister_step_case_sensitive():
    """Test that step names are case-sensitive."""
    def func(df):
        return df
    
    ar.register_step("CaseStep", func)
    
    with pytest.raises(UnknownStepError):
        ar.unregister_step("casestep")
    
    assert "CaseStep" in ar.list_steps()


def test_unregister_step_then_pipeline_fails():
    """Test that after unregistering, the step fails in pipelines."""
    def step_func(df):
        return df
    
    ar.register_step("pipeline_test", step_func)
    ar.unregister_step("pipeline_test")
    
    frame = ar.from_pandas(pd.DataFrame({"x": [1, 2, 3]}))
    with pytest.raises(UnknownStepError):
        ar.pipeline(frame, [("pipeline_test",)])


def test_unregister_step_double_unregister():
    """Test that unregistering the same step twice raises UnknownStepError."""
    def func(df):
        return df
    
    ar.register_step("double_test", func)
    ar.unregister_step("double_test")
    
    with pytest.raises(UnknownStepError):
        ar.unregister_step("double_test")


def test_unregister_step_with_special_characters():
    """Test that steps with special characters in names can be unregistered."""
    def func(df):
        return df
    
    ar.register_step("my-step_v2.0", func)
    ar.unregister_step("my-step_v2.0")
    
    assert "my-step_v2.0" not in ar.list_steps()


def test_unregister_builtin_strip_whitespace():
    """Test that built-in strip_whitespace cannot be unregistered."""
    with pytest.raises(ValueError, match="Cannot unregister built-in step"):
        ar.unregister_step("strip_whitespace")
