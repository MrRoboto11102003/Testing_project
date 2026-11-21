import pytest
import pandas as pd
from unittest.mock import patch
import arnio as ar


class TestRegisterStep:
    """Tests for arnio.pipeline.register_step."""
    
    def teardown_method(self):
        """Clean up after each test."""
        ar.reset_steps()
    
    def test_register_new_step(self):
        """Successfully register a new custom step."""
        def custom_step(df, threshold=0.5):
            return df.dropna(thresh=threshold)
        
        ar.register_step("custom_clean", custom_step)
        
        # Verify it was registered by attempting duplicate
        with pytest.raises(ValueError, match="already registered"):
            ar.register_step("custom_clean", custom_step)
    
    def test_register_with_overwrite_true(self):
        """Successfully overwrite existing custom step with overwrite=True."""
        def step_v1(df):
            return df
        
        def step_v2(df):
            return df.copy()
        
        ar.register_step("versioned_step", step_v1)
        ar.register_step("versioned_step", step_v2, overwrite=True)
        
        # Verify overwrites work by attempting duplicate without overwrite
        with pytest.raises(ValueError, match="already registered"):
            ar.register_step("versioned_step", step_v1, overwrite=False)
    
    def test_empty_string_name_raises_valueerror(self):
        """Registering with empty string name raises ValueError."""
        def fn(df):
            return df
        
        with pytest.raises(ValueError, match="non-empty string"):
            ar.register_step("", fn)
    
    def test_whitespace_only_name_raises_valueerror(self):
        """Registering with whitespace-only name raises ValueError."""
        def fn(df):
            return df
        
        with pytest.raises(ValueError, match="non-empty string"):
            ar.register_step("   ", fn)
    
    def test_non_string_name_raises_valueerror(self):
        """Registering with non-string name raises ValueError."""
        def fn(df):
            return df
        
        with pytest.raises(ValueError, match="non-empty string"):
            ar.register_step(123, fn)
    
    def test_non_callable_function_raises_typeerror(self):
        """Registering with non-callable function raises TypeError."""
        with pytest.raises(TypeError, match="must be a callable"):
            ar.register_step("step_name", "not_a_function")
    
    def test_non_bool_overwrite_raises_typeerror(self):
        """Registering with non-bool overwrite raises TypeError."""
        def fn(df):
            return df
        
        with pytest.raises(TypeError, match="must be a bool"):
            ar.register_step("step_name", fn, overwrite=1)
    
    def test_duplicate_registration_without_overwrite_raises(self):
        """Registering duplicate custom step without overwrite raises ValueError."""
        def fn(df):
            return df
        
        ar.register_step("dup_test", fn)
        
        with pytest.raises(ValueError, match="already registered"):
            ar.register_step("dup_test", fn)
    
    @patch("arnio.pipeline._STEP_REGISTRY", {"builtin_cpp": None})
    @patch("arnio.pipeline._BUILTIN_PYTHON_STEP_REGISTRY", {})
    @patch("arnio.pipeline._DEPRECATED_STEP_ALIASES", {})
    @patch("arnio.pipeline._PYTHON_STEP_REGISTRY", {})
    def test_builtin_cpp_step_conflict(self):
        """Registering with name conflicting with built-in C++ step raises ValueError."""
        def fn(df):
            return df
        
        with pytest.raises(ValueError, match="conflicts with built-in C\\+\\+ step"):
            ar.register_step("builtin_cpp", fn)
    
    @patch("arnio.pipeline._STEP_REGISTRY", {})
    @patch("arnio.pipeline._BUILTIN_PYTHON_STEP_REGISTRY", {"builtin_py": None})
    @patch("arnio.pipeline._DEPRECATED_STEP_ALIASES", {})
    @patch("arnio.pipeline._PYTHON_STEP_REGISTRY", {})
    def test_builtin_python_step_conflict(self):
        """Registering with name conflicting with built-in Python step raises ValueError."""
        def fn(df):
            return df
        
        with pytest.raises(ValueError, match="conflicts with built-in Python step"):
            ar.register_step("builtin_py", fn)
    
    @patch("arnio.pipeline._STEP_REGISTRY", {})
    @patch("arnio.pipeline._BUILTIN_PYTHON_STEP_REGISTRY", {})
    @patch("arnio.pipeline._DEPRECATED_STEP_ALIASES", {"deprecated": None})
    @patch("arnio.pipeline._PYTHON_STEP_REGISTRY", {})
    def test_deprecated_alias_conflict(self):
        """Registering with name conflicting with deprecated alias raises ValueError."""
        def fn(df):
            return df
        
        with pytest.raises(ValueError, match="reserved as a deprecated"):
            ar.register_step("deprecated", fn)


class TestResetSteps:
    """Tests for arnio.pipeline.reset_steps."""
    
    def test_reset_clears_custom_steps(self):
        """Reset clears custom steps and allows re-registration."""
        def fn(df):
            return df
        
        ar.register_step("temp_step", fn)
        
        # Verify it was registered
        with pytest.raises(ValueError, match="already registered"):
            ar.register_step("temp_step", fn)
        
        ar.reset_steps()
        
        # Should be able to register again after reset
        ar.register_step("temp_step", fn)
        ar.reset_steps()
    
    def test_reset_multiple_times_safe(self):
        """Multiple consecutive calls to reset are safe."""
        def fn(df):
            return df
        
        ar.register_step("step1", fn)
        ar.reset_steps()
        ar.reset_steps()
        ar.reset_steps()
        
        ar.register_step("step1", fn)
        ar.reset_steps()
