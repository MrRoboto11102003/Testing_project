import pytest
import pandas as pd
from arnio.pipeline import register_step, unregister_step


class TestUnregisterStep:
    """Tests for arnio.pipeline.unregister_step"""
    
    def test_unregister_valid_custom_step(self):
        """Test unregistering a valid custom step returns None"""
        def custom_step(df):
            return df
        
        step_name = "test_valid_unregister"
        register_step(step_name, custom_step)
        result = unregister_step(step_name)
        assert result is None
    
    def test_unregister_nonexistent_step_raises_error(self):
        """Test unregistering non-existent step raises error"""
        with pytest.raises(Exception):
            unregister_step("nonexistent_step_xyz")
    
    def test_unregister_builtin_step_raises_valueerror(self):
        """Test that unregistering built-in step raises ValueError"""
        with pytest.raises(ValueError, match="Cannot unregister built-in step"):
            unregister_step("dropna")
    
    def test_unregister_non_string_int_raises_typeerror(self):
        """Test that non-string int raises TypeError"""
        with pytest.raises(TypeError, match="parameter 'name' must be a string"):
            unregister_step(123)
    
    def test_unregister_non_string_none_raises_typeerror(self):
        """Test that None raises TypeError"""
        with pytest.raises(TypeError, match="parameter 'name' must be a string"):
            unregister_step(None)
    
    def test_unregister_non_string_list_raises_typeerror(self):
        """Test that list raises TypeError"""
        with pytest.raises(TypeError, match="parameter 'name' must be a string"):
            unregister_step(["step"])
    
    def test_unregister_non_string_dict_raises_typeerror(self):
        """Test that dict raises TypeError"""
        with pytest.raises(TypeError, match="parameter 'name' must be a string"):
            unregister_step({"key": "value"})
    
    def test_unregister_non_string_float_raises_typeerror(self):
        """Test that float raises TypeError"""
        with pytest.raises(TypeError, match="parameter 'name' must be a string"):
            unregister_step(3.14)
    
    def test_unregister_empty_string_raises_valueerror(self):
        """Test that empty string raises ValueError"""
        with pytest.raises(ValueError, match="parameter 'name' must be a non-empty string"):
            unregister_step("")
    
    def test_unregister_step_actually_removed(self):
        """Test that step is actually removed from registry"""
        def dummy_step(df):
            return df.copy()
        
        step_name = "test_step_actually_removed"
        register_step(step_name, dummy_step)
        unregister_step(step_name)
        
        # Second unregister should fail
        with pytest.raises(Exception):
            unregister_step(step_name)
    
    def test_unregister_multiple_custom_steps(self):
        """Test unregistering multiple custom steps sequentially"""
        def step_func(df):
            return df
        
        step_names = [f"test_multi_step_{i}" for i in range(3)]
        for name in step_names:
            register_step(name, step_func)
        
        for name in step_names:
            unregister_step(name)
        
        for name in step_names:
            with pytest.raises(Exception):
                unregister_step(name)
    
    def test_unregister_does_not_affect_other_steps(self):
        """Test that unregistering one step doesn't affect others"""
        def step_func(df):
            return df
        
        step1 = "test_unregister_step1"
        step2 = "test_unregister_step2"
        
        register_step(step1, step_func)
        register_step(step2, step_func)
        
        unregister_step(step1)
        
        # step2 should still exist and unregister successfully
        unregister_step(step2)
        
        with pytest.raises(Exception):
            unregister_step(step2)
    
    def test_unregister_case_sensitive_names(self):
        """Test that step names are case-sensitive"""
        def step_func(df):
            return df
        
        step_name = "TestCaseSensitive"
        register_step(step_name, step_func)
        
        # Correct case should work
        unregister_step(step_name)
        
        # Should be gone
        with pytest.raises(Exception):
            unregister_step(step_name)
    
    def test_unregister_non_string_bool_raises_typeerror(self):
        """Test that bool raises TypeError"""
        with pytest.raises(TypeError, match="parameter 'name' must be a string"):
            unregister_step(True)
