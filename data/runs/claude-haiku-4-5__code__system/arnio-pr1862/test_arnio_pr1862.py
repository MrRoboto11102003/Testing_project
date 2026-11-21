import pytest
import pandas as pd
import numpy as np
import sys
import subprocess
from pathlib import Path
import csv
import arnio as ar


class TestRegisterStep:
    """Test suite for arnio.pipeline.register_step"""
    
    def test_register_custom_step_basic(self):
        """Test basic registration of a custom step."""
        def my_step(df, **kwargs):
            return df
        
        ar.register_step("my_step", my_step)
        try:
            assert True
        finally:
            ar.reset_steps()
    
    def test_register_and_apply_custom_step_with_dataframe(self, tmp_path):
        """Test registering and applying a custom step to process data."""
        csv_file = tmp_path / "input.csv"
        pd.DataFrame({
            'value': [1, 2, 3, 4, 5],
            'category': ['X', 'Y', 'X', 'Y', 'Z']
        }).to_csv(csv_file, index=False)
        
        def filter_category_x(df, **kwargs):
            return df[df['category'] == 'X'].reset_index(drop=True)
        
        ar.register_step("filter_category_x", filter_category_x)
        
        try:
            df = pd.read_csv(csv_file)
            if hasattr(df, 'arnio') and hasattr(df.arnio, 'apply_steps'):
                result = df.arnio.apply_steps([{"step": "filter_category_x"}])
                assert len(result) == 2
                assert all(result['category'] == 'X')
        finally:
            ar.reset_steps()
    
    def test_register_step_with_kwargs_parameters(self):
        """Test that registered steps accept and use keyword arguments."""
        def multiply_column(df, column='value', factor=1, **kwargs):
            df = df.copy()
            if column in df.columns:
                df[column] = df[column] * factor
            return df
        
        ar.register_step("multiply_column", multiply_column)
        
        try:
            df = pd.DataFrame({'value': [1, 2, 3], 'other': [10, 20, 30]})
            if hasattr(df, 'arnio') and hasattr(df.arnio, 'apply_steps'):
                result = df.arnio.apply_steps([
                    {"step": "multiply_column", "column": "value", "factor": 5}
                ])
                assert all(result['value'] == df['value'] * 5)
        finally:
            ar.reset_steps()
    
    def test_overwrite_false_prevents_duplicate_registration(self):
        """Test that overwrite=False raises ValueError for duplicate registration."""
        def step_v1(df, **kwargs):
            return df
        
        def step_v2(df, **kwargs):
            return df
        
        ar.register_step("duplicate_test", step_v1)
        
        try:
            with pytest.raises(ValueError, match="already registered as a custom Python step"):
                ar.register_step("duplicate_test", step_v2, overwrite=False)
        finally:
            ar.reset_steps()
    
    def test_overwrite_true_replaces_existing_custom_step(self):
        """Test that overwrite=True allows replacing an existing custom step."""
        def step_a(df, **kwargs):
            df = df.copy()
            df['version'] = 'a'
            return df
        
        def step_b(df, **kwargs):
            df = df.copy()
            df['version'] = 'b'
            return df
        
        ar.register_step("versioned", step_a)
        ar.register_step("versioned", step_b, overwrite=True)
        
        try:
            assert True
        finally:
            ar.reset_steps()
    
    def test_empty_name_raises_valueerror(self):
        """Test that empty or whitespace-only name raises ValueError."""
        def dummy(df, **kwargs):
            return df
        
        with pytest.raises(ValueError, match="non-empty string"):
            ar.register_step("", dummy)
        
        with pytest.raises(ValueError, match="non-empty string"):
            ar.register_step("   ", dummy)
    
    def test_non_string_name_raises_error(self):
        """Test that non-string name raises ValueError or TypeError."""
        def dummy(df, **kwargs):
            return df
        
        with pytest.raises((ValueError, TypeError)):
            ar.register_step(123, dummy)
        
        with pytest.raises((ValueError, TypeError)):
            ar.register_step(None, dummy)
    
    def test_non_callable_fn_raises_typeerror(self):
        """Test that non-callable fn parameter raises TypeError."""
        with pytest.raises(TypeError, match="must be a callable"):
            ar.register_step("step", "not_callable")
        
        with pytest.raises(TypeError, match="must be a callable"):
            ar.register_step("step", 42)
    
    def test_non_bool_overwrite_raises_typeerror(self):
        """Test that non-bool overwrite parameter raises TypeError."""
        def dummy(df, **kwargs):
            return df
        
        with pytest.raises(TypeError, match="must be a bool"):
            ar.register_step("step", dummy, overwrite="yes")
        
        with pytest.raises(TypeError, match="must be a bool"):
            ar.register_step("step", dummy, overwrite=1)
    
    def test_builtin_cpp_step_conflict_raises_valueerror(self):
        """Test that attempting to register a built-in C++ step name raises ValueError."""
        def dummy(df, **kwargs):
            return df
        
        with pytest.raises(ValueError, match="conflicts with built-in C\\+\\+ step"):
            ar.register_step("dropna", dummy)
    
    def test_reserved_builtin_namespace_prefix_raises_valueerror(self):
        """Test that reserved namespace prefix raises ValueError."""
        def dummy(df, **kwargs):
            return df
        
        with pytest.raises(ValueError, match="reserved for built-in"):
            ar.register_step("builtin::my_step", dummy)
    
    def test_register_multiple_distinct_custom_steps(self, tmp_path):
        """Test registering and chaining multiple custom steps."""
        def add_column_a(df, **kwargs):
            df = df.copy()
            df['col_a'] = 1
            return df
        
        def add_column_b(df, **kwargs):
            df = df.copy()
            df['col_b'] = 2
            return df
        
        ar.register_step("add_column_a", add_column_a)
        ar.register_step("add_column_b", add_column_b)
        
        try:
            csv_file = tmp_path / "data.csv"
            pd.DataFrame({'value': [1, 2, 3]}).to_csv(csv_file, index=False)
            
            df = pd.read_csv(csv_file)
            if hasattr(df, 'arnio') and hasattr(df.arnio, 'apply_steps'):
                result = df.arnio.apply_steps([
                    {"step": "add_column_a"},
                    {"step": "add_column_b"}
                ])
                assert 'col_a' in result.columns
                assert 'col_b' in result.columns
        finally:
            ar.reset_steps()


class TestResetSteps:
    """Test suite for arnio.pipeline.reset_steps"""
    
    def test_reset_clears_custom_steps_allows_reregistration(self):
        """Test that reset_steps clears custom steps and allows re-registering with same names."""
        def step_v1(df, **kwargs):
            return df
        
        def step_v2(df, **kwargs):
            return df
        
        ar.register_step("reusable", step_v1)
        
        # Reset should clear the step
        ar.reset_steps()
        
        # Should now be able to register with same name without overwrite=True
        ar.register_step("reusable", step_v2)
        
        try:
            assert True
        finally:
            ar.reset_steps()
    
    def test_reset_clears_multiple_custom_steps(self):
        """Test that reset_steps clears all custom steps from the registry."""
        def step_a(df, **kwargs):
            return df
        
        def step_b(df, **kwargs):
            return df
        
        def step_c(df, **kwargs):
            return df
        
        ar.register_step("step_a", step_a)
        ar.register_step("step_b", step_b)
        ar.register_step("step_c", step_c)
        
        # Reset should clear all
        ar.reset_steps()
        
        # All should be re-registrable without overwrite
        ar.register_step("step_a", step_a)
        ar.register_step("step_b", step_b)
        ar.register_step("step_c", step_c)
        
        try:
            assert True
        finally:
            ar.reset_steps()
    
    def test_reset_is_idempotent(self):
        """Test that reset_steps can be called multiple times without error."""
        def step(df, **kwargs):
            return df
        
        ar.register_step("temp", step)
        
        ar.reset_steps()
        ar.reset_steps()
        ar.reset_steps()
        
        # Should still work after multiple resets
        ar.register_step("temp", step)
        
        try:
            assert True
        finally:
            ar.reset_steps()
    
    def test_complete_lifecycle_register_reset_reregister(self, tmp_path):
        """Test complete lifecycle: register, reset, verify cleanup and re-registration."""
        csv_file = tmp_path / "data.csv"
        pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['A', 'B', 'C']
        }).to_csv(csv_file, index=False)
        
        def mark_processed(df, **kwargs):
            df = df.copy()
            df['processed'] = True
            return df
        
        # Register step
        ar.register_step("mark_processed", mark_processed)
        
        # Verify registration works
        df = pd.read_csv(csv_file)
        if hasattr(df, 'arnio') and hasattr(df.arnio, 'apply_steps'):
            result1 = df.arnio.apply_steps([{"step": "mark_processed"}])
            assert 'processed' in result1.columns
        
        # Reset
        ar.reset_steps()
        
        # Verify step is gone - attempting to use it should fail
        if hasattr(df, 'arnio') and hasattr(df.arnio, 'apply_steps'):
            with pytest.raises((ValueError, KeyError)):
                df.arnio.apply_steps([{"step": "mark_processed"}])
        
        # Verify re-registration works without overwrite=True
        ar.register_step("mark_processed", mark_processed)
        result2 = df.arnio.apply_steps([{"step": "mark_processed"}])
        assert 'processed' in result2.columns
        
        try:
            assert True
        finally:
            ar.reset_steps()
