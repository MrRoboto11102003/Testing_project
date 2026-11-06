import pytest
import pandas as pd
import arnio as ar
import numpy as np


class TestRegisterStep:
    """Test arnio.pipeline.register_step functionality."""
    
    def teardown_method(self):
        """Clean up custom steps after each test."""
        ar.reset_steps()
    
    def test_register_simple_custom_step(self):
        """Test registering a basic custom step."""
        def simple_step(df):
            return df
        
        initial_steps = set(ar.list_steps())
        ar.register_step("simple_step", simple_step)
        final_steps = set(ar.list_steps())
        
        assert "simple_step" in final_steps
        assert "simple_step" not in initial_steps
    
    def test_custom_step_in_pipeline(self):
        """Test that a registered custom step can be used in a pipeline."""
        def multiply_column(df, column, factor):
            df = df.copy()
            df[column] = df[column] * factor
            return df
        
        ar.register_step("multiply_column", multiply_column)
        
        frame = ar.from_pandas(pd.DataFrame({"value": [1, 2, 3]}))
        result = ar.pipeline(frame, [
            ("multiply_column", {"column": "value", "factor": 2}),
        ])
        
        result_df = ar.to_pandas(result)
        assert result_df["value"].tolist() == [2, 4, 6]
    
    def test_register_step_name_must_be_string(self):
        """Test that name parameter must be a non-empty string."""
        def dummy_fn(df):
            return df
        
        with pytest.raises(ValueError, match="non-empty string"):
            ar.register_step(123, dummy_fn)
        
        with pytest.raises(ValueError, match="non-empty string"):
            ar.register_step("", dummy_fn)
        
        with pytest.raises(ValueError, match="non-empty string"):
            ar.register_step("   ", dummy_fn)
    
    def test_register_step_fn_must_be_callable(self):
        """Test that fn parameter must be callable."""
        with pytest.raises(TypeError, match="must be a callable"):
            ar.register_step("not_callable", "not a function")
        
        with pytest.raises(TypeError, match="must be a callable"):
            ar.register_step("not_callable", 42)
    
    def test_register_step_overwrite_must_be_bool(self):
        """Test that overwrite parameter must be a boolean."""
        def dummy_fn(df):
            return df
        
        with pytest.raises(TypeError, match="must be a bool"):
            ar.register_step("test_step", dummy_fn, overwrite="yes")
        
        with pytest.raises(TypeError, match="must be a bool"):
            ar.register_step("test_step", dummy_fn, overwrite=1)
    
    def test_cannot_overwrite_builtin_step(self):
        """Test that built-in steps cannot be overwritten."""
        def fake_drop_nulls(df):
            return df
        
        with pytest.raises(ValueError, match="conflicts with built-in"):
            ar.register_step("drop_nulls", fake_drop_nulls)
    
    def test_register_without_overwrite_raises_on_duplicate_custom(self):
        """Test that registering the same custom name twice without overwrite raises."""
        def step_v1(df):
            return df
        
        def step_v2(df):
            return df
        
        ar.register_step("my_custom_step", step_v1)
        
        with pytest.raises(ValueError, match="already registered"):
            ar.register_step("my_custom_step", step_v2)
    
    def test_overwrite_true_replaces_custom_step(self):
        """Test that overwrite=True allows replacing a custom Python step."""
        def step_v1(df):
            df = df.copy()
            df["version"] = "v1"
            return df
        
        def step_v2(df):
            df = df.copy()
            df["version"] = "v2"
            return df
        
        ar.register_step("my_step", step_v1)
        ar.register_step("my_step", step_v2, overwrite=True)
        
        frame = ar.from_pandas(pd.DataFrame({"x": [1, 2]}))
        result = ar.pipeline(frame, [("my_step",)])
        result_df = ar.to_pandas(result)
        
        assert result_df["version"].iloc[0] == "v2"
    
    def test_cannot_register_builtin_namespace_prefix(self):
        """Test that steps cannot use the builtin: namespace prefix."""
        def dummy_fn(df):
            return df
        
        with pytest.raises(ValueError, match="reserved for built-in"):
            ar.register_step("builtin:custom_step", dummy_fn)


class TestResetSteps:
    """Test arnio.pipeline.reset_steps functionality."""
    
    def test_reset_steps_clears_custom_registry(self):
        """Test that reset_steps removes all custom steps."""
        def custom_step_1(df):
            return df
        
        def custom_step_2(df):
            return df
        
        ar.register_step("custom_step_1", custom_step_1)
        ar.register_step("custom_step_2", custom_step_2)
        
        steps_before = set(ar.list_steps())
        assert "custom_step_1" in steps_before
        assert "custom_step_2" in steps_before
        
        ar.reset_steps()
        
        steps_after = set(ar.list_steps())
        assert "custom_step_1" not in steps_after
        assert "custom_step_2" not in steps_after
    
    def test_reset_steps_preserves_builtins(self):
        """Test that reset_steps keeps built-in steps available."""
        def custom_step(df):
            return df
        
        initial_steps = set(ar.list_steps())
        
        ar.register_step("custom_step", custom_step)
        ar.reset_steps()
        
        final_steps = set(ar.list_steps())
        
        assert initial_steps == final_steps
        assert "custom_step" not in final_steps
    
    def test_reset_steps_allows_reregistering_name(self):
        """Test that after reset, you can register a step with the same name again."""
        def custom_step_v1(df):
            df = df.copy()
            df["tag"] = "v1"
            return df
        
        ar.register_step("my_step", custom_step_v1)
        ar.reset_steps()
        
        def custom_step_v2(df):
            df = df.copy()
            df["tag"] = "v2"
            return df
        
        ar.register_step("my_step", custom_step_v2)
        
        frame = ar.from_pandas(pd.DataFrame({"x": [1]}))
        result = ar.pipeline(frame, [("my_step",)])
        result_df = ar.to_pandas(result)
        
        assert result_df["tag"].iloc[0] == "v2"
    
    def test_multiple_resets_are_safe(self):
        """Test that calling reset_steps multiple times is safe."""
        def custom_step(df):
            return df
        
        ar.register_step("custom_step", custom_step)
        ar.reset_steps()
        ar.reset_steps()
        
        builtin_steps = ar.list_steps()
        assert len(builtin_steps) > 0


class TestCustomStepIntegration:
    """Integration tests for custom steps in pipelines."""
    
    def teardown_method(self):
        """Clean up custom steps after each test."""
        ar.reset_steps()
    
    def test_custom_step_with_multiple_kwargs(self):
        """Test custom step that accepts multiple keyword arguments."""
        def conditional_fill(df, column, value, only_if_null=True):
            df = df.copy()
            if only_if_null:
                mask = pd.isna(df[column])
                df.loc[mask, column] = value
            else:
                df[column] = value
            return df
        
        ar.register_step("conditional_fill", conditional_fill)
        
        frame = ar.from_pandas(pd.DataFrame({
            "col": [1.0, np.nan, 3.0]
        }))
        
        result = ar.pipeline(frame, [
            ("conditional_fill", {
                "column": "col",
                "value": 999.0,
                "only_if_null": True
            }),
        ])
        
        result_df = ar.to_pandas(result)
        assert result_df.loc[1, "col"] == 999.0
