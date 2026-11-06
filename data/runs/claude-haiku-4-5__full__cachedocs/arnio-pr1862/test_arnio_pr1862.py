"""
Pytest tests for arnio.pipeline.register_step and arnio.pipeline.reset_steps

Tests documented behavior including edge cases around step registration,
overwriting, and registry reset.
"""
import pytest
import pandas as pd
import arnio as ar


class TestRegisterStep:
    """Tests for ar.register_step()"""

    def test_register_and_use_basic_custom_step(self):
        """Test basic registration and usage of a custom step in pipeline."""
        def add_value(df, column, value):
            df = df.copy()
            df[column] = df[column] + value
            return df

        ar.register_step("test_add_value", add_value)
        frame = ar.from_pandas(pd.DataFrame({"x": [1, 2, 3]}))
        result = ar.pipeline(frame, [("test_add_value", {"column": "x", "value": 10})])
        df = ar.to_pandas(result)
        assert df["x"].tolist() == [11, 12, 13]

    def test_custom_step_with_default_parameters(self):
        """Custom step with default kwargs works correctly."""
        def multiply(df, column, factor=2):
            df = df.copy()
            df[column] = df[column] * factor
            return df

        ar.register_step("test_multiply", multiply)
        frame = ar.from_pandas(pd.DataFrame({"x": [1, 2, 3]}))
        result = ar.pipeline(frame, [("test_multiply", {"column": "x", "factor": 5})])
        df = ar.to_pandas(result)
        assert df["x"].tolist() == [5, 10, 15]

    def test_custom_step_no_parameters(self):
        """Custom step without parameters works in pipeline."""
        def add_constant(df):
            df = df.copy()
            df["const"] = 42
            return df

        ar.register_step("test_no_params", add_constant)
        frame = ar.from_pandas(pd.DataFrame({"x": [1, 2]}))
        result = ar.pipeline(frame, [("test_no_params",)])
        df = ar.to_pandas(result)
        assert "const" in df.columns
        assert (df["const"] == 42).all()

    def test_custom_step_in_multi_step_pipeline(self):
        """Custom step works correctly in pipeline with built-in steps."""
        def double_values(df):
            df = df.copy()
            df["x"] = df["x"] * 2
            return df

        ar.register_step("test_double", double_values)
        frame = ar.from_pandas(pd.DataFrame({"x": [1.0, 2.0, None]}))
        result = ar.pipeline(frame, [
            ("test_double",),
            ("drop_nulls",),
        ])
        df = ar.to_pandas(result)
        assert len(df) == 2
        assert df["x"].tolist() == [2.0, 4.0]

    def test_overwrite_custom_step_with_flag_enabled(self):
        """Can overwrite existing custom Python step when overwrite=True."""
        def version_one(df):
            df = df.copy()
            df["version"] = 1
            return df

        ar.register_step("test_ow_enabled", version_one)

        def version_two(df):
            df = df.copy()
            df["version"] = 2
            return df

        ar.register_step("test_ow_enabled", version_two, overwrite=True)
        
        frame = ar.from_pandas(pd.DataFrame({"x": [1]}))
        result = ar.pipeline(frame, [("test_ow_enabled",)])
        df = ar.to_pandas(result)
        assert df["version"].iloc[0] == 2

    def test_prevent_overwrite_without_flag(self):
        """ValueError raised when overwriting custom step with overwrite=False."""
        def fn_one(df):
            return df

        ar.register_step("test_ow_disabled", fn_one)

        def fn_two(df):
            return df

        with pytest.raises(ValueError):
            ar.register_step("test_ow_disabled", fn_two, overwrite=False)

    def test_prevent_overwrite_builtin_step_even_with_flag(self):
        """ValueError raised when trying to overwrite built-in step."""
        def fake_builtin(df):
            return df

        with pytest.raises(ValueError):
            ar.register_step("drop_nulls", fake_builtin, overwrite=True)

    def test_registered_step_appears_in_list_steps(self):
        """Registered custom step appears in ar.list_steps()."""
        def custom_step(df):
            return df

        ar.register_step("test_list_check", custom_step)
        assert "test_list_check" in ar.list_steps()

    def test_namespaced_custom_step(self):
        """Namespaced step names (e.g., 'team:name') are supported."""
        def namespaced_fn(df):
            df = df.copy()
            df["ns"] = "value"
            return df

        ar.register_step("my_team:custom_step", namespaced_fn)
        frame = ar.from_pandas(pd.DataFrame({"x": [1]}))
        result = ar.pipeline(frame, [("my_team:custom_step",)])
        df = ar.to_pandas(result)
        assert df["ns"].iloc[0] == "value"


class TestResetSteps:
    """Tests for ar.reset_steps()"""

    def test_reset_removes_all_custom_steps(self):
        """reset_steps() removes all registered custom steps."""
        ar.register_step("test_reset_1", lambda df: df)
        ar.register_step("test_reset_2", lambda df: df)
        ar.register_step("test_reset_3", lambda df: df)
        
        ar.reset_steps()
        
        steps = ar.list_steps()
        assert "test_reset_1" not in steps
        assert "test_reset_2" not in steps
        assert "test_reset_3" not in steps

    def test_reset_preserves_builtin_steps(self):
        """Built-in steps remain available after reset_steps()."""
        ar.register_step("test_before_reset", lambda df: df)
        ar.reset_steps()
        
        steps = ar.list_steps()
        # Verify some known built-in steps are preserved
        assert "drop_nulls" in steps
        assert "strip_whitespace" in steps
        assert "drop_duplicates" in steps
        assert "fill_nulls" in steps

    def test_reset_steps_idempotent(self):
        """Calling reset_steps() multiple times is safe."""
        ar.register_step("temp_step", lambda df: df)
        ar.reset_steps()
        ar.reset_steps()  # Should not raise
        ar.reset_steps()  # Should not raise
        assert "temp_step" not in ar.list_steps()

    def test_reset_allows_reuse_of_step_name(self):
        """After reset, can register a step with a previously used name."""
        def step_version_one(df):
            df = df.copy()
            df["ver"] = 1
            return df

        ar.register_step("test_reuse", step_version_one)
        ar.reset_steps()

        def step_version_two(df):
            df = df.copy()
            df["ver"] = 2
            return df

        # Should succeed without overwrite=True after reset
        ar.register_step("test_reuse", step_version_two)
        
        frame = ar.from_pandas(pd.DataFrame({"x": [1]}))
        result = ar.pipeline(frame, [("test_reuse",)])
        df = ar.to_pandas(result)
        assert df["ver"].iloc[0] == 2
