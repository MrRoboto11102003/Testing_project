import pytest
import pandas as pd
import arnio as ar


class TestRegisterStep:
    """Tests for ar.register_step()"""

    def setup_method(self):
        """Reset steps before each test"""
        ar.reset_steps()

    def teardown_method(self):
        """Clean up after each test"""
        ar.reset_steps()

    def test_register_simple_custom_step(self):
        """Test basic registration of a custom step"""
        def my_step(df):
            return df.copy()

        ar.register_step("my_custom_step", my_step)
        steps = ar.list_steps()
        assert "my_custom_step" in steps

    def test_registered_step_works_in_pipeline(self):
        """Test that a registered step can be used in a pipeline"""
        def double_numeric(df, subset=None):
            df = df.copy()
            cols = subset if subset else df.select_dtypes('number').columns
            for col in cols:
                df[col] = df[col] * 2
            return df

        ar.register_step("double_numeric", double_numeric)

        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
        result = ar.pipeline(frame, [("double_numeric",)])
        result_df = ar.to_pandas(result)

        assert result_df["a"].tolist() == [2, 4, 6]
        assert result_df["b"].tolist() == [8, 10, 12]

    def test_register_step_with_kwargs(self):
        """Test that registered step can accept and use kwargs"""
        def add_constant(df, value=10, subset=None):
            df = df.copy()
            cols = subset if subset else df.select_dtypes('number').columns
            for col in cols:
                df[col] = df[col] + value
            return df

        ar.register_step("add_constant", add_constant)

        frame = ar.from_pandas(pd.DataFrame({"x": [1, 2, 3]}))
        result = ar.pipeline(frame, [("add_constant", {"value": 5})])
        result_df = ar.to_pandas(result)

        assert result_df["x"].tolist() == [6, 7, 8]

    def test_duplicate_custom_step_raises_without_overwrite(self):
        """Test registering duplicate custom step raises ValueError"""
        def step_v1(df):
            return df.copy()

        ar.register_step("my_step", step_v1)

        def step_v2(df):
            return df.copy()

        with pytest.raises(ValueError):
            ar.register_step("my_step", step_v2, overwrite=False)

    def test_overwrite_custom_step(self):
        """Test that overwrite=True replaces existing custom step"""
        def step_v1(df):
            df = df.copy()
            df["version"] = 1
            return df

        ar.register_step("versioned_step", step_v1)

        def step_v2(df):
            df = df.copy()
            df["version"] = 2
            return df

        ar.register_step("versioned_step", step_v2, overwrite=True)

        frame = ar.from_pandas(pd.DataFrame({"x": [1, 2, 3]}))
        result = ar.pipeline(frame, [("versioned_step",)])
        result_df = ar.to_pandas(result)

        assert result_df["version"].iloc[0] == 2

    def test_cannot_overwrite_builtin_step(self):
        """Test that built-in steps cannot be overwritten"""
        builtin_steps = ar.list_steps()
        builtin_name = "drop_nulls"

        if builtin_name in builtin_steps:
            def fake_step(df):
                return df.copy()

            with pytest.raises(ValueError):
                ar.register_step(builtin_name, fake_step, overwrite=True)

    def test_cannot_register_with_builtin_name(self):
        """Test that registering with built-in step name raises ValueError"""
        builtin_steps = ar.list_steps()
        builtin_name = "drop_nulls"

        if builtin_name in builtin_steps:
            def my_func(df):
                return df.copy()

            with pytest.raises(ValueError):
                ar.register_step(builtin_name, my_func)

    def test_register_multiple_custom_steps(self):
        """Test registering multiple different custom steps"""
        ar.register_step("step1", lambda df: df.copy())
        ar.register_step("step2", lambda df: df.copy())
        ar.register_step("step3", lambda df: df.copy())

        steps = ar.list_steps()
        assert "step1" in steps
        assert "step2" in steps
        assert "step3" in steps


class TestResetSteps:
    """Tests for ar.reset_steps()"""

    def setup_method(self):
        """Reset steps before each test"""
        ar.reset_steps()

    def teardown_method(self):
        """Clean up after each test"""
        ar.reset_steps()

    def test_reset_removes_custom_steps(self):
        """Test that reset_steps removes custom steps"""
        def custom_step(df):
            return df.copy()

        ar.register_step("my_custom", custom_step)
        assert "my_custom" in ar.list_steps()

        ar.reset_steps()
        assert "my_custom" not in ar.list_steps()

    def test_reset_preserves_builtin_steps(self):
        """Test that reset_steps preserves built-in steps"""
        builtin_before = set(ar.list_steps())

        ar.register_step("custom_step", lambda df: df.copy())
        ar.reset_steps()

        builtin_after = set(ar.list_steps())

        assert builtin_before == builtin_after

    def test_reset_allows_reregistering_same_name(self):
        """Test that after reset, same custom name can be registered again"""
        def step_v1(df):
            df = df.copy()
            df["version"] = 1
            return df

        ar.register_step("my_step", step_v1)
        ar.reset_steps()

        def step_v2(df):
            df = df.copy()
            df["version"] = 2
            return df

        ar.register_step("my_step", step_v2)

        frame = ar.from_pandas(pd.DataFrame({"x": [1]}))
        result = ar.pipeline(frame, [("my_step",)])
        result_df = ar.to_pandas(result)
        assert result_df["version"].iloc[0] == 2

    def test_reset_multiple_custom_steps(self):
        """Test that reset clears multiple custom steps"""
        ar.register_step("step1", lambda df: df.copy())
        ar.register_step("step2", lambda df: df.copy())
        ar.register_step("step3", lambda df: df.copy())

        ar.reset_steps()

        steps = ar.list_steps()
        assert "step1" not in steps
        assert "step2" not in steps
        assert "step3" not in steps

    def test_reset_idempotent(self):
        """Test that calling reset_steps multiple times is safe"""
        ar.register_step("custom", lambda df: df.copy())
        ar.reset_steps()
        ar.reset_steps()
        ar.reset_steps()

        assert "custom" not in ar.list_steps()

    def test_builtin_steps_work_after_reset(self):
        """Test that built-in steps are functional after reset"""
        ar.register_step("custom", lambda df: df.copy())
        ar.reset_steps()

        frame = ar.from_pandas(pd.DataFrame({
            "x": [1, 2, 3],
            "y": [" a", " b", " c"]
        }))

        result = ar.pipeline(frame, [("strip_whitespace",)])
        result_df = ar.to_pandas(result)

        assert result_df["y"].iloc[0] == "a"


class TestIntegration:
    """Integration tests combining register_step and reset_steps"""

    def setup_method(self):
        ar.reset_steps()

    def teardown_method(self):
        ar.reset_steps()

    def test_custom_and_builtin_together_in_pipeline(self):
        """Test that custom and built-in steps work together"""
        def multiply_by_two(df, subset=None):
            df = df.copy()
            cols = subset if subset else df.select_dtypes('number').columns
            for col in cols:
                df[col] = df[col] * 2
            return df

        ar.register_step("multiply_by_two", multiply_by_two)

        frame = ar.from_pandas(pd.DataFrame({
            "value": [1, 2, 3, None],
            "name": ["a", "b", "c", "d"]
        }))

        result = ar.pipeline(frame, [
            ("multiply_by_two", {"subset": ["value"]}),
            ("drop_nulls", {"subset": ["value"]}),
        ])
        result_df = ar.to_pandas(result)

        assert len(result_df) == 3
        assert list(result_df["value"]) == [2.0, 4.0, 6.0]
