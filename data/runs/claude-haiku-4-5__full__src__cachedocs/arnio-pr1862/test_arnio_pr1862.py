import pytest
import pandas as pd
import arnio as ar


class TestRegisterStep:
    """Tests for ar.register_step()"""

    def setup_method(self):
        """Reset step registry before each test."""
        ar.reset_steps()

    def teardown_method(self):
        """Clean up after each test."""
        ar.reset_steps()

    def test_register_and_use_custom_step(self):
        """Test basic custom step registration and usage in pipeline."""
        def add_constant(df, value=10):
            df = df.copy()
            df["constant"] = value
            return df

        ar.register_step("add_constant", add_constant)

        frame = ar.from_pandas(pd.DataFrame({"x": [1, 2, 3]}))
        result = ar.pipeline(frame, [("add_constant", {"value": 42})])
        df = ar.to_pandas(result)

        assert "constant" in df.columns
        assert (df["constant"] == 42).all()

    def test_register_step_empty_name_raises_value_error(self):
        """Test that empty string name raises ValueError."""
        def dummy(df):
            return df

        with pytest.raises(ValueError, match="non-empty string"):
            ar.register_step("", dummy)

    def test_register_step_whitespace_name_raises_value_error(self):
        """Test that whitespace-only name raises ValueError."""
        def dummy(df):
            return df

        with pytest.raises(ValueError, match="non-empty string"):
            ar.register_step("   ", dummy)

    def test_register_step_non_callable_raises_type_error(self):
        """Test that non-callable fn raises TypeError."""
        with pytest.raises(TypeError, match="callable"):
            ar.register_step("bad_step", "not callable")

    def test_register_step_overwrite_not_bool_raises_type_error(self):
        """Test that overwrite parameter must be bool."""
        def dummy(df):
            return df

        with pytest.raises(TypeError, match="bool"):
            ar.register_step("test_step", dummy, overwrite=1)

    def test_register_duplicate_without_overwrite_raises_value_error(self):
        """Test duplicate registration without overwrite raises ValueError."""
        def step_v1(df):
            return df

        def step_v2(df):
            return df

        ar.register_step("my_step", step_v1)

        with pytest.raises(ValueError, match="already registered"):
            ar.register_step("my_step", step_v2, overwrite=False)

    def test_register_duplicate_with_overwrite_succeeds(self):
        """Test that overwrite=True allows replacing custom steps."""
        def step_v1(df):
            df = df.copy()
            df["version"] = 1
            return df

        def step_v2(df):
            df = df.copy()
            df["version"] = 2
            return df

        ar.register_step("versioned", step_v1)
        ar.register_step("versioned", step_v2, overwrite=True)

        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        result = ar.pipeline(frame, [("versioned",)])
        df = ar.to_pandas(result)

        assert df["version"].iloc[0] == 2

    def test_register_builtin_prefix_reserved_raises_value_error(self):
        """Test that 'builtin:' prefix is reserved."""
        def dummy(df):
            return df

        with pytest.raises(ValueError, match="reserved"):
            ar.register_step("builtin:custom", dummy)

    def test_register_builtin_cpp_step_name_raises_value_error(self):
        """Test that built-in C++ step names cannot be overwritten."""
        def dummy(df):
            return df

        with pytest.raises(ValueError, match="conflicts with built-in"):
            ar.register_step("drop_nulls", dummy)

    def test_register_multiple_custom_steps(self):
        """Test registering and using multiple custom steps in sequence."""
        def step1(df):
            df = df.copy()
            df["s1"] = 1
            return df

        def step2(df):
            df = df.copy()
            df["s2"] = 2
            return df

        ar.register_step("step1", step1)
        ar.register_step("step2", step2)

        frame = ar.from_pandas(pd.DataFrame({"x": [1]}))
        result = ar.pipeline(frame, [("step1",), ("step2",)])
        df = ar.to_pandas(result)

        assert "s1" in df.columns and "s2" in df.columns


class TestResetSteps:
    """Tests for ar.reset_steps()"""

    def test_reset_steps_removes_custom_steps(self):
        """Test that reset_steps clears all custom steps."""
        def custom(df):
            df = df.copy()
            df["marker"] = True
            return df

        ar.register_step("custom_step", custom)

        # Verify it works before reset
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        result = ar.pipeline(frame, [("custom_step",)])
        assert "marker" in ar.to_pandas(result).columns

        # After reset, step should be gone
        ar.reset_steps()

        with pytest.raises(Exception):  # UnknownStepError
            ar.pipeline(frame, [("custom_step",)])

    def test_reset_steps_preserves_builtin_steps(self):
        """Test that reset_steps leaves built-in steps intact."""
        def custom(df):
            return df

        ar.register_step("custom", custom)

        # Verify built-in step works with custom registered
        frame = ar.from_pandas(pd.DataFrame({"a": [1, None, 3]}))
        result1 = ar.pipeline(frame, [("drop_nulls",)])
        assert len(ar.to_pandas(result1)) == 2

        # After reset, built-in should still work
        ar.reset_steps()
        result2 = ar.pipeline(frame, [("drop_nulls",)])
        assert len(ar.to_pandas(result2)) == 2

    def test_reset_steps_clears_multiple_custom_steps(self):
        """Test that reset_steps clears all registered custom steps."""
        for i in range(3):
            ar.register_step(f"custom{i}", lambda df: df)

        ar.reset_steps()

        frame = ar.from_pandas(pd.DataFrame({"x": [1]}))
        for i in range(3):
            with pytest.raises(Exception):
                ar.pipeline(frame, [(f"custom{i}",)])

    def test_reset_steps_returns_none(self):
        """Test that reset_steps returns None."""
        result = ar.reset_steps()
        assert result is None

    def test_can_reregister_after_reset(self):
        """Test that steps can be re-registered with same name after reset."""
        def step_v1(df):
            df = df.copy()
            df["version"] = 1
            return df

        ar.register_step("reusable", step_v1)
        ar.reset_steps()

        def step_v2(df):
            df = df.copy()
            df["version"] = 2
            return df

        # Should register without error after reset
        ar.register_step("reusable", step_v2)

        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        result = ar.pipeline(frame, [("reusable",)])
        df = ar.to_pandas(result)

        assert df["version"].iloc[0] == 2
