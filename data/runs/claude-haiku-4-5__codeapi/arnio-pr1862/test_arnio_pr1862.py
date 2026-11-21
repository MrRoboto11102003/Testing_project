import pytest
import arnio as ar
from arnio.pipeline import register_step, reset_steps, list_steps, unregister_step


class TestRegisterStep:
    """Tests for arnio.pipeline.register_step"""

    def setup_method(self):
        """Reset steps before each test to ensure clean state"""
        reset_steps()

    def teardown_method(self):
        """Clean up after each test"""
        reset_steps()

    def test_register_simple_custom_step(self):
        """Test basic registration of a custom step"""
        def my_step(df, **kwargs):
            return df
        
        register_step("my_custom_step", my_step)
        assert "my_custom_step" in list_steps()

    def test_register_multiple_custom_steps(self):
        """Test registering multiple different custom steps"""
        def step1(df, **kwargs):
            return df
        
        def step2(df, **kwargs):
            return df
        
        register_step("step1", step1)
        register_step("step2", step2)
        
        steps = list_steps()
        assert "step1" in steps
        assert "step2" in steps

    def test_register_step_name_must_be_nonempty_string(self):
        """Test that name must be a non-empty string"""
        def dummy_step(df, **kwargs):
            return df
        
        with pytest.raises(ValueError):
            register_step("", dummy_step)
        
        with pytest.raises(ValueError):
            register_step("   ", dummy_step)

    def test_register_step_name_not_string_type(self):
        """Test that name must be string type"""
        def dummy_step(df, **kwargs):
            return df
        
        with pytest.raises(ValueError):
            register_step(123, dummy_step)

    def test_register_step_fn_must_be_callable(self):
        """Test that fn parameter must be callable"""
        with pytest.raises(TypeError):
            register_step("my_step", "not_callable")
        
        with pytest.raises(TypeError):
            register_step("my_step", 42)

    def test_register_step_overwrite_must_be_bool(self):
        """Test that overwrite parameter must be a boolean"""
        def dummy_step(df, **kwargs):
            return df
        
        with pytest.raises(TypeError):
            register_step("my_step", dummy_step, overwrite="True")
        
        with pytest.raises(TypeError):
            register_step("my_step", dummy_step, overwrite=1)

    def test_register_step_cannot_overwrite_builtin_cpp(self):
        """Test that built-in C++ steps cannot be overwritten"""
        def dummy_step(df, **kwargs):
            return df
        
        with pytest.raises(ValueError, match="conflicts with built-in C\\+\\+ step"):
            register_step("drop_nulls", dummy_step)

    def test_register_step_overwrite_false_prevents_replacement(self):
        """Test that overwrite=False prevents replacing a custom step"""
        def original_step(df, **kwargs):
            return df
        
        def replacement_step(df, **kwargs):
            return df
        
        register_step("my_custom_step", original_step)
        
        with pytest.raises(ValueError, match="already registered"):
            register_step("my_custom_step", replacement_step, overwrite=False)

    def test_register_step_overwrite_true_allows_replacement(self):
        """Test that overwrite=True allows replacing a custom step"""
        def original_step(df, **kwargs):
            return df
        
        def replacement_step(df, **kwargs):
            return df
        
        register_step("my_custom_step", original_step)
        register_step("my_custom_step", replacement_step, overwrite=True)
        
        assert "my_custom_step" in list_steps()

    def test_register_step_with_lambda(self):
        """Test that lambda functions can be registered"""
        register_step("lambda_step", lambda df, **kwargs: df)
        assert "lambda_step" in list_steps()

    def test_register_step_default_overwrite_is_false(self):
        """Test that default overwrite value is False"""
        def step1(df, **kwargs):
            return df
        
        def step2(df, **kwargs):
            return df
        
        register_step("step", step1)
        
        with pytest.raises(ValueError):
            register_step("step", step2)  # overwrite defaults to False


class TestResetSteps:
    """Tests for arnio.pipeline.reset_steps"""

    def test_reset_steps_removes_custom_steps(self):
        """Test that reset_steps removes all custom steps"""
        def custom_step(df, **kwargs):
            return df
        
        register_step("custom_step", custom_step)
        assert "custom_step" in list_steps()
        
        reset_steps()
        
        assert "custom_step" not in list_steps()

    def test_reset_steps_preserves_builtin_steps(self):
        """Test that reset_steps preserves built-in steps"""
        # Get baseline of built-in steps
        reset_steps()
        builtin_before = set(list_steps())
        
        # Register and reset
        def custom_step(df, **kwargs):
            return df
        register_step("custom_step", custom_step)
        reset_steps()
        
        builtin_after = set(list_steps())
        
        # Built-in steps should be preserved
        assert builtin_before == builtin_after
        assert len(builtin_after) > 0

    def test_reset_steps_idempotent(self):
        """Test that reset_steps is idempotent"""
        reset_steps()
        steps_1 = list_steps()
        
        reset_steps()
        steps_2 = list_steps()
        
        assert steps_1 == steps_2

    def test_reset_steps_with_multiple_custom_steps(self):
        """Test reset_steps removes all custom steps at once"""
        def step_a(df, **kwargs):
            return df
        
        def step_b(df, **kwargs):
            return df
        
        def step_c(df, **kwargs):
            return df
        
        register_step("step_a", step_a)
        register_step("step_b", step_b)
        register_step("step_c", step_c)
        
        reset_steps()
        
        current = list_steps()
        assert "step_a" not in current
        assert "step_b" not in current
        assert "step_c" not in current

    def test_reset_steps_allows_reregistration(self):
        """Test that after reset, we can re-register with same name"""
        def my_step(df, **kwargs):
            return df
        
        register_step("my_step", my_step)
        reset_steps()
        register_step("my_step", my_step)
        
        assert "my_step" in list_steps()


class TestRegisterAndResetInteraction:
    """Integration tests for register_step and reset_steps"""

    def setup_method(self):
        reset_steps()

    def teardown_method(self):
        reset_steps()

    def test_register_reset_register_cycle(self):
        """Test complete register -> reset -> register cycle"""
        def step(df, **kwargs):
            return df
        
        register_step("step", step)
        assert "step" in list_steps()
        
        reset_steps()
        assert "step" not in list_steps()
        
        register_step("step", step)
        assert "step" in list_steps()

    def test_multiple_steps_survive_reset_individually(self):
        """Test that multiple custom steps all removed by single reset"""
        names = ["test_a", "test_b", "test_c", "test_d"]
        
        for name in names:
            def dummy(df, **kwargs):
                return df
            register_step(name, dummy)
        
        all_present = all(name in list_steps() for name in names)
        assert all_present
        
        reset_steps()
        
        all_gone = all(name not in list_steps() for name in names)
        assert all_gone
