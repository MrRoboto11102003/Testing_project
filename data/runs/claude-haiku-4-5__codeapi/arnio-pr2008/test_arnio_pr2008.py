import pytest
import arnio as ar
from arnio.exceptions import UnknownStepError
from arnio.pipeline import register_step, unregister_step, list_steps, reset_steps


class TestUnregisterStep:
    """Test suite for arnio.pipeline.unregister_step"""

    def setup_method(self):
        """Reset pipeline registry before each test"""
        reset_steps()

    def teardown_method(self):
        """Clean up after each test"""
        reset_steps()

    # ===== Success Cases =====

    def test_unregister_custom_step_success(self):
        """Successfully unregister a custom step that was registered"""
        def dummy_step(frame):
            return frame

        register_step("my_custom_step", dummy_step)
        assert "my_custom_step" in list_steps()

        unregister_step("my_custom_step")
        assert "my_custom_step" not in list_steps()

    def test_unregister_and_reregister_same_name(self):
        """Can re-register a step name after unregistering"""
        def dummy1(frame):
            return frame

        def dummy2(frame):
            return frame

        register_step("reusable", dummy1)
        unregister_step("reusable")
        register_step("reusable", dummy2)

        assert "reusable" in list_steps()

    def test_unregister_multiple_custom_steps(self):
        """Successfully unregister multiple custom steps independently"""
        def dummy(frame):
            return frame

        register_step("step1", dummy)
        register_step("step2", dummy)
        assert "step1" in list_steps()
        assert "step2" in list_steps()

        unregister_step("step1")
        assert "step1" not in list_steps()
        assert "step2" in list_steps()

    # ===== Type Validation =====

    def test_unregister_raises_typeerror_for_int(self):
        """Raise TypeError if name is an integer"""
        with pytest.raises(TypeError, match="parameter 'name' must be a string"):
            unregister_step(123)

    def test_unregister_raises_typeerror_for_none(self):
        """Raise TypeError if name is None"""
        with pytest.raises(TypeError, match="parameter 'name' must be a string"):
            unregister_step(None)

    def test_unregister_raises_typeerror_for_list(self):
        """Raise TypeError if name is a list"""
        with pytest.raises(TypeError, match="parameter 'name' must be a string"):
            unregister_step(["step"])

    def test_unregister_raises_typeerror_for_dict(self):
        """Raise TypeError if name is a dict"""
        with pytest.raises(TypeError, match="parameter 'name' must be a string"):
            unregister_step({"name": "step"})

    # ===== Empty String Validation =====

    def test_unregister_raises_valueerror_for_empty_string(self):
        """Raise ValueError if name is empty string"""
        with pytest.raises(ValueError, match="parameter 'name' must be a non-empty string"):
            unregister_step("")

    # ===== Unknown Step =====

    def test_unregister_raises_unknown_step_error_for_nonexistent(self):
        """Raise UnknownStepError if step doesn't exist in registry"""
        with pytest.raises(UnknownStepError):
            unregister_step("nonexistent_step_xyz_123")

    def test_unregister_same_step_twice_raises_error(self):
        """Second unregister of same step raises UnknownStepError"""
        def dummy(frame):
            return frame

        register_step("custom", dummy)
        unregister_step("custom")

        with pytest.raises(UnknownStepError):
            unregister_step("custom")

    # ===== Built-in Step Protection =====

    def test_unregister_raises_error_for_builtin_drop_nulls(self):
        """Raise ValueError when trying to unregister built-in step"""
        with pytest.raises(ValueError, match="Cannot unregister built-in step"):
            unregister_step("drop_nulls")

    def test_unregister_raises_error_for_builtin_strip_whitespace(self):
        """Raise ValueError for built-in strip_whitespace"""
        with pytest.raises(ValueError, match="Cannot unregister built-in step"):
            unregister_step("strip_whitespace")

    def test_unregister_raises_error_for_builtin_fill_nulls(self):
        """Raise ValueError for built-in fill_nulls"""
        with pytest.raises(ValueError, match="Cannot unregister built-in step"):
            unregister_step("fill_nulls")

    # ===== Case Sensitivity =====

    def test_unregister_step_names_are_case_sensitive(self):
        """Step names are case-sensitive"""
        def dummy(frame):
            return frame

        register_step("MyStep", dummy)

        # Different case should not find it
        with pytest.raises(UnknownStepError):
            unregister_step("mystep")

        # Original case should work
        unregister_step("MyStep")
        assert "MyStep" not in list_steps()

    # ===== Special Characters in Names =====

    def test_unregister_step_with_underscores(self):
        """Step name with underscores can be unregistered"""
        def dummy(frame):
            return frame

        register_step("my_custom_step_v2", dummy)
        unregister_step("my_custom_step_v2")
        assert "my_custom_step_v2" not in list_steps()

    def test_unregister_step_with_hyphens(self):
        """Step name with hyphens can be unregistered"""
        def dummy(frame):
            return frame

        register_step("my-custom-step", dummy)
        unregister_step("my-custom-step")
        assert "my-custom-step" not in list_steps()

    def test_unregister_numeric_string_name(self):
        """Step name that is a numeric string can be unregistered"""
        def dummy(frame):
            return frame

        register_step("12345", dummy)
        unregister_step("12345")
        assert "12345" not in list_steps()

    # ===== Registry State Verification =====

    def test_unregister_removes_from_list_steps(self):
        """Unregistered step is no longer in list_steps()"""
        initial_steps = set(list_steps())

        def dummy(frame):
            return frame

        register_step("temp_step", dummy)
        unregister_step("temp_step")

        # Should match initial state
        assert set(list_steps()) == initial_steps

    def test_unregister_does_not_affect_other_steps(self):
        """Unregistering one step doesn't affect others"""
        def dummy(frame):
            return frame

        register_step("keep_me", dummy)
        register_step("remove_me", dummy)

        unregister_step("remove_me")

        assert "keep_me" in list_steps()
        assert "remove_me" not in list_steps()
