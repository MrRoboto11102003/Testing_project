import pytest
import re
import pandas as pd
import numpy as np
import arnio as ar


class TestStringBasics:
    """Test basic String field creation and properties."""
    
    def test_string_default_parameters(self):
        """Test String with default parameters."""
        field = ar.String()
        assert field.dtype == "string"
        assert field.nullable is True
        assert field.pattern is None
        assert field.allowed is None
        assert field.case_sensitive is True
        assert field.unique is False
        assert field.severity == "error"
        assert field.min_length is None
        assert field.max_length is None
        assert field.required_if is None
    
    def test_string_equality_with_field(self):
        """Test that String() equals Field(dtype='string')."""
        assert ar.String() == ar.Field(dtype="string")
    
    def test_string_no_positional_arguments(self):
        """Test that String rejects positional arguments."""
        with pytest.raises(TypeError):
            ar.String("x")
    
    def test_string_returns_frozen_field(self):
        """Test that returned Field is frozen and immutable."""
        field = ar.String()
        with pytest.raises(Exception):  # FrozenInstanceError
            field.nullable = False


class TestStringLength:
    """Test min_length and max_length parameters."""
    
    def test_min_length_zero(self):
        """Test that min_length=0 is allowed."""
        field = ar.String(min_length=0)
        assert field.min_length == 0
    
    def test_min_length_positive(self):
        """Test valid positive min_length."""
        field = ar.String(min_length=5)
        assert field.min_length == 5
    
    def test_min_length_numpy_int_converted(self):
        """Test that numpy int is accepted and converted to Python int."""
        field = ar.String(min_length=np.int64(3))
        assert field.min_length == 3
        assert isinstance(field.min_length, int)
    
    def test_min_length_bool_rejected(self):
        """Test that bool value for min_length raises TypeError."""
        with pytest.raises(TypeError, match="min_length must be an integer"):
            ar.String(min_length=True)
    
    def test_min_length_float_rejected(self):
        """Test that float value for min_length raises TypeError."""
        with pytest.raises(TypeError, match="min_length must be an integer"):
            ar.String(min_length=1.5)
    
    def test_min_length_negative_rejected(self):
        """Test that negative min_length raises ValueError."""
        with pytest.raises(ValueError, match="min_length must be greater than or equal to 0"):
            ar.String(min_length=-1)
    
    def test_max_length_positive(self):
        """Test valid positive max_length."""
        field = ar.String(max_length=10)
        assert field.max_length == 10
    
    def test_max_length_bool_rejected(self):
        """Test that bool value for max_length raises TypeError."""
        with pytest.raises(TypeError, match="max_length must be an integer"):
            ar.String(max_length=False)
    
    def test_max_length_negative_rejected(self):
        """Test that negative max_length raises ValueError."""
        with pytest.raises(ValueError, match="max_length must be greater than or equal to 0"):
            ar.String(max_length=-1)
    
    def test_min_length_greater_than_max_rejected(self):
        """Test that min_length > max_length raises ValueError."""
        with pytest.raises(ValueError, match="min_length must be less than or equal to max_length"):
            ar.String(min_length=5, max_length=2)
    
    def test_min_length_equal_max_length_allowed(self):
        """Test that min_length == max_length is allowed."""
        field = ar.String(min_length=3, max_length=3)
        assert field.min_length == 3
        assert field.max_length == 3


class TestStringAllowed:
    """Test allowed parameter."""
    
    def test_allowed_list(self):
        """Test allowed as a list."""
        field = ar.String(allowed=["a", "b", "c"])
        assert field.allowed == {"a", "b", "c"}
    
    def test_allowed_tuple(self):
        """Test allowed as a tuple."""
        field = ar.String(allowed=("x", "y"))
        assert field.allowed == {"x", "y"}
    
    def test_allowed_set(self):
        """Test allowed as a set."""
        field = ar.String(allowed={"p", "q"})
        assert field.allowed == {"p", "q"}
    
    def test_allowed_duplicates_collapse(self):
        """Test that duplicate values in allowed collapse to set."""
        field = ar.String(allowed=["a", "a", "b"])
        assert field.allowed == {"a", "b"}
    
    def test_allowed_empty_list_becomes_empty_set(self):
        """Test that allowed=[] creates empty set, not None."""
        field = ar.String(allowed=[])
        assert field.allowed == set()
        assert field.allowed is not None
    
    def test_allowed_string_rejected(self):
        """Test that string value for allowed raises TypeError."""
        with pytest.raises(TypeError, match="allowed must be a sequence of allowed values, not a bare string"):
            ar.String(allowed="abc")
    
    def test_allowed_bytes_rejected(self):
        """Test that bytes value for allowed raises TypeError."""
        with pytest.raises(TypeError, match="allowed must be a sequence of allowed values, not a bare string"):
            ar.String(allowed=b"x")
    
    def test_allowed_non_iterable_rejected(self):
        """Test that non-iterable value for allowed raises TypeError."""
        with pytest.raises(TypeError):
            ar.String(allowed=5)
    
    def test_allowed_unhashable_items_rejected(self):
        """Test that unhashable items in allowed raise TypeError."""
        with pytest.raises(TypeError):
            ar.String(allowed=[[1]])


class TestStringPattern:
    """Test pattern parameter."""
    
    def test_pattern_valid_simple(self):
        """Test valid regex pattern is stored."""
        field = ar.String(pattern=r"^\d+$")
        assert field.pattern == r"^\d+$"
    
    def test_pattern_non_string_rejected(self):
        """Test that non-string pattern raises TypeError."""
        with pytest.raises(TypeError, match="pattern must be a string or None"):
            ar.String(pattern=123)
    
    def test_pattern_invalid_regex_rejected(self):
        """Test that invalid regex pattern raises ValueError."""
        with pytest.raises(ValueError, match=re.escape("Invalid regex pattern: '('")):
            ar.String(pattern="(")
    
    def test_pattern_unsafe_nested_quantifier_plus_plus_rejected(self):
        """Test that (a+)+ is rejected as unsafe."""
        with pytest.raises(ValueError, match="Unsafe regex pattern rejected"):
            ar.String(pattern="(a+)+")
    
    def test_pattern_unsafe_nested_quantifier_star_star_rejected(self):
        """Test that (a*)* is rejected as unsafe."""
        with pytest.raises(ValueError, match="Unsafe regex pattern rejected"):
            ar.String(pattern="(a*)*")
    
    def test_pattern_safe_fixed_repetition_allowed(self):
        """Test that (ab)+ with no internal quantifier is safe."""
        field = ar.String(pattern="(ab)+")
        assert field.pattern == "(ab)+"
    
    def test_pattern_safe_fixed_count_allowed(self):
        """Test that (\\d{3})+ with fixed count is safe."""
        field = ar.String(pattern=r"(\d{3})+")
        assert field.pattern == r"(\d{3})+"


class TestStringOtherParameters:
    """Test other parameters like nullable, unique, case_sensitive, severity."""
    
    def test_nullable_false(self):
        """Test nullable=False."""
        field = ar.String(nullable=False)
        assert field.nullable is False
    
    def test_nullable_non_bool_rejected(self):
        """Test that non-bool value for nullable raises TypeError."""
        with pytest.raises(TypeError, match="nullable must be a bool"):
            ar.String(nullable=1)
    
    def test_case_sensitive_false(self):
        """Test case_sensitive=False."""
        field = ar.String(case_sensitive=False)
        assert field.case_sensitive is False
    
    def test_case_sensitive_non_bool_rejected(self):
        """Test that non-bool value for case_sensitive raises TypeError."""
        with pytest.raises(TypeError, match="case_sensitive must be a bool"):
            ar.String(case_sensitive=None)
    
    def test_unique_true(self):
        """Test unique=True."""
        field = ar.String(unique=True)
        assert field.unique is True
    
    def test_unique_non_bool_rejected(self):
        """Test that non-bool value for unique raises TypeError."""
        with pytest.raises(TypeError, match="unique must be a bool"):
            ar.String(unique="y")
    
    def test_severity_warning(self):
        """Test severity='warning'."""
        field = ar.String(severity="warning")
        assert field.severity == "warning"
    
    def test_severity_invalid_rejected(self):
        """Test that invalid severity value raises ValueError."""
        with pytest.raises(ValueError, match="severity must be 'error' or 'warning'"):
            ar.String(severity="info")


class TestStringRequiredIf:
    """Test required_if parameter."""
    
    def test_required_if_valid_tuple(self):
        """Test valid required_if tuple."""
        field = ar.String(required_if=("flag", "Y"))
        assert field.required_if == ("flag", "Y")
    
    def test_required_if_non_tuple_rejected(self):
        """Test that list for required_if raises TypeError."""
        with pytest.raises(TypeError, match="required_if must be a tuple or None"):
            ar.String(required_if=["a", 1])
    
    def test_required_if_wrong_length_rejected(self):
        """Test that single-element tuple for required_if raises TypeError."""
        with pytest.raises(TypeError, match="required_if must be a \\(column_name, expected_value\\) tuple"):
            ar.String(required_if=("a",))
    
    def test_required_if_non_string_column_rejected(self):
        """Test that non-string column name in required_if raises TypeError."""
        with pytest.raises(TypeError, match="required_if column name must be a string"):
            ar.String(required_if=(1, "x"))


class TestStringComplexCombinations:
    """Test combinations and complex scenarios."""
    
    def test_all_parameters_set(self):
        """Test String with all parameters specified."""
        field = ar.String(
            nullable=False,
            pattern=r"^\w+$",
            allowed=["hello", "world"],
            case_sensitive=False,
            unique=True,
            severity="warning",
            min_length=1,
            max_length=50,
            required_if=("active", True)
        )
        assert field.dtype == "string"
        assert field.nullable is False
        assert field.pattern == r"^\w+$"
        assert field.allowed == {"hello", "world"}
        assert field.case_sensitive is False
        assert field.unique is True
        assert field.severity == "warning"
        assert field.min_length == 1
        assert field.max_length == 50
        assert field.required_if == ("active", True)
    
    def test_error_precedence_min_length_before_pattern(self):
        """Test that min_length validation error occurs before pattern error."""
        with pytest.raises(ValueError, match="min_length must be greater than or equal to 0"):
            ar.String(min_length=-1, pattern="(")
    
    def test_field_has_expected_none_attributes(self):
        """Test that Field attributes are None as documented."""
        field = ar.String()
        assert field.min is None
        assert field.max is None
        assert field.semantic is None
        assert field.format is None
