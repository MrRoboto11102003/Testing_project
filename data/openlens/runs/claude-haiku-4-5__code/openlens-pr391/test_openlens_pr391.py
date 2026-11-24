import pytest
import math
from openlens.validation import safe_float_conversion, validate_conic_constant, ValidationError


class TestSafeFloatConversion:
    """Test safe_float_conversion function"""
    
    def test_valid_int(self):
        """Test conversion of valid integer"""
        success, value = safe_float_conversion(42)
        assert success is True
        assert value == 42.0
    
    def test_valid_float(self):
        """Test conversion of valid float"""
        success, value = safe_float_conversion(3.14)
        assert success is True
        assert value == 3.14
    
    def test_valid_string(self):
        """Test conversion of valid numeric string"""
        success, value = safe_float_conversion("123.45")
        assert success is True
        assert value == 123.45
    
    def test_boolean_rejection(self):
        """Test that booleans are explicitly rejected despite int subclass"""
        for bool_val in [True, False]:
            success, value = safe_float_conversion(bool_val)
            assert success is False
            assert value == 0.0
    
    def test_nan_inf_rejection(self):
        """Test that NaN and infinity are rejected"""
        for nan_inf in [float('nan'), float('inf'), float('-inf')]:
            success, value = safe_float_conversion(nan_inf)
            assert success is False
            assert value == 0.0
    
    def test_invalid_input_types(self):
        """Test that invalid types fail gracefully without raising"""
        invalid_inputs = ["not_a_number", None, [1, 2], {"key": "value"}]
        for invalid in invalid_inputs:
            success, value = safe_float_conversion(invalid)
            assert success is False
            assert value == 0.0
    
    def test_large_int_overflow(self):
        """Test that very large integers that overflow are handled"""
        success, value = safe_float_conversion(10**400)
        assert success is False
        assert value == 0.0
    
    def test_custom_default_value(self):
        """Test that custom default value is used on failure"""
        success, value = safe_float_conversion("invalid", default=99.5)
        assert success is False
        assert value == 99.5
    
    def test_string_scientific_notation(self):
        """Test scientific notation in string"""
        success, value = safe_float_conversion("1.5e-3")
        assert success is True
        assert abs(value - 0.0015) < 1e-10
    
    def test_string_with_whitespace(self):
        """Test string with leading/trailing whitespace is handled"""
        success, value = safe_float_conversion("  -42.5  ")
        assert success is True
        assert value == -42.5


class TestValidateConicConstant:
    """Test validate_conic_constant function"""
    
    def test_valid_typical_values(self):
        """Test valid conic constants including hyperbolic range"""
        for val in [0.0, -1.5, 5.0, -100.0, 100.0]:
            result = validate_conic_constant(val)
            assert result == val
    
    def test_out_of_bounds_raises_error(self):
        """Test that values outside [-100, 100] raise ValidationError"""
        for val in [-100.1, 100.1, -200.0, 200.0]:
            with pytest.raises(ValidationError, match="must be between -100 and 100"):
                validate_conic_constant(val)
    
    def test_non_finite_values_raise_error(self):
        """Test that NaN and infinity raise ValidationError"""
        for val in [float('nan'), float('inf'), float('-inf')]:
            with pytest.raises(ValidationError):
                validate_conic_constant(val)
    
    def test_custom_param_name_in_error_message(self):
        """Test that custom param_name appears in error messages"""
        with pytest.raises(ValidationError, match="my_param"):
            validate_conic_constant(200.0, param_name="my_param")
