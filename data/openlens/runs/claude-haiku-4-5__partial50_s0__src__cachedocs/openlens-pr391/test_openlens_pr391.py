import pytest
import math
from openlens.validation import safe_float_conversion, validate_conic_constant, ValidationError


class TestSafeFloatConversion:
    """Test safe_float_conversion function."""
    
    def test_valid_conversions(self):
        """Test valid conversions from float, int, and string."""
        assert safe_float_conversion(3.14) == (True, 3.14)
        assert safe_float_conversion(42) == (True, 42.0)
        assert safe_float_conversion("3.14") == (True, 3.14)
        assert safe_float_conversion(0) == (True, 0.0)
        assert safe_float_conversion(-5.5) == (True, -5.5)
    
    def test_boolean_rejected(self):
        """Test that booleans are explicitly rejected despite being int subclass."""
        assert safe_float_conversion(True, default=99.0) == (False, 99.0)
        assert safe_float_conversion(False, default=88.0) == (False, 88.0)
    
    def test_nan_inf_rejected(self):
        """Test that NaN and Inf values are rejected."""
        assert safe_float_conversion(float('nan'), default=1.0) == (False, 1.0)
        assert safe_float_conversion(float('inf'), default=2.0) == (False, 2.0)
        assert safe_float_conversion(float('-inf'), default=3.0) == (False, 3.0)
        assert safe_float_conversion("nan", default=4.0) == (False, 4.0)
        assert safe_float_conversion("inf", default=5.0) == (False, 5.0)
    
    def test_large_int_overflow(self):
        """Test that very large int (10**400) causes overflow and returns default."""
        success, value = safe_float_conversion(10**400, default=77.0)
        assert success is False
        assert value == 77.0
    
    def test_invalid_inputs(self):
        """Test that invalid inputs return False and default value."""
        assert safe_float_conversion("invalid", default=11.0) == (False, 11.0)
        assert safe_float_conversion(None, default=22.0) == (False, 22.0)
        assert safe_float_conversion([1, 2], default=33.0) == (False, 33.0)
    
    def test_never_raises(self):
        """Test that function never raises exceptions for any input."""
        # Function contract is to never raise
        try:
            safe_float_conversion(object())
            safe_float_conversion(lambda x: x)
            safe_float_conversion({"key": "value"})
        except Exception as e:
            pytest.fail(f"safe_float_conversion should never raise, but got {type(e).__name__}")


class TestValidateConicConstant:
    """Test validate_conic_constant function."""
    
    def test_valid_optical_values(self):
        """Test that optical values in range pass."""
        assert validate_conic_constant(0.0) == 0.0  # sphere
        assert validate_conic_constant(-0.5) == -0.5  # ellipsoid
        assert validate_conic_constant(-1.0) == -1.0  # paraboloid
        assert validate_conic_constant(-2.5) == -2.5  # hyperboloid (K < -1)
        assert validate_conic_constant(1.5) == 1.5  # oblate ellipsoid
    
    def test_boundary_values(self):
        """Test that boundary values ±100 are valid."""
        assert validate_conic_constant(-100.0) == -100.0
        assert validate_conic_constant(100.0) == 100.0
    
    def test_out_of_range_raises_error(self):
        """Test that values outside [-100, 100] raise ValidationError."""
        with pytest.raises(ValidationError):
            validate_conic_constant(-100.1)
        with pytest.raises(ValidationError):
            validate_conic_constant(100.1)
    
    def test_non_finite_raises_error(self):
        """Test that NaN and Inf raise ValidationError."""
        with pytest.raises(ValidationError):
            validate_conic_constant(float('nan'))
        with pytest.raises(ValidationError):
            validate_conic_constant(float('inf'))
        with pytest.raises(ValidationError):
            validate_conic_constant(float('-inf'))
    
    def test_error_message_includes_param_name(self):
        """Test that error messages include param_name for debugging."""
        with pytest.raises(ValidationError) as exc_info:
            validate_conic_constant(200.0, param_name="my_conic_k")
        error_msg = str(exc_info.value)
        assert "my_conic_k" in error_msg
