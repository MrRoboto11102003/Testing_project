import pytest
from openlens.validation import safe_float_conversion, validate_conic_constant, ValidationError


class TestSafeFloatConversion:
    """Tests for safe_float_conversion function."""
    
    def test_valid_numeric(self):
        """Valid numeric inputs return success."""
        success, value = safe_float_conversion(3.14)
        assert success is True
        assert value == 3.14
        
        success, value = safe_float_conversion(42)
        assert success is True
        assert value == 42.0
    
    def test_valid_string(self):
        """Valid string representation returns success."""
        success, value = safe_float_conversion("3.14")
        assert success is True
        assert value == 3.14
    
    def test_zero_valid(self):
        """Zero is a valid input."""
        success, value = safe_float_conversion(0)
        assert success is True
        assert value == 0.0
    
    def test_negative_valid(self):
        """Negative values are valid."""
        success, value = safe_float_conversion(-3.14)
        assert success is True
        assert value == -3.14
    
    def test_boolean_rejected(self):
        """Booleans are rejected despite int inheritance."""
        success, value = safe_float_conversion(True)
        assert success is False and value == 0.0
        
        success, value = safe_float_conversion(False)
        assert success is False and value == 0.0
    
    def test_nan_infinity_rejected(self):
        """NaN and infinity are rejected."""
        success, value = safe_float_conversion(float('nan'))
        assert success is False and value == 0.0
        
        success, value = safe_float_conversion(float('inf'))
        assert success is False and value == 0.0
    
    def test_large_int_overflow(self):
        """Very large int (10**400) causes OverflowError."""
        success, value = safe_float_conversion(10**400)
        assert success is False and value == 0.0
    
    def test_invalid_input(self):
        """Invalid strings and None are rejected."""
        success, value = safe_float_conversion("not_a_number")
        assert success is False and value == 0.0
        
        success, value = safe_float_conversion(None)
        assert success is False and value == 0.0
    
    def test_custom_default(self):
        """Custom default is returned on conversion failure."""
        success, value = safe_float_conversion("invalid", default=99.5)
        assert success is False
        assert value == 99.5


class TestValidateConicConstant:
    """Tests for validate_conic_constant function."""
    
    def test_valid_typical_values(self):
        """Typical optical conic constants pass validation."""
        assert validate_conic_constant(0.0) == 0.0  # sphere
        assert validate_conic_constant(-1.0) == -1.0  # parabola
        assert validate_conic_constant(-0.5) == -0.5  # ellipsoid
        assert validate_conic_constant(0.5) == 0.5  # oblate
    
    def test_boundaries(self):
        """Boundary values -100 and 100 are valid."""
        assert validate_conic_constant(-100.0) == -100.0
        assert validate_conic_constant(100.0) == 100.0
    
    def test_outside_upper_bound(self):
        """Values > 100 raise ValidationError."""
        with pytest.raises(ValidationError, match="must be between"):
            validate_conic_constant(101.0)
    
    def test_outside_lower_bound(self):
        """Values < -100 raise ValidationError."""
        with pytest.raises(ValidationError, match="must be between"):
            validate_conic_constant(-101.0)
    
    def test_non_finite(self):
        """Non-finite values raise ValidationError."""
        with pytest.raises(ValidationError):
            validate_conic_constant(float('nan'))
        
        with pytest.raises(ValidationError):
            validate_conic_constant(float('inf'))
