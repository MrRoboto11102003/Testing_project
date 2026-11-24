import pytest
import re
from openlens.validation import safe_float_conversion, validate_conic_constant, ValidationError


class TestSafeFloatConversion:
    """Tests for safe_float_conversion function."""
    
    def test_valid_float(self):
        """Valid float returns (True, value)."""
        result = safe_float_conversion(3.5)
        assert result == (True, 3.5)
        assert isinstance(result, tuple) and len(result) == 2
        assert result[0] is True
    
    def test_valid_int_becomes_float(self):
        """Valid int is converted to float type."""
        result = safe_float_conversion(7)
        assert result[0] is True
        assert type(result[1]) is float
        assert result[1] == 7.0
    
    def test_valid_string(self):
        """Valid numeric string returns (True, converted_value)."""
        result = safe_float_conversion("3.5")
        assert result == (True, 3.5)
        result = safe_float_conversion("  42  ")  # Whitespace handling
        assert result == (True, 42.0)
    
    def test_booleans_rejected(self):
        """Booleans are explicitly rejected (bool subclasses int)."""
        assert safe_float_conversion(True, default=9.0) == (False, 9.0)
        assert safe_float_conversion(False) == (False, 0.0)
    
    def test_nan_and_inf_rejected(self):
        """NaN and infinity values are rejected."""
        assert safe_float_conversion(float('nan')) == (False, 0.0)
        assert safe_float_conversion(float('inf')) == (False, 0.0)
        assert safe_float_conversion(float('-inf'), default=-1.0) == (False, -1.0)
    
    def test_large_int_overflow(self):
        """Very large ints that overflow are handled gracefully."""
        result = safe_float_conversion(10**400)
        assert result == (False, 0.0)
    
    def test_invalid_inputs(self):
        """Invalid input types return (False, default)."""
        assert safe_float_conversion(None) == (False, 0.0)
        assert safe_float_conversion(b"1") == (False, 0.0)
        assert safe_float_conversion([1]) == (False, 0.0)
        assert safe_float_conversion("abc", default=-1.0) == (False, -1.0)
    
    def test_string_nan_inf_rejected(self):
        """String representations of NaN/inf are rejected after parsing."""
        assert safe_float_conversion("nan") == (False, 0.0)
        assert safe_float_conversion("inf") == (False, 0.0)


class TestValidateConicConstant:
    """Tests for validate_conic_constant function."""
    
    def test_valid_values_return_float(self):
        """Valid numeric values are returned as float type."""
        result = validate_conic_constant(-1)
        assert type(result) is float
        assert result == -1.0
        
        result = validate_conic_constant(0)
        assert type(result) is float
        assert result == 0.0
    
    def test_boundaries_accepted(self):
        """Boundary values -100 and 100 are accepted."""
        assert validate_conic_constant(-100) == -100.0
        assert validate_conic_constant(100) == 100.0
    
    def test_values_outside_range_rejected(self):
        """Values outside [-100, 100] raise ValidationError."""
        with pytest.raises(ValidationError, match=re.escape("must be between -100 and 100, got 150.0")):
            validate_conic_constant(150)
        with pytest.raises(ValidationError, match=re.escape("must be between -100 and 100, got -100.5")):
            validate_conic_constant(-100.5)
    
    def test_type_errors(self):
        """Non-numeric types raise ValidationError."""
        with pytest.raises(ValidationError, match=re.escape("must be a number, got bool")):
            validate_conic_constant(True)
        with pytest.raises(ValidationError, match=re.escape("must be a number, got str")):
            validate_conic_constant("1")
        with pytest.raises(ValidationError, match=re.escape("must be a number, got NoneType")):
            validate_conic_constant(None)
    
    def test_non_finite_values_rejected(self):
        """NaN and infinity values are rejected."""
        with pytest.raises(ValidationError, match="must be a finite number"):
            validate_conic_constant(float('nan'))
        with pytest.raises(ValidationError, match="must be a finite number"):
            validate_conic_constant(float('inf'))
        with pytest.raises(ValidationError, match="must be a finite number"):
            validate_conic_constant(float('-inf'))
    
    def test_custom_param_name(self):
        """Custom param_name appears in error messages."""
        with pytest.raises(ValidationError, match=re.escape("K1 must be between -100 and 100, got 101.0")):
            validate_conic_constant(101, param_name="K1")
