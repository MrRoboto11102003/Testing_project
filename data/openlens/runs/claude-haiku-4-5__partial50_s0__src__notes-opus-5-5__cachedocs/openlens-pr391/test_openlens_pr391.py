import pytest
from openlens.validation import safe_float_conversion, validate_conic_constant, ValidationError


class TestSafeFloatConversion:
    """Tests for safe_float_conversion function."""
    
    def test_valid_float(self):
        """Test valid float input."""
        success, value = safe_float_conversion(3.5)
        assert success is True and value == 3.5 and type(value) is float
    
    def test_valid_int_converts_to_float(self):
        """Test int input converts to float type."""
        success, value = safe_float_conversion(7)
        assert success is True and value == 7.0 and type(value) is float
    
    def test_valid_string_with_whitespace(self):
        """Test string parsing with whitespace."""
        success, value = safe_float_conversion("  42  ")
        assert success is True and value == 42.0
    
    def test_bool_rejected(self):
        """Test bool is rejected (bool subclasses int)."""
        success, value = safe_float_conversion(True, default=-1.0)
        assert success is False and value == -1.0
    
    def test_nan_inf_rejected(self):
        """Test NaN and infinity are rejected."""
        assert safe_float_conversion(float('nan'), default=-1.0) == (False, -1.0)
        assert safe_float_conversion(float('inf'), default=-1.0) == (False, -1.0)
        assert safe_float_conversion(float('-inf'), default=-1.0) == (False, -1.0)
    
    def test_large_int_overflow(self):
        """Test very large int (OverflowError) is handled."""
        success, value = safe_float_conversion(10**400, default=-1.0)
        assert success is False and value == -1.0
    
    def test_invalid_string(self):
        """Test invalid string input."""
        success, value = safe_float_conversion("abc", default=-1.0)
        assert success is False and value == -1.0
    
    def test_default_returned_unconverted(self):
        """Test default is returned as-is without conversion."""
        sentinel = object()
        success, value = safe_float_conversion("invalid", default=sentinel)
        assert success is False and value is sentinel


class TestValidateConicConstant:
    """Tests for validate_conic_constant function."""
    
    def test_valid_sphere_parabola_hyperbola(self):
        """Test valid conic constants (sphere, parabola, hyperbola)."""
        assert validate_conic_constant(0) == 0.0 and type(validate_conic_constant(0)) is float
        assert validate_conic_constant(-1) == -1.0
        assert validate_conic_constant(-2.5) == -2.5
    
    def test_boundaries_accepted(self):
        """Test boundary values -100 and 100 are accepted."""
        assert validate_conic_constant(-100) == -100.0
        assert validate_conic_constant(100) == 100.0
    
    def test_out_of_range_rejected(self):
        """Test values outside [-100, 100] raise ValidationError."""
        with pytest.raises(ValidationError, match=r"must be between -100 and 100, got 150\.0"):
            validate_conic_constant(150)
        with pytest.raises(ValidationError, match=r"must be between -100 and 100, got -100\.5"):
            validate_conic_constant(-100.5)
    
    def test_bool_rejected(self):
        """Test bool is rejected with specific message."""
        with pytest.raises(ValidationError, match=r"must be a number, got bool"):
            validate_conic_constant(True)
    
    def test_non_numeric_types_rejected(self):
        """Test non-numeric types (str, None, complex) are rejected."""
        with pytest.raises(ValidationError, match=r"must be a number, got str"):
            validate_conic_constant("1")
        with pytest.raises(ValidationError, match=r"must be a number, got NoneType"):
            validate_conic_constant(None)
        with pytest.raises(ValidationError, match=r"must be a number, got complex"):
            validate_conic_constant(1j)
    
    def test_non_finite_rejected(self):
        """Test NaN, inf, and -inf are rejected."""
        with pytest.raises(ValidationError, match=r"must be a finite number, got nan"):
            validate_conic_constant(float('nan'))
        with pytest.raises(ValidationError, match=r"must be a finite number, got inf"):
            validate_conic_constant(float('inf'))
        with pytest.raises(ValidationError, match=r"must be a finite number, got -inf"):
            validate_conic_constant(float('-inf'))
    
    def test_always_returns_float_type(self):
        """Test return type is always float, never input object."""
        result = validate_conic_constant(-1)  # int input
        assert type(result) is float
    
    def test_custom_param_name_in_error(self):
        """Test custom param_name is used in error message."""
        with pytest.raises(ValidationError, match=r"K1 must be between -100 and 100"):
            validate_conic_constant(150, param_name="K1")
