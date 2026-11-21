import pytest
import arnio as ar


class TestEmailField:
    """Tests for arnio.schema.Email field creation."""
    
    def test_email_default_parameters(self):
        """Test Email field with default parameters."""
        field = ar.schema.Email()
        assert field.dtype == "string"
        assert field.nullable is True
        assert field.unique is False
        assert field.severity == "error"
        assert field.semantic == "email"
        assert field.required_if is None
    
    def test_email_light_validation(self):
        """Test Email field with 'light' validation mode."""
        field = ar.schema.Email(validation="light")
        assert field.semantic == "email"
    
    def test_email_strict_validation(self):
        """Test Email field with 'strict' validation mode."""
        field = ar.schema.Email(validation="strict")
        assert field.semantic == "email:strict"
    
    def test_email_nullable_false(self):
        """Test Email field with nullable=False."""
        field = ar.schema.Email(nullable=False)
        assert field.nullable is False
    
    def test_email_nullable_true(self):
        """Test Email field with nullable=True."""
        field = ar.schema.Email(nullable=True)
        assert field.nullable is True
    
    def test_email_unique_true(self):
        """Test Email field with unique=True."""
        field = ar.schema.Email(unique=True)
        assert field.unique is True
    
    def test_email_unique_false(self):
        """Test Email field with unique=False."""
        field = ar.schema.Email(unique=False)
        assert field.unique is False
    
    def test_email_custom_severity(self):
        """Test Email field with custom severity level."""
        field = ar.schema.Email(severity="warning")
        assert field.severity == "warning"
    
    def test_email_required_if_condition(self):
        """Test Email field with required_if condition."""
        condition = ("status", "active")
        field = ar.schema.Email(required_if=condition)
        assert field.required_if == condition
    
    def test_email_validation_type_error_int(self):
        """Test Email field raises TypeError when validation is not a string."""
        with pytest.raises(TypeError, match="Email validation must be a string"):
            ar.schema.Email(validation=123)
    
    def test_email_validation_type_error_none(self):
        """Test Email field raises TypeError when validation is None."""
        with pytest.raises(TypeError, match="Email validation must be a string"):
            ar.schema.Email(validation=None)
    
    def test_email_validation_value_error_invalid_mode(self):
        """Test Email field raises ValueError for invalid validation mode."""
        with pytest.raises(ValueError, match="Email validation must be 'light' or 'strict'"):
            ar.schema.Email(validation="medium")
    
    def test_email_validation_value_error_empty_string(self):
        """Test Email field raises ValueError for empty validation string."""
        with pytest.raises(ValueError, match="Email validation must be 'light' or 'strict'"):
            ar.schema.Email(validation="")
    
    def test_email_all_parameters_combined(self):
        """Test Email field with all parameters specified."""
        condition = ("tier", "premium")
        field = ar.schema.Email(
            nullable=False,
            unique=True,
            severity="warning",
            validation="strict",
            required_if=condition
        )
        assert field.dtype == "string"
        assert field.nullable is False
        assert field.unique is True
        assert field.severity == "warning"
        assert field.semantic == "email:strict"
        assert field.required_if == condition
    
    def test_email_dtype_always_string(self):
        """Test that Email field dtype is always 'string' regardless of parameters."""
        field1 = ar.schema.Email(validation="light", nullable=False)
        field2 = ar.schema.Email(validation="strict", unique=True)
        assert field1.dtype == "string"
        assert field2.dtype == "string"
