import pytest
from arnio.schema import Email, Field


class TestEmailCreation:
    """Test Email field creation."""
    
    def test_email_returns_field(self):
        """Email() should return a Field object."""
        field = Email()
        assert isinstance(field, Field)
    
    def test_email_default_properties(self):
        """Email with defaults should have correct properties."""
        field = Email()
        assert field.dtype == "string"
        assert field.nullable is True
        assert field.unique is False
        assert field.severity == "error"
        assert field.semantic == "email"
        assert field.required_if is None


class TestEmailValidationModes:
    """Test validation parameter behavior."""
    
    def test_email_light_validation_default(self):
        """Default validation should be 'light' with semantic='email'."""
        field = Email()
        assert field.semantic == "email"
    
    def test_email_light_validation_explicit(self):
        """validation='light' should set semantic='email'."""
        field = Email(validation="light")
        assert field.semantic == "email"
    
    def test_email_strict_validation(self):
        """validation='strict' should set semantic='email:strict'."""
        field = Email(validation="strict")
        assert field.semantic == "email:strict"
    
    def test_email_validation_not_string_raises_type_error(self):
        """Non-string validation should raise TypeError."""
        with pytest.raises(TypeError, match="Email validation must be a string"):
            Email(validation=123)
    
    def test_email_validation_none_raises_type_error(self):
        """validation=None should raise TypeError (not a string)."""
        with pytest.raises(TypeError, match="Email validation must be a string"):
            Email(validation=None)
    
    def test_email_validation_invalid_string_raises_value_error(self):
        """Invalid validation string should raise ValueError."""
        with pytest.raises(ValueError, match="Email validation must be 'light' or 'strict'"):
            Email(validation="medium")


class TestEmailParameters:
    """Test individual parameters."""
    
    def test_email_nullable_true(self):
        """nullable=True should be respected."""
        field = Email(nullable=True)
        assert field.nullable is True
    
    def test_email_nullable_false(self):
        """nullable=False should be respected."""
        field = Email(nullable=False)
        assert field.nullable is False
    
    def test_email_unique_true(self):
        """unique=True should be respected."""
        field = Email(unique=True)
        assert field.unique is True
    
    def test_email_unique_false(self):
        """unique=False should be respected."""
        field = Email(unique=False)
        assert field.unique is False
    
    def test_email_severity_parameter(self):
        """severity parameter should control field.severity."""
        field = Email(severity="warning")
        assert field.severity == "warning"
    
    def test_email_required_if_parameter(self):
        """required_if parameter should be set correctly."""
        condition = ("status", "verified")
        field = Email(required_if=condition)
        assert field.required_if == condition
    
    def test_email_all_parameters_combined(self):
        """All parameters should work together correctly."""
        field = Email(
            nullable=False,
            unique=True,
            severity="warning",
            validation="strict",
            required_if=("active", True)
        )
        assert field.dtype == "string"
        assert field.nullable is False
        assert field.unique is True
        assert field.severity == "warning"
        assert field.semantic == "email:strict"
        assert field.required_if == ("active", True)
