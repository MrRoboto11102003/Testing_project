import pytest
import arnio as ar
from dataclasses import FrozenInstanceError


class TestEmailDefaults:
    """Test Email field creation with default parameters."""
    
    def test_email_default_values(self):
        """Email() creates field with correct defaults."""
        field = ar.Email()
        assert field.dtype == "string"
        assert field.semantic == "email"
        assert field.nullable is True
        assert field.unique is False
        assert field.severity == "error"
        assert field.required_if is None


class TestEmailValidationParameter:
    """Test the validation parameter."""
    
    def test_validation_light(self):
        """validation='light' sets semantic to 'email'."""
        field = ar.Email(validation="light")
        assert field.semantic == "email"
    
    def test_validation_strict(self):
        """validation='strict' sets semantic to 'email:strict'."""
        field = ar.Email(validation="strict")
        assert field.semantic == "email:strict"
    
    def test_validation_not_string_raises_type_error(self):
        """Non-string validation raises TypeError."""
        with pytest.raises(TypeError, match="Email validation must be a string"):
            ar.Email(validation=None)
        
        with pytest.raises(TypeError, match="Email validation must be a string"):
            ar.Email(validation=123)
    
    def test_validation_invalid_string_raises_value_error(self):
        """Invalid validation string raises ValueError."""
        with pytest.raises(ValueError, match="Email validation must be 'light' or 'strict'"):
            ar.Email(validation="LIGHT")
        
        with pytest.raises(ValueError, match="Email validation must be 'light' or 'strict'"):
            ar.Email(validation="medium")


class TestEmailOtherParameters:
    """Test other parameters are passed through correctly."""
    
    def test_nullable_false(self):
        """nullable=False is passed through."""
        field = ar.Email(nullable=False)
        assert field.nullable is False
    
    def test_unique_true(self):
        """unique=True is passed through."""
        field = ar.Email(unique=True)
        assert field.unique is True
    
    def test_severity_warning(self):
        """severity='warning' is passed through."""
        field = ar.Email(severity="warning")
        assert field.severity == "warning"
    
    def test_required_if_tuple(self):
        """required_if tuple is passed through."""
        field = ar.Email(required_if=("country", "US"))
        assert field.required_if == ("country", "US")


class TestEmailFieldPostInitValidation:
    """Test Field.__post_init__ parameter validation."""
    
    def test_nullable_must_be_bool(self):
        """nullable must be bool, not int."""
        with pytest.raises(TypeError, match="nullable must be a bool"):
            ar.Email(nullable=1)
    
    def test_unique_must_be_bool(self):
        """unique must be bool."""
        with pytest.raises(TypeError, match="unique must be a bool"):
            ar.Email(unique="yes")
    
    def test_severity_validation(self):
        """severity must be string and one of 'error' or 'warning'."""
        with pytest.raises(TypeError, match="severity must be a string"):
            ar.Email(severity=1)
        
        with pytest.raises(ValueError, match="severity must be 'error' or 'warning'"):
            ar.Email(severity="info")
    
    def test_required_if_must_be_tuple_or_none(self):
        """required_if must be tuple or None, not list."""
        with pytest.raises(TypeError, match="required_if must be a tuple or None"):
            ar.Email(required_if=["col", "val"])
    
    def test_required_if_must_have_length_2(self):
        """required_if tuple must have exactly 2 elements."""
        with pytest.raises(TypeError, match="required_if must be a \\(column_name, expected_value\\) tuple"):
            ar.Email(required_if=("col",))
    
    def test_required_if_column_must_be_string(self):
        """required_if first element must be string."""
        with pytest.raises(TypeError, match="required_if column name must be a string"):
            ar.Email(required_if=(1, "value"))
    
    def test_required_if_value_must_be_scalar(self):
        """required_if second element must be scalar."""
        with pytest.raises(TypeError, match="required_if expected value must be a scalar"):
            ar.Email(required_if=("col", [1]))


class TestEmailOrderAndBehavior:
    """Test validation order and field behavior."""
    
    def test_validation_error_before_field_errors(self):
        """validation errors raised before Field.__post_init__ errors."""
        with pytest.raises(ValueError, match="Email validation must be 'light' or 'strict'"):
            ar.Email(validation="LIGHT", nullable=1)
    
    def test_positional_arguments_not_accepted(self):
        """Positional arguments are rejected."""
        with pytest.raises(TypeError):
            ar.Email(True)
    
    def test_field_is_frozen(self):
        """Email fields are immutable frozen dataclasses."""
        field = ar.Email()
        with pytest.raises(FrozenInstanceError):
            field.nullable = False
    
    def test_email_equals_equivalent_field(self):
        """Email() equals Field with equivalent configuration."""
        email = ar.Email()
        field = ar.Field(dtype="string", semantic="email")
        assert email == field
