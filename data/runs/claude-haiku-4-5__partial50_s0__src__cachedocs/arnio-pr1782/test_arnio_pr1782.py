import pytest
import arnio as ar


def test_email_returns_field():
    """Email() returns a Field object."""
    field = ar.Email()
    assert field is not None


def test_email_default_dtype():
    """Email field defaults to dtype='string'."""
    field = ar.Email()
    assert field.dtype == "string"


def test_email_default_nullable():
    """Email field is nullable by default."""
    field = ar.Email()
    assert field.nullable is True


def test_email_nullable_parameter():
    """Email respects nullable parameter."""
    assert ar.Email(nullable=True).nullable is True
    assert ar.Email(nullable=False).nullable is False


def test_email_default_unique():
    """Email field is not unique by default."""
    field = ar.Email()
    assert field.unique is False


def test_email_unique_parameter():
    """Email respects unique parameter."""
    assert ar.Email(unique=True).unique is True
    assert ar.Email(unique=False).unique is False


def test_email_default_severity():
    """Email field defaults to severity='error'."""
    field = ar.Email()
    assert field.severity == "error"


def test_email_severity_parameter():
    """Email respects severity parameter."""
    field = ar.Email(severity="warning")
    assert field.severity == "warning"


def test_email_validation_light_default():
    """Email defaults to light validation (semantic='email')."""
    field = ar.Email()
    assert field.semantic == "email"


def test_email_validation_light():
    """Email with validation='light' uses semantic='email'."""
    field = ar.Email(validation="light")
    assert field.semantic == "email"


def test_email_validation_strict():
    """Email with validation='strict' uses semantic='email:strict'."""
    field = ar.Email(validation="strict")
    assert field.semantic == "email:strict"


def test_email_required_if_parameter():
    """Email respects required_if parameter."""
    field = ar.Email(required_if=("user_type", "premium"))
    assert field.required_if == ("user_type", "premium")


def test_email_validation_non_string_raises_typeerror():
    """Email raises TypeError when validation is not a string."""
    with pytest.raises(TypeError, match="Email validation must be a string"):
        ar.Email(validation=123)


def test_email_validation_invalid_string_raises_valueerror():
    """Email raises ValueError for invalid validation string."""
    with pytest.raises(ValueError, match="Email validation must be 'light' or 'strict'"):
        ar.Email(validation="invalid")


def test_email_all_parameters():
    """Email correctly handles all parameters together."""
    field = ar.Email(
        nullable=False,
        unique=True,
        severity="warning",
        validation="strict",
        required_if=("status", "active")
    )
    
    assert field.dtype == "string"
    assert field.nullable is False
    assert field.unique is True
    assert field.severity == "warning"
    assert field.semantic == "email:strict"
    assert field.required_if == ("status", "active")
