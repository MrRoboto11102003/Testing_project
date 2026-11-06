import pytest
import pandas as pd
import arnio as ar


class TestEmailFieldCreation:
    """Test Email field creation with various parameters."""
    
    def test_email_default_parameters(self):
        """Email field created with default parameters."""
        field = ar.Email()
        assert field is not None
    
    def test_email_with_nullable_false(self):
        """Email field with nullable=False."""
        field = ar.Email(nullable=False)
        assert field is not None
    
    def test_email_with_unique_true(self):
        """Email field with unique=True."""
        field = ar.Email(unique=True)
        assert field is not None
    
    def test_email_with_severity_warning(self):
        """Email field with severity='warning'."""
        field = ar.Email(severity="warning")
        assert field is not None
    
    def test_email_with_validation_strict(self):
        """Email field with validation='strict'."""
        field = ar.Email(validation="strict")
        assert field is not None
    
    def test_email_with_required_if(self):
        """Email field with required_if condition."""
        field = ar.Email(required_if=("user_type", "premium"))
        assert field is not None


class TestEmailValidation:
    """Test email format validation."""
    
    def test_valid_email_basic_format(self):
        """Valid basic email address passes validation."""
        schema = ar.Schema({"email": ar.Email(nullable=False)})
        frame = ar.from_dict({"email": ["user@example.com"]})
        result = ar.validate(frame, schema)
        assert result.passed
    
    def test_valid_email_with_plus_addressing(self):
        """Valid email with plus addressing passes validation."""
        schema = ar.Schema({"email": ar.Email(nullable=False)})
        frame = ar.from_dict({"email": ["user+tag@example.com"]})
        result = ar.validate(frame, schema)
        assert result.passed
    
    def test_invalid_email_missing_at_symbol(self):
        """Email without @ symbol is rejected."""
        schema = ar.Schema({"email": ar.Email(nullable=False)})
        frame = ar.from_dict({"email": ["notanemail.com"]})
        result = ar.validate(frame, schema)
        assert not result.passed
    
    def test_invalid_email_missing_domain(self):
        """Email without domain portion is rejected."""
        schema = ar.Schema({"email": ar.Email(nullable=False)})
        frame = ar.from_dict({"email": ["user@"]})
        result = ar.validate(frame, schema)
        assert not result.passed


class TestEmailNullableConstraint:
    """Test nullable parameter behavior."""
    
    def test_nullable_true_accepts_null_values(self):
        """nullable=True allows null/missing values."""
        schema = ar.Schema({"email": ar.Email(nullable=True)})
        frame = ar.from_dict({"email": ["valid@example.com", None]})
        result = ar.validate(frame, schema)
        assert result.passed
    
    def test_nullable_false_rejects_null_values(self):
        """nullable=False rejects null/missing values."""
        schema = ar.Schema({"email": ar.Email(nullable=False)})
        frame = ar.from_dict({"email": ["valid@example.com", None]})
        result = ar.validate(frame, schema)
        assert not result.passed


class TestEmailUniqueConstraint:
    """Test unique parameter behavior."""
    
    def test_unique_false_allows_duplicates(self):
        """unique=False permits duplicate email values."""
        schema = ar.Schema({"email": ar.Email(nullable=False, unique=False)})
        frame = ar.from_dict({"email": ["user@example.com", "user@example.com"]})
        result = ar.validate(frame, schema)
        assert result.passed
    
    def test_unique_true_rejects_duplicates(self):
        """unique=True rejects duplicate email values."""
        schema = ar.Schema({"email": ar.Email(nullable=False, unique=True)})
        frame = ar.from_dict({"email": ["user@example.com", "user@example.com"]})
        result = ar.validate(frame, schema)
        assert not result.passed


class TestEmailSeverityLevels:
    """Test severity parameter for validation error handling."""
    
    def test_severity_error_fails_validation(self):
        """severity='error' causes validation to fail on invalid email."""
        schema = ar.Schema({"email": ar.Email(nullable=False, severity="error")})
        frame = ar.from_dict({"email": ["invalid-email"]})
        result = ar.validate(frame, schema)
        assert not result.passed
    
    def test_severity_warning_passes_validation(self):
        """severity='warning' allows validation to pass with warnings."""
        schema = ar.Schema({"email": ar.Email(nullable=False, severity="warning")})
        frame = ar.from_dict({"email": ["invalid-email"]})
        result = ar.validate(frame, schema)
        assert result.passed
        assert result.issue_count > 0


class TestEmailEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_string_is_invalid(self):
        """Empty string is not treated as a valid email."""
        schema = ar.Schema({"email": ar.Email(nullable=False)})
        frame = ar.from_dict({"email": [""]})
        result = ar.validate(frame, schema)
        assert not result.passed
    
    def test_whitespace_only_is_invalid(self):
        """Whitespace-only string is not a valid email."""
        schema = ar.Schema({"email": ar.Email(nullable=False)})
        frame = ar.from_dict({"email": ["   "]})
        result = ar.validate(frame, schema)
        assert not result.passed
