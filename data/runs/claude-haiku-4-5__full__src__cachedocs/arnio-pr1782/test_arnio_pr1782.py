import pytest
import arnio as ar
import pandas as pd


class TestEmailField:
    """Tests for arnio.schema.Email field factory function."""

    def test_email_default_parameters(self):
        """Email field created with default parameters."""
        field = ar.Email()
        assert field is not None

    def test_email_validation_light_mode(self):
        """Email field with light validation mode accepts valid emails."""
        field = ar.Email(validation="light")
        schema = ar.Schema({"email": field})
        frame = ar.from_pandas(pd.DataFrame({"email": ["test@example.com"]}))
        result = ar.validate(frame, schema)
        assert result.passed

    def test_email_validation_strict_mode(self):
        """Email field with strict validation mode accepts valid emails."""
        field = ar.Email(validation="strict")
        schema = ar.Schema({"email": field})
        frame = ar.from_pandas(pd.DataFrame({"email": ["test@example.com"]}))
        result = ar.validate(frame, schema)
        assert result.passed

    def test_email_validation_invalid_type_raises_typeerror(self):
        """Email raises TypeError if validation parameter is not a string."""
        with pytest.raises(TypeError, match="Email validation must be a string"):
            ar.Email(validation=123)

    def test_email_validation_invalid_mode_raises_valueerror(self):
        """Email raises ValueError for validation mode other than light or strict."""
        with pytest.raises(ValueError, match="Email validation must be 'light' or 'strict'"):
            ar.Email(validation="invalid")

    def test_email_nullable_false_rejects_null(self):
        """Email field with nullable=False rejects null values."""
        field = ar.Email(nullable=False)
        schema = ar.Schema({"email": field})
        frame = ar.from_pandas(pd.DataFrame({"email": [None]}))
        result = ar.validate(frame, schema)
        assert not result.passed

    def test_email_nullable_true_accepts_null(self):
        """Email field with nullable=True accepts null values."""
        field = ar.Email(nullable=True)
        schema = ar.Schema({"email": field})
        frame = ar.from_pandas(pd.DataFrame({"email": [None]}))
        result = ar.validate(frame, schema)
        assert result.passed

    def test_email_unique_true_rejects_duplicates(self):
        """Email field with unique=True rejects duplicate non-null values."""
        field = ar.Email(unique=True)
        schema = ar.Schema({"email": field})
        frame = ar.from_pandas(
            pd.DataFrame({"email": ["test@example.com", "test@example.com"]})
        )
        result = ar.validate(frame, schema)
        assert not result.passed

    def test_email_unique_false_allows_duplicates(self):
        """Email field with unique=False allows duplicate values."""
        field = ar.Email(unique=False)
        schema = ar.Schema({"email": field})
        frame = ar.from_pandas(
            pd.DataFrame({"email": ["test@example.com", "test@example.com"]})
        )
        result = ar.validate(frame, schema)
        assert result.passed

    def test_email_severity_parameter_accepted(self):
        """Email field accepts severity parameter."""
        field = ar.Email(severity="warning")
        schema = ar.Schema({"email": field})
        assert schema is not None

    def test_email_required_if_parameter_accepted(self):
        """Email field accepts required_if parameter as tuple."""
        field = ar.Email(required_if=("user_type", "business"))
        schema = ar.Schema({"email": field})
        assert schema is not None

    def test_email_invalid_format_rejected(self):
        """Email field rejects values that don't look like email addresses."""
        field = ar.Email(validation="light")
        schema = ar.Schema({"email": field})
        frame = ar.from_pandas(pd.DataFrame({"email": ["not-an-email"]}))
        result = ar.validate(frame, schema)
        assert not result.passed

    def test_email_multiple_valid_formats(self):
        """Email field accepts multiple valid email formats in light mode."""
        field = ar.Email(validation="light")
        schema = ar.Schema({"email": field})
        valid_emails = [
            "simple@example.com",
            "user.name@example.co.uk",
            "user+tag@example.com",
        ]
        frame = ar.from_pandas(pd.DataFrame({"email": valid_emails}))
        result = ar.validate(frame, schema)
        assert result.passed

    def test_email_nullable_false_unique_true_combination(self):
        """Email field works with nullable=False and unique=True together."""
        field = ar.Email(nullable=False, unique=True)
        schema = ar.Schema({"email": field})
        frame = ar.from_pandas(pd.DataFrame({"email": ["test@example.com"]}))
        result = ar.validate(frame, schema)
        assert result.passed

    def test_email_all_parameters_specified(self):
        """Email field accepts all parameters together."""
        field = ar.Email(
            nullable=False,
            unique=True,
            severity="error",
            validation="strict",
            required_if=("account_type", "business"),
        )
        schema = ar.Schema({"email": field})
        frame = ar.from_pandas(pd.DataFrame({"email": ["test@example.com"]}))
        result = ar.validate(frame, schema)
        assert result.passed
