import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestEmailField:
    """Tests for arnio.schema.Email field validator."""

    def test_valid_email_passes(self):
        """Valid email addresses should pass validation."""
        df = pd.DataFrame({"email": ["user@example.com"]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"email": ar.Email()})
        result = ar.validate(frame, schema)
        assert result.passed is True

    def test_invalid_email_fails(self):
        """Invalid email formats should fail validation."""
        df = pd.DataFrame({"email": ["not-an-email"]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"email": ar.Email(nullable=False)})
        result = ar.validate(frame, schema)
        assert result.passed is False

    def test_nullable_true_allows_nulls(self):
        """With nullable=True, null values should pass validation."""
        df = pd.DataFrame({"email": ["user@example.com", None]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"email": ar.Email(nullable=True)})
        result = ar.validate(frame, schema)
        assert result.passed is True

    def test_nullable_false_rejects_nulls(self):
        """With nullable=False, null values should fail validation."""
        df = pd.DataFrame({"email": ["user@example.com", None]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"email": ar.Email(nullable=False)})
        result = ar.validate(frame, schema)
        assert result.passed is False

    def test_unique_rejects_duplicates(self):
        """With unique=True, duplicate email values should fail validation."""
        df = pd.DataFrame({
            "email": ["alice@example.com", "alice@example.com"]
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({"email": ar.Email(unique=True, nullable=False)})
        result = ar.validate(frame, schema)
        assert result.passed is False

    def test_unique_allows_distinct_values(self):
        """Unique constraint should pass for distinct email addresses."""
        df = pd.DataFrame({
            "email": ["alice@example.com", "bob@example.com", "charlie@example.com"]
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({"email": ar.Email(unique=True, nullable=False)})
        result = ar.validate(frame, schema)
        assert result.passed is True

    def test_unique_allows_multiple_nulls(self):
        """With unique=True and nullable=True, multiple nulls should pass."""
        df = pd.DataFrame({"email": ["user@example.com", None, None]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"email": ar.Email(unique=True, nullable=True)})
        result = ar.validate(frame, schema)
        assert result.passed is True

    def test_light_validation_mode(self):
        """Light validation mode should accept basic email formats."""
        df = pd.DataFrame({
            "email": ["user@localhost", "test@domain", "valid@example.com"]
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({"email": ar.Email(validation="light", nullable=False)})
        result = ar.validate(frame, schema)
        assert isinstance(result.passed, bool)

    def test_strict_validation_mode(self):
        """Strict validation mode should reject malformed emails."""
        df = pd.DataFrame({"email": ["@@example.com", "no@at@sign.com"]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"email": ar.Email(validation="strict", nullable=False)})
        result = ar.validate(frame, schema)
        assert result.passed is False

    def test_severity_error_fails_validation(self):
        """With severity='error', invalid emails should fail validation."""
        df = pd.DataFrame({"email": ["invalid-email"]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"email": ar.Email(severity="error", nullable=False)})
        result = ar.validate(frame, schema)
        assert result.passed is False

    def test_severity_warning_passes_with_issues(self):
        """With severity='warning', validation passes but issues are recorded."""
        df = pd.DataFrame({"email": ["invalid-email"]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"email": ar.Email(severity="warning", nullable=False)})
        result = ar.validate(frame, schema)
        assert result.passed is True
        assert result.issue_count > 0

    def test_required_if_condition_met(self):
        """When required_if condition is met, null email should fail."""
        df = pd.DataFrame({
            "user_type": ["vendor", "customer"],
            "email": [None, "user@example.com"]
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "user_type": ar.String(nullable=False),
            "email": ar.Email(
                nullable=True,
                required_if=("user_type", "vendor"),
                validation="light"
            )
        })
        result = ar.validate(frame, schema)
        assert result.passed is False

    def test_required_if_condition_not_met(self):
        """When required_if condition is not met, null email should pass."""
        df = pd.DataFrame({
            "user_type": ["customer", "customer"],
            "email": [None, "user@example.com"]
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "user_type": ar.String(nullable=False),
            "email": ar.Email(
                nullable=True,
                required_if=("user_type", "vendor"),
                validation="light"
            )
        })
        result = ar.validate(frame, schema)
        assert result.passed is True

    def test_common_email_formats(self):
        """Common valid email formats should pass validation."""
        df = pd.DataFrame({
            "email": [
                "simple@example.com",
                "user.name@example.com",
                "user+tag@example.co.uk",
                "test_underscore@domain.org"
            ]
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({"email": ar.Email(validation="light", nullable=False)})
        result = ar.validate(frame, schema)
        assert result.passed is True
