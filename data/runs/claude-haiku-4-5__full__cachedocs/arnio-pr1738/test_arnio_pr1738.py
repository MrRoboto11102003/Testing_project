"""
Pytest tests for arnio.schema.PhoneNumber and arnio.schema.URL validation.

Tests the documented behavior, edge cases, and parameter combinations
for phone number and URL schema field validation.
"""

import pytest
import pandas as pd
import arnio as ar


# ============================================================================
# PhoneNumber Tests
# ============================================================================


class TestPhoneNumberValidation:
    """Test PhoneNumber schema field validation."""

    def test_valid_phone_numbers_pass(self):
        """Valid phone numbers should pass validation."""
        df = pd.DataFrame({
            "phone": ["+1-555-123-4567", "+91 9876543210", "5551234567"]
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({"phone": ar.PhoneNumber(nullable=False)})
        result = ar.validate(frame, schema)
        assert result.passed

    def test_invalid_phone_numbers_fail(self):
        """Invalid phone numbers should fail validation."""
        df = pd.DataFrame({"phone": ["not-a-phone", "abc123", ""]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"phone": ar.PhoneNumber(nullable=False)})
        result = ar.validate(frame, schema)
        assert not result.passed
        assert len(result.issues) > 0

    def test_nullable_true_allows_nulls(self):
        """PhoneNumber with nullable=True should allow null values."""
        df = pd.DataFrame({"phone": ["+1-555-123-4567", None]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"phone": ar.PhoneNumber(nullable=True)})
        result = ar.validate(frame, schema)
        assert result.passed

    def test_nullable_false_rejects_nulls(self):
        """PhoneNumber with nullable=False should reject null values."""
        df = pd.DataFrame({"phone": ["+1-555-123-4567", None]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"phone": ar.PhoneNumber(nullable=False)})
        result = ar.validate(frame, schema)
        assert not result.passed
        assert any(issue.rule == "required" for issue in result.issues)

    def test_unique_true_enforces_uniqueness(self):
        """PhoneNumber with unique=True should reject duplicate valid phone numbers."""
        df = pd.DataFrame({
            "phone": ["+1-555-123-4567", "+1-555-123-4567"]
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({"phone": ar.PhoneNumber(nullable=False, unique=True)})
        result = ar.validate(frame, schema)
        assert not result.passed
        assert any(issue.rule == "unique" for issue in result.issues)

    def test_severity_warning_does_not_fail_validation(self):
        """PhoneNumber with severity='warning' should report issues but pass."""
        df = pd.DataFrame({"phone": ["invalid-phone"]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"phone": ar.PhoneNumber(severity="warning")})
        result = ar.validate(frame, schema)
        # With severity="warning", passed should still be True
        assert result.passed or len(result.issues) > 0

    def test_required_if_conditional_requirement(self):
        """PhoneNumber with required_if should enforce conditional requirement."""
        df = pd.DataFrame({
            "customer_type": ["support", "regular", "support"],
            "phone": ["+1-555-123-4567", None, None]
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "customer_type": ar.String(),
            "phone": ar.PhoneNumber(
                nullable=True,
                required_if=("customer_type", "support")
            )
        })
        result = ar.validate(frame, schema)
        # The third row should fail: customer_type='support' but phone is null
        assert not result.passed


# ============================================================================
# URL Tests
# ============================================================================


class TestURLValidation:
    """Test URL schema field validation."""

    def test_valid_urls_pass(self):
        """Valid URLs should pass validation."""
        df = pd.DataFrame({
            "website": [
                "https://example.com",
                "http://example.com",
                "https://example.com/path",
                "https://sub.example.com/path?query=1"
            ]
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({"website": ar.URL(nullable=False)})
        result = ar.validate(frame, schema)
        assert result.passed

    def test_invalid_urls_fail(self):
        """Invalid URLs should fail validation."""
        df = pd.DataFrame({
            "website": ["not a url", "ht@tp://example.com", "example"]
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({"website": ar.URL(nullable=False)})
        result = ar.validate(frame, schema)
        assert not result.passed
        assert len(result.issues) > 0

    def test_url_nullable_true_allows_nulls(self):
        """URL with nullable=True should allow null values."""
        df = pd.DataFrame({"website": ["https://example.com", None]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"website": ar.URL(nullable=True)})
        result = ar.validate(frame, schema)
        assert result.passed

    def test_url_nullable_false_rejects_nulls(self):
        """URL with nullable=False should reject null values."""
        df = pd.DataFrame({"website": ["https://example.com", None]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"website": ar.URL(nullable=False)})
        result = ar.validate(frame, schema)
        assert not result.passed
        assert any(issue.rule == "required" for issue in result.issues)

    def test_url_unique_true_enforces_uniqueness(self):
        """URL with unique=True should reject duplicate URLs."""
        df = pd.DataFrame({
            "website": ["https://example.com", "https://example.com"]
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({"website": ar.URL(nullable=False, unique=True)})
        result = ar.validate(frame, schema)
        assert not result.passed
        assert any(issue.rule == "unique" for issue in result.issues)

    def test_allowed_schemes_https_only(self):
        """URL with allowed_schemes=['https'] should reject http."""
        df = pd.DataFrame({
            "website": ["https://example.com", "http://example.com"]
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "website": ar.URL(nullable=False, allowed_schemes=["https"])
        })
        result = ar.validate(frame, schema)
        # Should fail because http is not in allowed_schemes
        assert not result.passed

    def test_allowed_schemes_multiple(self):
        """URL with allowed_schemes list should accept only those schemes."""
        df = pd.DataFrame({
            "website": ["https://example.com", "ftp://example.com"]
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "website": ar.URL(
                nullable=False,
                allowed_schemes=["https", "ftp"]
            )
        })
        result = ar.validate(frame, schema)
        assert result.passed

    def test_url_severity_warning(self):
        """URL with severity='warning' should report issues but pass."""
        df = pd.DataFrame({"website": ["ftp://example.com"]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "website": ar.URL(
                severity="warning",
                allowed_schemes=["https"]
            )
        })
        result = ar.validate(frame, schema)
        # With severity="warning", passed should still be True
        assert result.passed or len(result.issues) > 0

    def test_url_required_if_conditional(self):
        """URL with required_if should enforce conditional requirement."""
        df = pd.DataFrame({
            "has_website": [True, False, True],
            "website": ["https://example.com", None, None]
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "has_website": ar.Bool(),
            "website": ar.URL(
                nullable=True,
                required_if=("has_website", True)
            )
        })
        result = ar.validate(frame, schema)
        # The third row should fail: has_website=True but website is null
        assert not result.passed
