import pytest
import pandas as pd
import arnio as ar


class TestPhoneNumberValidator:
    """Tests for arnio.schema.PhoneNumber validator."""

    def test_valid_international_phone_formats(self):
        """Test that valid international phone formats pass validation."""
        frame = ar.from_pandas(
            pd.DataFrame({
                "phone": ["+1-555-123-4567", "+91 9876543210", "5551234567"],
            })
        )
        schema = ar.Schema({"phone": ar.PhoneNumber(nullable=False)})
        result = ar.validate(frame, schema)
        assert result.passed

    def test_nullable_true_allows_none_values(self):
        """Test that nullable=True allows None values."""
        frame = ar.from_pandas(
            pd.DataFrame({
                "phone": ["+1-555-123-4567", None, "+91 9876543210"],
            })
        )
        schema = ar.Schema({"phone": ar.PhoneNumber(nullable=True)})
        result = ar.validate(frame, schema)
        assert result.passed

    def test_nullable_false_rejects_none_values(self):
        """Test that nullable=False rejects None values."""
        frame = ar.from_pandas(
            pd.DataFrame({
                "phone": ["+1-555-123-4567", None],
            })
        )
        schema = ar.Schema({"phone": ar.PhoneNumber(nullable=False)})
        result = ar.validate(frame, schema)
        assert not result.passed
        assert any(issue.column == "phone" for issue in result.issues)

    def test_invalid_phone_formats_fail_validation(self):
        """Test that invalid phone formats fail validation."""
        frame = ar.from_pandas(
            pd.DataFrame({
                "phone": ["invalid", "abc123"],
            })
        )
        schema = ar.Schema({"phone": ar.PhoneNumber(nullable=False)})
        result = ar.validate(frame, schema)
        assert not result.passed

    def test_warning_severity_does_not_fail_validation(self):
        """Test that warning severity level reports issues without failing."""
        frame = ar.from_pandas(
            pd.DataFrame({
                "phone": ["invalid"],
            })
        )
        schema = ar.Schema({"phone": ar.PhoneNumber(severity="warning")})
        result = ar.validate(frame, schema)
        assert result.passed
        assert result.issue_count > 0

    def test_unique_constraint_on_phone_numbers(self):
        """Test that unique=True enforces uniqueness."""
        frame = ar.from_pandas(
            pd.DataFrame({
                "phone": ["+1-555-123-4567", "+1-555-123-4567"],
            })
        )
        schema = ar.Schema({"phone": ar.PhoneNumber(nullable=False, unique=True)})
        result = ar.validate(frame, schema)
        assert not result.passed


class TestURLValidator:
    """Tests for arnio.schema.URL validator."""

    def test_valid_http_and_https_urls(self):
        """Test that valid HTTP and HTTPS URLs pass validation."""
        frame = ar.from_pandas(
            pd.DataFrame({
                "website": ["http://example.com", "https://google.com"],
            })
        )
        schema = ar.Schema({"website": ar.URL(nullable=False)})
        result = ar.validate(frame, schema)
        assert result.passed

    def test_nullable_true_allows_none_values(self):
        """Test that nullable=True allows None values."""
        frame = ar.from_pandas(
            pd.DataFrame({
                "website": ["https://example.com", None],
            })
        )
        schema = ar.Schema({"website": ar.URL(nullable=True)})
        result = ar.validate(frame, schema)
        assert result.passed

    def test_nullable_false_rejects_none_values(self):
        """Test that nullable=False rejects None values."""
        frame = ar.from_pandas(
            pd.DataFrame({
                "website": ["https://example.com", None],
            })
        )
        schema = ar.Schema({"website": ar.URL(nullable=False)})
        result = ar.validate(frame, schema)
        assert not result.passed
        assert any(issue.column == "website" for issue in result.issues)

    def test_allowed_schemes_https_only_rejects_http(self):
        """Test that allowed_schemes=['https'] rejects http URLs."""
        frame = ar.from_pandas(
            pd.DataFrame({
                "website": ["http://example.com"],
            })
        )
        schema = ar.Schema({"website": ar.URL(allowed_schemes=["https"])})
        result = ar.validate(frame, schema)
        assert not result.passed

    def test_allowed_schemes_https_only_accepts_https(self):
        """Test that allowed_schemes=['https'] accepts https URLs."""
        frame = ar.from_pandas(
            pd.DataFrame({
                "website": ["https://example.com", "https://google.com"],
            })
        )
        schema = ar.Schema({"website": ar.URL(allowed_schemes=["https"])})
        result = ar.validate(frame, schema)
        assert result.passed

    def test_allowed_schemes_multiple_schemes(self):
        """Test allowed_schemes with multiple schemes."""
        frame = ar.from_pandas(
            pd.DataFrame({
                "website": ["https://example.com", "ftp://files.example.com"],
            })
        )
        schema = ar.Schema({"website": ar.URL(allowed_schemes=["https", "ftp"])})
        result = ar.validate(frame, schema)
        assert result.passed

    def test_invalid_url_format_fails_validation(self):
        """Test that invalid URL formats fail validation."""
        frame = ar.from_pandas(
            pd.DataFrame({
                "website": ["not a url", "example"],
            })
        )
        schema = ar.Schema({"website": ar.URL(nullable=False)})
        result = ar.validate(frame, schema)
        assert not result.passed

    def test_warning_severity_does_not_fail_validation(self):
        """Test that warning severity level reports issues without failing."""
        frame = ar.from_pandas(
            pd.DataFrame({
                "website": ["invalid-url"],
            })
        )
        schema = ar.Schema({"website": ar.URL(severity="warning")})
        result = ar.validate(frame, schema)
        assert result.passed
        assert result.issue_count > 0

    def test_unique_constraint_on_urls(self):
        """Test that unique=True enforces uniqueness."""
        frame = ar.from_pandas(
            pd.DataFrame({
                "website": ["https://example.com", "https://example.com"],
            })
        )
        schema = ar.Schema({"website": ar.URL(nullable=False, unique=True)})
        result = ar.validate(frame, schema)
        assert not result.passed
