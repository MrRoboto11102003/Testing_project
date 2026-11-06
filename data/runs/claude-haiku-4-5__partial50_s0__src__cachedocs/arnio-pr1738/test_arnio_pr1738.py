import pytest
import pandas as pd
import arnio as ar


class TestPhoneNumber:
    """Test cases for arnio.schema.PhoneNumber field creation."""

    def test_phonenumber_default_parameters(self):
        """Test PhoneNumber field creation with default parameters."""
        field = ar.PhoneNumber()
        assert field is not None

    def test_phonenumber_custom_parameters(self):
        """Test PhoneNumber field creation with custom parameters."""
        field = ar.PhoneNumber(
            nullable=False,
            unique=True,
            severity="warning",
            required_if=("user_type", "international"),
        )
        assert field is not None

    def test_phonenumber_in_schema_validation(self):
        """Test PhoneNumber field in schema validation with valid phone numbers."""
        frame = ar.from_pandas(
            pd.DataFrame(
                {
                    "phone": [
                        "+1-555-123-4567",
                        "+91 9876543210",
                        "5551234567",
                    ]
                }
            )
        )

        schema = ar.Schema({"phone": ar.PhoneNumber(nullable=False)})

        result = ar.validate(frame, schema)
        assert result.passed is True


class TestURL:
    """Test cases for arnio.schema.URL field creation."""

    def test_url_default_parameters(self):
        """Test URL field creation with default parameters."""
        field = ar.URL()
        assert field is not None

    def test_url_custom_parameters(self):
        """Test URL field creation with custom parameters."""
        field = ar.URL(
            nullable=False,
            unique=True,
            severity="warning",
            required_if=("has_website", True),
        )
        assert field is not None

    def test_url_allowed_schemes_single(self):
        """Test URL field with single allowed scheme."""
        field = ar.URL(allowed_schemes=["https"])
        assert field is not None

    def test_url_allowed_schemes_multiple(self):
        """Test URL field with multiple allowed schemes."""
        field = ar.URL(allowed_schemes=["https", "ftp", "http"])
        assert field is not None

    def test_url_allowed_schemes_bare_string_error(self):
        """Test that allowed_schemes rejects bare string."""
        with pytest.raises(TypeError, match="must be a sequence"):
            ar.URL(allowed_schemes="https")

    def test_url_allowed_schemes_empty_sequence_error(self):
        """Test that allowed_schemes rejects empty sequence."""
        with pytest.raises(ValueError, match="non-empty"):
            ar.URL(allowed_schemes=[])

    def test_url_allowed_schemes_empty_string_error(self):
        """Test that allowed_schemes rejects empty strings in sequence."""
        with pytest.raises(ValueError, match="non-empty strings"):
            ar.URL(allowed_schemes=["https", ""])

    def test_url_allowed_schemes_non_string_error(self):
        """Test that allowed_schemes rejects non-string elements."""
        with pytest.raises(ValueError, match="non-empty strings"):
            ar.URL(allowed_schemes=["https", 123])

    def test_url_allowed_schemes_invalid_format_error(self):
        """Test that allowed_schemes rejects invalid scheme names."""
        with pytest.raises(ValueError, match="URL scheme names"):
            ar.URL(allowed_schemes=["ht!tp://", "htt ps"])

    def test_url_in_schema_validation(self):
        """Test URL field in schema validation with default schemes."""
        frame = ar.from_pandas(
            pd.DataFrame({"website": ["https://example.com", "http://test.org"]})
        )

        schema = ar.Schema({"website": ar.URL(nullable=False)})

        result = ar.validate(frame, schema)
        assert result.passed is True

    def test_url_with_scheme_restriction(self):
        """Test URL field schema validation with scheme restriction."""
        frame = ar.from_pandas(
            pd.DataFrame({"website": ["https://example.com"]})
        )

        schema = ar.Schema({"website": ar.URL(allowed_schemes=["https"])})

        result = ar.validate(frame, schema)
        assert result.passed is True
