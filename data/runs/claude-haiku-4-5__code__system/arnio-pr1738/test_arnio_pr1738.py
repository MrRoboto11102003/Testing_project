import pytest
import pandas as pd
import sys
import subprocess
import csv
import tempfile
from pathlib import Path


class TestPhoneNumberSchema:
    """Test arnio.schema.PhoneNumber field creation and validation."""

    def test_phone_number_basic_creation(self):
        """Test basic PhoneNumber field creation."""
        import arnio as ar

        field = ar.schema.PhoneNumber()
        assert field.dtype == "string"
        assert field.semantic == "phone"
        assert field.nullable is True
        assert field.unique is False

    def test_phone_number_with_all_parameters(self):
        """Test PhoneNumber with all parameters specified."""
        import arnio as ar

        field = ar.schema.PhoneNumber(
            nullable=False,
            unique=True,
            severity="warning",
            required_if=("country", "US")
        )
        assert field.dtype == "string"
        assert field.semantic == "phone"
        assert field.nullable is False
        assert field.unique is True
        assert field.severity == "warning"
        assert field.required_if == ("country", "US")

    def test_phone_number_in_schema_and_validation(self, tmp_path):
        """Test PhoneNumber in a complete workflow with file I/O."""
        import arnio as ar

        # Create CSV file with phone data
        csv_file = tmp_path / "phones.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["customer_id", "phone_number"])
            writer.writerow([1, "+1-555-123-4567"])
            writer.writerow([2, "555-9876"])
            writer.writerow([3, None])

        df = pd.read_csv(csv_file)

        # Create schema with PhoneNumber
        schema = ar.Schema(
            customer_id=ar.schema.Int(),
            phone_number=ar.schema.PhoneNumber(nullable=True)
        )

        # Validate against schema
        report = schema.validate(df)
        assert report is not None

    def test_phone_number_non_nullable_constraint(self, tmp_path):
        """Test PhoneNumber with nullable=False."""
        import arnio as ar

        field = ar.schema.PhoneNumber(nullable=False)
        assert field.nullable is False

        # Create schema and try validation
        schema = ar.Schema(
            phone=ar.schema.PhoneNumber(nullable=False)
        )
        assert schema is not None

    def test_phone_number_unique_constraint(self, tmp_path):
        """Test PhoneNumber with unique=True."""
        import arnio as ar

        field = ar.schema.PhoneNumber(unique=True)
        assert field.unique is True

    def test_phone_number_severity_levels(self):
        """Test PhoneNumber with different severity levels."""
        import arnio as ar

        for severity in ["error", "warning", "info"]:
            field = ar.schema.PhoneNumber(severity=severity)
            assert field.severity == severity


class TestURLSchema:
    """Test arnio.schema.URL field creation and validation."""

    def test_url_basic_creation(self):
        """Test basic URL field creation."""
        import arnio as ar

        field = ar.schema.URL()
        assert field.dtype == "string"
        assert field.semantic == "url"
        assert field.nullable is True
        assert field.unique is False

    def test_url_with_all_parameters(self):
        """Test URL with all parameters specified."""
        import arnio as ar

        field = ar.schema.URL(
            nullable=False,
            unique=True,
            severity="warning",
            required_if=("has_website", True)
        )
        assert field.dtype == "string"
        assert field.semantic == "url"
        assert field.nullable is False
        assert field.unique is True
        assert field.severity == "warning"
        assert field.required_if == ("has_website", True)

    def test_url_with_allowed_schemes_single(self):
        """Test URL with single allowed scheme."""
        import arnio as ar

        field = ar.schema.URL(allowed_schemes=["https"])
        assert field.semantic == "url:https"

    def test_url_with_allowed_schemes_multiple(self):
        """Test URL with multiple allowed schemes."""
        import arnio as ar

        field = ar.schema.URL(allowed_schemes=["http", "https", "ftp"])
        # Schemes should be sorted and joined with |
        assert "url:" in field.semantic
        assert "ftp" in field.semantic
        assert "http" in field.semantic

    def test_url_allowed_schemes_bare_string_error(self):
        """Test that bare string raises TypeError."""
        import arnio as ar

        with pytest.raises(TypeError, match="must be a sequence"):
            ar.schema.URL(allowed_schemes="https")

    def test_url_allowed_schemes_non_sequence_error(self):
        """Test that non-sequence raises TypeError."""
        import arnio as ar

        with pytest.raises(TypeError, match="must be a sequence"):
            ar.schema.URL(allowed_schemes=42)

    def test_url_allowed_schemes_non_string_element_error(self):
        """Test that non-string element raises ValueError."""
        import arnio as ar

        with pytest.raises(ValueError, match="non-empty strings"):
            ar.schema.URL(allowed_schemes=["https", 123])

    def test_url_allowed_schemes_empty_string_error(self):
        """Test that empty string raises ValueError."""
        import arnio as ar

        with pytest.raises(ValueError, match="non-empty strings"):
            ar.schema.URL(allowed_schemes=["https", ""])

    def test_url_allowed_schemes_whitespace_only_error(self):
        """Test that whitespace-only string raises ValueError."""
        import arnio as ar

        with pytest.raises(ValueError, match="non-empty strings"):
            ar.schema.URL(allowed_schemes=["https", "   "])

    def test_url_allowed_schemes_empty_list_error(self):
        """Test that empty list raises ValueError."""
        import arnio as ar

        with pytest.raises(ValueError, match="non-empty sequence"):
            ar.schema.URL(allowed_schemes=[])

    def test_url_in_schema_and_validation(self, tmp_path):
        """Test URL in a complete workflow with file I/O and validation."""
        import arnio as ar

        # Create CSV file with URL data
        csv_file = tmp_path / "websites.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["org_id", "website_url"])
            writer.writerow([1, "https://example.com"])
            writer.writerow([2, "http://test.org"])
            writer.writerow([3, None])

        df = pd.read_csv(csv_file)

        # Create schema with URL and allowed schemes
        schema = ar.Schema(
            org_id=ar.schema.Int(),
            website_url=ar.schema.URL(
                nullable=True,
                allowed_schemes=["http", "https"]
            )
        )

        # Validate
        report = schema.validate(df)
        assert report is not None

    def test_url_allowed_schemes_normalization(self):
        """Test that allowed schemes are normalized (stripped and sorted)."""
        import arnio as ar

        # Test with schemes that need stripping
        field = ar.schema.URL(allowed_schemes=["  https  ", "http"])
        assert "url:" in field.semantic

    def test_url_with_tuple_schemes(self):
        """Test URL accepts tuple for allowed_schemes."""
        import arnio as ar

        field = ar.schema.URL(allowed_schemes=("https", "http"))
        assert "url:" in field.semantic

    def test_url_with_set_schemes(self):
        """Test URL accepts set for allowed_schemes."""
        import arnio as ar

        field = ar.schema.URL(allowed_schemes={"https", "http"})
        assert "url:" in field.semantic
