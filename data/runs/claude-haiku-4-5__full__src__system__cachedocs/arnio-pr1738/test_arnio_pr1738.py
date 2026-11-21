"""End-to-end tests for arnio.schema.PhoneNumber and URL field validation.

Tests drive complete user workflows through arnio's public API:
- Load CSV files with real data
- Create schemas with PhoneNumber and URL fields
- Validate frames and inspect results
- Use pandas .arnio accessor
- Test error handling and edge cases
"""

import csv
import sys
from pathlib import Path

import pandas as pd
import pytest

import arnio as ar


class TestPhoneNumberField:
    """Test arnio.schema.PhoneNumber field validation."""

    def test_phonenumber_valid_international_formats(self, tmp_path):
        """PhoneNumber accepts common international phone number formats."""
        csv_file = tmp_path / "phones.csv"
        csv_file.write_text(
            "id,phone\n"
            "1,+1-555-123-4567\n"
            "2,+91 9876543210\n"
            "3,5551234567\n"
        )

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({"phone": ar.PhoneNumber(nullable=False)})
        result = ar.validate(frame, schema)

        assert result.passed, f"Expected validation to pass but got issues: {result.issues}"
        assert len(result.issues) == 0

    def test_phonenumber_rejects_invalid_formats(self, tmp_path):
        """PhoneNumber validation fails on clearly invalid input."""
        csv_file = tmp_path / "phones.csv"
        csv_file.write_text(
            "id,phone\n"
            "1,not-a-phone\n"
            "2,abc123\n"
        )

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({"phone": ar.PhoneNumber(nullable=False)})
        result = ar.validate(frame, schema)

        assert not result.passed
        assert len(result.issues) > 0
        # Verify row_index is 1-based (data rows only, header excluded)
        assert all(isinstance(issue.row_index, int) and issue.row_index >= 1 for issue in result.issues)

    def test_phonenumber_nullable_true_accepts_empty(self, tmp_path):
        """PhoneNumber with nullable=True allows null/empty values."""
        csv_file = tmp_path / "phones.csv"
        csv_file.write_text(
            "id,phone\n"
            "1,+1-555-123-4567\n"
            "2,\n"
            "3,+91 9876543210\n"
        )

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({"phone": ar.PhoneNumber(nullable=True)})
        result = ar.validate(frame, schema)

        assert result.passed

    def test_phonenumber_nullable_false_rejects_empty(self, tmp_path):
        """PhoneNumber with nullable=False rejects empty values."""
        csv_file = tmp_path / "phones.csv"
        csv_file.write_text(
            "id,phone\n"
            "1,+1-555-123-4567\n"
            "2,\n"
        )

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({"phone": ar.PhoneNumber(nullable=False)})
        result = ar.validate(frame, schema)

        assert not result.passed
        assert len(result.issues) >= 1

    def test_phonenumber_unique_constraint_detects_duplicates(self, tmp_path):
        """PhoneNumber with unique=True rejects duplicate phone numbers."""
        csv_file = tmp_path / "phones.csv"
        csv_file.write_text(
            "id,phone\n"
            "1,+1-555-123-4567\n"
            "2,+1-555-123-4567\n"
        )

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({"phone": ar.PhoneNumber(nullable=False, unique=True)})
        result = ar.validate(frame, schema)

        assert not result.passed

    def test_phonenumber_via_pandas_accessor(self):
        """PhoneNumber validation works through pandas .arnio accessor."""
        df = pd.DataFrame({
            "phone": ["+1-555-123-4567", "+91 9876543210", "5551234567"],
        })

        schema = ar.Schema({"phone": ar.PhoneNumber(nullable=False)})
        result = df.arnio.validate(schema)

        assert result.passed


class TestURLField:
    """Test arnio.schema.URL field validation."""

    def test_url_accepts_http_and_https_by_default(self, tmp_path):
        """URL validates standard HTTP and HTTPS URLs."""
        csv_file = tmp_path / "urls.csv"
        csv_file.write_text(
            "id,site\n"
            "1,https://example.com\n"
            "2,http://github.com/path\n"
            "3,https://example.com:8080/path?query=1&other=2\n"
        )

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({"site": ar.URL(nullable=False)})
        result = ar.validate(frame, schema)

        assert result.passed
        assert len(result.issues) == 0

    def test_url_rejects_malformed_input(self, tmp_path):
        """URL validation fails on malformed or incomplete URLs."""
        csv_file = tmp_path / "urls.csv"
        csv_file.write_text(
            "id,site\n"
            "1,not a url at all\n"
            "2,example.com\n"
        )

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({"site": ar.URL(nullable=False)})
        result = ar.validate(frame, schema)

        assert not result.passed
        assert len(result.issues) > 0

    def test_url_nullable_true_accepts_empty(self, tmp_path):
        """URL with nullable=True allows null/empty values."""
        csv_file = tmp_path / "urls.csv"
        csv_file.write_text(
            "id,site\n"
            "1,https://example.com\n"
            "2,\n"
            "3,http://github.com\n"
        )

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({"site": ar.URL(nullable=True)})
        result = ar.validate(frame, schema)

        assert result.passed

    def test_url_allowed_schemes_https_only_rejects_http(self, tmp_path):
        """URL with allowed_schemes=['https'] rejects http:// URLs."""
        csv_file = tmp_path / "urls.csv"
        csv_file.write_text(
            "id,site\n"
            "1,https://secure.example.com\n"
            "2,http://insecure.example.com\n"
        )

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({"site": ar.URL(nullable=False, allowed_schemes=["https"])})
        result = ar.validate(frame, schema)

        assert not result.passed
        # Second row should trigger a scheme violation
        assert any(issue.row_index == 2 for issue in result.issues)

    def test_url_allowed_schemes_multiple_schemes(self, tmp_path):
        """URL accepts data when multiple schemes are explicitly allowed."""
        csv_file = tmp_path / "urls.csv"
        csv_file.write_text(
            "id,site\n"
            "1,https://example.com\n"
            "2,ftp://files.example.com\n"
            "3,ftps://secure-files.example.com\n"
        )

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({
            "site": ar.URL(nullable=False, allowed_schemes=["https", "ftp", "ftps"])
        })
        result = ar.validate(frame, schema)

        assert result.passed

    def test_url_allowed_schemes_bare_string_raises_typeerror(self):
        """URL raises TypeError if allowed_schemes is a bare string instead of list."""
        with pytest.raises(TypeError, match="must be a sequence"):
            ar.URL(allowed_schemes="https")

    def test_url_allowed_schemes_empty_list_raises_valueerror(self):
        """URL raises ValueError if allowed_schemes is an empty sequence."""
        with pytest.raises(ValueError, match="non-empty"):
            ar.URL(allowed_schemes=[])

    def test_url_allowed_schemes_empty_strings_raises_valueerror(self):
        """URL raises ValueError if allowed_schemes contains empty strings."""
        with pytest.raises(ValueError, match="non-empty strings"):
            ar.URL(allowed_schemes=["https", "", "ftp"])

    def test_url_via_pandas_accessor(self):
        """URL validation works through pandas .arnio accessor."""
        df = pd.DataFrame({
            "site": ["https://example.com", "http://github.com", "https://example.org"],
        })

        schema = ar.Schema({"site": ar.URL(nullable=False)})
        result = df.arnio.validate(schema)

        assert result.passed


class TestMixedPhoneAndURLSchema:
    """Test validation of schemas combining PhoneNumber and URL fields."""

    def test_mixed_schema_all_fields_valid(self, tmp_path):
        """Mixed schema validates frames with valid phone and URL data."""
        csv_file = tmp_path / "contacts.csv"
        csv_file.write_text(
            "id,phone,website\n"
            "1,+1-555-123-4567,https://example.com\n"
            "2,+91 9876543210,https://example.org\n"
        )

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({
            "phone": ar.PhoneNumber(nullable=False),
            "website": ar.URL(nullable=False),
        })
        result = ar.validate(frame, schema)

        assert result.passed

    def test_mixed_schema_phone_field_fails(self, tmp_path):
        """Mixed schema detects invalid phone numbers."""
        csv_file = tmp_path / "contacts.csv"
        csv_file.write_text(
            "id,phone,website\n"
            "1,invalid-phone,https://example.com\n"
        )

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({
            "phone": ar.PhoneNumber(nullable=False),
            "website": ar.URL(nullable=False),
        })
        result = ar.validate(frame, schema)

        assert not result.passed
        phone_issues = [i for i in result.issues if i.column == "phone"]
        assert len(phone_issues) > 0

    def test_mixed_schema_url_field_fails(self, tmp_path):
        """Mixed schema detects invalid URLs."""
        csv_file = tmp_path / "contacts.csv"
        csv_file.write_text(
            "id,phone,website\n"
            "1,+1-555-123-4567,not-a-url\n"
        )

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({
            "phone": ar.PhoneNumber(nullable=False),
            "website": ar.URL(nullable=False),
        })
        result = ar.validate(frame, schema)

        assert not result.passed
        url_issues = [i for i in result.issues if i.column == "website"]
        assert len(url_issues) > 0

    def test_validation_result_provides_actionable_summary(self, tmp_path):
        """ValidationResult provides structured access to issues via summary()."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            "id,phone,site\n"
            "1,+1-555-123-4567,https://example.com\n"
            "2,bad-phone,bad-url\n"
        )

        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({
            "phone": ar.PhoneNumber(nullable=False),
            "site": ar.URL(nullable=False),
        })
        result = ar.validate(frame, schema)

        assert not result.passed
        summary = result.summary()
        
        # Summary provides structured keys for issue organization
        assert isinstance(summary, dict)
        # Issues are accessible and row indices are correct (1-based)
        assert all(1 <= issue.row_index for issue in result.issues)
        # Can convert to markdown/pandas for reporting
        markdown = result.to_markdown()
        assert isinstance(markdown, str)
        assert len(markdown) > 0
