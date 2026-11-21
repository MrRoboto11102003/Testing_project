import pytest
import re
import pandas as pd
import arnio as ar
from dataclasses import FrozenInstanceError


class TestPhoneNumberField:
    """Tests for ar.PhoneNumber factory function."""

    def test_phone_number_default_creation(self):
        """PhoneNumber() creates a Field with correct defaults."""
        field = ar.PhoneNumber()
        assert field.dtype == "string"
        assert field.semantic == "phone"
        assert field.nullable is True
        assert field.unique is False
        assert field.severity == "error"
        assert field.required_if is None

    def test_phone_number_with_all_options(self):
        """PhoneNumber respects all parameters."""
        field = ar.PhoneNumber(
            nullable=False,
            unique=True,
            severity="warning",
            required_if=("country", "US")
        )
        assert field.nullable is False
        assert field.unique is True
        assert field.severity == "warning"
        assert field.required_if == ("country", "US")

    def test_phone_number_invalid_nullable_type(self):
        """PhoneNumber raises TypeError for non-bool nullable."""
        with pytest.raises(TypeError, match=re.escape("nullable must be a bool")):
            ar.PhoneNumber(nullable=1)

    def test_phone_number_invalid_unique_type(self):
        """PhoneNumber raises TypeError for non-bool unique."""
        with pytest.raises(TypeError, match=re.escape("unique must be a bool")):
            ar.PhoneNumber(unique="yes")

    def test_phone_number_invalid_severity_type(self):
        """PhoneNumber raises TypeError for non-string severity."""
        with pytest.raises(TypeError, match=re.escape("severity must be a string")):
            ar.PhoneNumber(severity=1)

    def test_phone_number_invalid_severity_value(self):
        """PhoneNumber raises ValueError for invalid severity."""
        with pytest.raises(ValueError, match=re.escape("severity must be 'error' or 'warning'")):
            ar.PhoneNumber(severity="info")

    def test_phone_number_field_is_frozen(self):
        """PhoneNumber returns a frozen Field that cannot be modified."""
        field = ar.PhoneNumber()
        with pytest.raises(FrozenInstanceError):
            field.nullable = False


class TestPhoneNumberValidation:
    """Integration tests for PhoneNumber validation."""

    def test_phone_number_valid_formats(self):
        """Valid phone numbers pass validation."""
        df = pd.DataFrame({
            "phone": ["555-123-4567", "+1 (555) 123-4567", "12345678"]
        })
        frame = ar.from_pandas(df)
        result = ar.validate(frame, ar.Schema({"phone": ar.PhoneNumber()}))
        phone_issues = [i for i in result.issues if i.rule == "phone"]
        assert len(phone_issues) == 0

    def test_phone_number_invalid_formats(self):
        """Invalid phone numbers fail validation."""
        df = pd.DataFrame({"phone": ["5551234", "abc"]})
        frame = ar.from_pandas(df)
        result = ar.validate(frame, ar.Schema({"phone": ar.PhoneNumber()}))
        phone_issues = [i for i in result.issues if i.rule == "phone"]
        assert len(phone_issues) == 2


class TestURLField:
    """Tests for ar.URL factory function."""

    def test_url_default_creation(self):
        """URL() creates a Field with correct defaults."""
        field = ar.URL()
        assert field.dtype == "string"
        assert field.semantic == "url"
        assert field.nullable is True
        assert field.unique is False
        assert field.severity == "error"
        assert field.required_if is None

    def test_url_allowed_schemes_single(self):
        """URL with single allowed_schemes sets semantic correctly."""
        field = ar.URL(allowed_schemes=["https"])
        assert field.semantic == "url:https"

    def test_url_allowed_schemes_multiple_sorted(self):
        """URL with multiple schemes sorts them alphabetically."""
        field = ar.URL(allowed_schemes=["https", "http"])
        assert field.semantic == "url:http|https"

    def test_url_allowed_schemes_whitespace_stripped(self):
        """URL strips whitespace from scheme names."""
        field = ar.URL(allowed_schemes=[" ftp "])
        assert field.semantic == "url:ftp"

    def test_url_allowed_schemes_bare_string_raises(self):
        """URL raises TypeError if allowed_schemes is a bare string."""
        with pytest.raises(
            TypeError,
            match=re.escape("allowed_schemes must be a sequence of scheme names, not a bare string")
        ):
            ar.URL(allowed_schemes="https")

    def test_url_allowed_schemes_non_sequence_raises(self):
        """URL raises TypeError for non-sequence allowed_schemes."""
        with pytest.raises(
            TypeError,
            match=re.escape("allowed_schemes must be a sequence of scheme names")
        ):
            ar.URL(allowed_schemes={"https": 1})

    def test_url_allowed_schemes_non_string_item_raises(self):
        """URL raises ValueError for non-string items."""
        with pytest.raises(ValueError, match=re.escape("allowed_schemes must contain non-empty strings")):
            ar.URL(allowed_schemes=[1])

    def test_url_allowed_schemes_empty_string_raises(self):
        """URL raises ValueError for empty strings in allowed_schemes."""
        with pytest.raises(ValueError, match=re.escape("allowed_schemes must contain non-empty strings")):
            ar.URL(allowed_schemes=["  "])

    def test_url_allowed_schemes_invalid_format_raises(self):
        """URL raises ValueError for invalid scheme format."""
        with pytest.raises(
            ValueError,
            match=re.escape("allowed_schemes must contain URL scheme names such as 'http' or 'https'")
        ):
            ar.URL(allowed_schemes=["1http"])

    def test_url_allowed_schemes_empty_sequence_raises(self):
        """URL raises ValueError for empty allowed_schemes."""
        with pytest.raises(ValueError, match=re.escape("allowed_schemes must be a non-empty sequence")):
            ar.URL(allowed_schemes=[])

    def test_url_invalid_severity_type(self):
        """URL raises TypeError for non-string severity."""
        with pytest.raises(TypeError, match=re.escape("severity must be a string")):
            ar.URL(severity=1)

    def test_url_field_is_frozen(self):
        """URL returns a frozen Field that cannot be modified."""
        field = ar.URL()
        with pytest.raises(FrozenInstanceError):
            field.semantic = "other"


class TestURLValidation:
    """Integration tests for URL validation."""

    def test_url_valid_formats(self):
        """Valid URLs pass validation."""
        df = pd.DataFrame({
            "site": ["http://example.com", "https://test.org"]
        })
        frame = ar.from_pandas(df)
        result = ar.validate(frame, ar.Schema({"site": ar.URL()}))
        url_issues = [i for i in result.issues if i.rule == "url"]
        assert len(url_issues) == 0

    def test_url_scheme_restriction(self):
        """URL with allowed_schemes restricts valid schemes."""
        df = pd.DataFrame({
            "site": ["https://example.com", "http://example.com", "ftp://test.org"]
        })
        frame = ar.from_pandas(df)
        result = ar.validate(frame, ar.Schema({"site": ar.URL(allowed_schemes=["https"])}))
        url_issues = [i for i in result.issues if "url" in i.rule]
        assert len(url_issues) == 2  # http and ftp should fail
