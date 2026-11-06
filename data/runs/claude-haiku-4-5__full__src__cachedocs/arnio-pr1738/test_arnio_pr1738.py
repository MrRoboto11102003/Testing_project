import pytest
import arnio as ar


class TestPhoneNumber:
    """Tests for arnio.schema.PhoneNumber factory function."""

    def test_creates_field_with_defaults(self):
        """PhoneNumber() returns a Field with correct defaults."""
        field = ar.PhoneNumber()
        assert field.dtype == "string"
        assert field.nullable is True
        assert field.unique is False
        assert field.severity == "error"
        assert field.semantic == "phone"

    def test_nullable_parameter(self):
        """PhoneNumber(nullable=False) creates non-nullable field."""
        field = ar.PhoneNumber(nullable=False)
        assert field.nullable is False
        assert field.dtype == "string"
        assert field.semantic == "phone"

    def test_unique_parameter(self):
        """PhoneNumber(unique=True) creates field with unique constraint."""
        field = ar.PhoneNumber(unique=True)
        assert field.unique is True

    def test_severity_parameter(self):
        """PhoneNumber(severity='warning') sets warning-level severity."""
        field = ar.PhoneNumber(severity="warning")
        assert field.severity == "warning"

    def test_required_if_parameter(self):
        """PhoneNumber(required_if=...) sets conditional requirement."""
        field = ar.PhoneNumber(required_if=("user_type", "customer"))
        assert field.required_if == ("user_type", "customer")

    def test_all_parameters_combined(self):
        """PhoneNumber with all parameters set."""
        field = ar.PhoneNumber(
            nullable=False,
            unique=True,
            severity="warning",
            required_if=("status", "active"),
        )
        assert field.dtype == "string"
        assert field.semantic == "phone"
        assert field.nullable is False
        assert field.unique is True
        assert field.severity == "warning"
        assert field.required_if == ("status", "active")


class TestURL:
    """Tests for arnio.schema.URL factory function."""

    def test_creates_field_with_defaults(self):
        """URL() returns a Field with correct defaults."""
        field = ar.URL()
        assert field.dtype == "string"
        assert field.nullable is True
        assert field.unique is False
        assert field.severity == "error"
        assert field.semantic == "url"

    def test_nullable_parameter(self):
        """URL(nullable=False) creates non-nullable field."""
        field = ar.URL(nullable=False)
        assert field.nullable is False

    def test_unique_parameter(self):
        """URL(unique=True) creates field with unique constraint."""
        field = ar.URL(unique=True)
        assert field.unique is True

    def test_severity_parameter(self):
        """URL(severity='warning') sets warning-level severity."""
        field = ar.URL(severity="warning")
        assert field.severity == "warning"

    def test_required_if_parameter(self):
        """URL(required_if=...) sets conditional requirement."""
        field = ar.URL(required_if=("has_website", True))
        assert field.required_if == ("has_website", True)

    def test_allowed_schemes_single_scheme(self):
        """URL(allowed_schemes=['https']) restricts to HTTPS only."""
        field = ar.URL(allowed_schemes=["https"])
        assert field.semantic != "url"
        assert "https" in field.semantic
        assert "http" not in field.semantic or field.semantic.count("http") == 1

    def test_allowed_schemes_multiple_schemes(self):
        """URL(allowed_schemes=['http', 'https']) allows both schemes."""
        field = ar.URL(allowed_schemes=["http", "https"])
        assert "http" in field.semantic
        assert "https" in field.semantic

    def test_allowed_schemes_whitespace_stripping(self):
        """URL allows schemes with leading/trailing whitespace."""
        field = ar.URL(allowed_schemes=["  https  "])
        assert "https" in field.semantic

    def test_allowed_schemes_bare_string_raises_typeerror(self):
        """URL(allowed_schemes='https') raises TypeError."""
        with pytest.raises(TypeError, match="must be a sequence"):
            ar.URL(allowed_schemes="https")

    def test_allowed_schemes_invalid_type_raises_typeerror(self):
        """URL(allowed_schemes=42) raises TypeError."""
        with pytest.raises(TypeError, match="must be a sequence"):
            ar.URL(allowed_schemes=42)

    def test_allowed_schemes_non_string_element_raises_valueerror(self):
        """URL(allowed_schemes=['https', 123]) raises ValueError."""
        with pytest.raises(ValueError, match="must contain non-empty strings"):
            ar.URL(allowed_schemes=["https", 123])

    def test_allowed_schemes_empty_string_raises_valueerror(self):
        """URL(allowed_schemes=['']) raises ValueError."""
        with pytest.raises(ValueError, match="must contain non-empty strings"):
            ar.URL(allowed_schemes=[""])

    def test_allowed_schemes_invalid_scheme_name_raises_valueerror(self):
        """URL(allowed_schemes=['not a valid scheme!']) raises ValueError."""
        with pytest.raises(ValueError, match="URL scheme names"):
            ar.URL(allowed_schemes=["not a valid scheme!"])

    def test_allowed_schemes_empty_list_raises_valueerror(self):
        """URL(allowed_schemes=[]) raises ValueError."""
        with pytest.raises(ValueError, match="non-empty sequence"):
            ar.URL(allowed_schemes=[])

    def test_all_parameters_with_schemes(self):
        """URL with all parameters including allowed_schemes."""
        field = ar.URL(
            nullable=False,
            unique=True,
            severity="warning",
            required_if=("needs_url", True),
            allowed_schemes=["https", "ftp"],
        )
        assert field.dtype == "string"
        assert field.nullable is False
        assert field.unique is True
        assert field.severity == "warning"
        assert field.required_if == ("needs_url", True)
        assert "https" in field.semantic
        assert "ftp" in field.semantic
