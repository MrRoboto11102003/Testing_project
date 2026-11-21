import pytest
from arnio.schema import PhoneNumber, URL, Field


class TestPhoneNumber:
    """Tests for arnio.schema.PhoneNumber"""
    
    def test_returns_field_with_phone_semantic(self):
        """PhoneNumber returns Field with phone semantic"""
        field = PhoneNumber()
        assert isinstance(field, Field)
        assert field.dtype == "string"
        assert field.semantic == "phone"
        assert field.nullable is True
        assert field.unique is False
        assert field.severity == "error"
    
    def test_respects_nullable_parameter(self):
        """PhoneNumber respects nullable parameter"""
        assert PhoneNumber(nullable=False).nullable is False
        assert PhoneNumber(nullable=True).nullable is True
    
    def test_respects_unique_parameter(self):
        """PhoneNumber respects unique parameter"""
        assert PhoneNumber(unique=True).unique is True
        assert PhoneNumber(unique=False).unique is False
    
    def test_respects_severity_parameter(self):
        """PhoneNumber respects severity parameter"""
        assert PhoneNumber(severity="warning").severity == "warning"
    
    def test_respects_required_if_parameter(self):
        """PhoneNumber respects required_if parameter"""
        condition = ("country", "US")
        assert PhoneNumber(required_if=condition).required_if == condition


class TestURL:
    """Tests for arnio.schema.URL"""
    
    def test_returns_field_with_url_semantic(self):
        """URL returns Field with url semantic"""
        field = URL()
        assert isinstance(field, Field)
        assert field.dtype == "string"
        assert field.semantic == "url"
        assert field.nullable is True
        assert field.unique is False
        assert field.severity == "error"
    
    def test_allowed_schemes_single_in_semantic(self):
        """URL includes single scheme in semantic"""
        field = URL(allowed_schemes=["https"])
        assert field.semantic == "url:https"
    
    def test_allowed_schemes_multiple_sorted_in_semantic(self):
        """URL includes multiple schemes sorted in semantic"""
        field = URL(allowed_schemes=["ftp", "https", "http"])
        assert field.semantic == "url:ftp|http|https"
    
    def test_allowed_schemes_accepts_list_tuple_set(self):
        """URL accepts list, tuple, and set for allowed_schemes"""
        URL(allowed_schemes=["http"])
        URL(allowed_schemes=("http",))
        URL(allowed_schemes={"http"})
    
    def test_allowed_schemes_rejects_string_and_bytes(self):
        """URL rejects bare string and bytes for allowed_schemes"""
        with pytest.raises(TypeError):
            URL(allowed_schemes="https")
        with pytest.raises(TypeError):
            URL(allowed_schemes=b"https")
    
    def test_allowed_schemes_rejects_empty_list(self):
        """URL rejects empty allowed_schemes"""
        with pytest.raises(ValueError):
            URL(allowed_schemes=[])
    
    def test_allowed_schemes_rejects_empty_or_invalid_strings(self):
        """URL rejects empty/whitespace/non-string in allowed_schemes"""
        with pytest.raises(ValueError):
            URL(allowed_schemes=[""])
        with pytest.raises(ValueError):
            URL(allowed_schemes=["   "])
        with pytest.raises(ValueError):
            URL(allowed_schemes=[123])
    
    def test_allowed_schemes_rejects_invalid_scheme_pattern(self):
        """URL rejects scheme that doesn't match URL scheme pattern"""
        with pytest.raises(ValueError):
            URL(allowed_schemes=["9invalid"])
    
    def test_respects_other_parameters(self):
        """URL respects nullable, unique, severity, required_if"""
        field = URL(
            nullable=False,
            unique=True,
            severity="warning",
            required_if=("type", "external")
        )
        assert field.nullable is False
        assert field.unique is True
        assert field.severity == "warning"
        assert field.required_if == ("type", "external")
