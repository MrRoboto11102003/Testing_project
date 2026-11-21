import pytest
from arnio.schema import PhoneNumber, URL


class TestPhoneNumber:
    """Tests for arnio.schema.PhoneNumber field creation."""
    
    def test_defaults(self):
        """PhoneNumber creates Field with correct defaults."""
        field = PhoneNumber()
        assert field.dtype == "string"
        assert field.semantic == "phone"
        assert field.nullable is True
        assert field.unique is False
        assert field.severity == "error"
    
    def test_nullable_false(self):
        """PhoneNumber respects nullable=False."""
        field = PhoneNumber(nullable=False)
        assert field.nullable is False
    
    def test_unique_true(self):
        """PhoneNumber respects unique=True."""
        field = PhoneNumber(unique=True)
        assert field.unique is True
    
    def test_custom_severity(self):
        """PhoneNumber respects custom severity."""
        field = PhoneNumber(severity="warning")
        assert field.severity == "warning"
    
    def test_required_if(self):
        """PhoneNumber respects required_if parameter."""
        field = PhoneNumber(required_if=("country", "US"))
        assert field.required_if == ("country", "US")


class TestURL:
    """Tests for arnio.schema.URL field creation."""
    
    def test_defaults(self):
        """URL creates Field with correct defaults."""
        field = URL()
        assert field.dtype == "string"
        assert field.semantic == "url"
        assert field.nullable is True
        assert field.unique is False
    
    def test_single_allowed_scheme(self):
        """URL with single allowed scheme."""
        field = URL(allowed_schemes=["https"])
        assert field.semantic == "url:https"
    
    def test_multiple_schemes_sorted(self):
        """URL sorts multiple allowed schemes."""
        field = URL(allowed_schemes=["https", "http"])
        assert field.semantic == "url:http|https"
    
    def test_allowed_schemes_as_tuple_and_set(self):
        """URL accepts allowed_schemes as tuple or set."""
        field_tuple = URL(allowed_schemes=("ftp", "https"))
        field_set = URL(allowed_schemes={"https"})
        assert "url:" in field_tuple.semantic
        assert field_set.semantic == "url:https"
    
    def test_bare_string_raises_typeerror(self):
        """URL rejects bare string for allowed_schemes."""
        with pytest.raises(TypeError, match="must be a sequence of scheme names, not a bare string"):
            URL(allowed_schemes="https")
    
    def test_empty_or_invalid_schemes_raise_valueerror(self):
        """URL rejects empty strings, non-strings, and invalid schemes."""
        with pytest.raises(ValueError, match="non-empty strings"):
            URL(allowed_schemes=["https", ""])
        
        with pytest.raises(ValueError, match="non-empty strings"):
            URL(allowed_schemes=[123])
        
        with pytest.raises(ValueError, match="URL scheme names"):
            URL(allowed_schemes=["https://"])
    
    def test_empty_sequence_raises_valueerror(self):
        """URL rejects empty allowed_schemes."""
        with pytest.raises(ValueError, match="non-empty sequence"):
            URL(allowed_schemes=[])
    
    def test_scheme_whitespace_stripped(self):
        """URL strips whitespace from scheme names."""
        field = URL(allowed_schemes=["  https  "])
        assert field.semantic == "url:https"
    
    def test_url_parameters_with_schemes(self):
        """URL combines allowed_schemes with other parameters."""
        field = URL(
            nullable=False,
            unique=True,
            severity="warning",
            required_if=("verified", True),
            allowed_schemes=["https"]
        )
        assert field.nullable is False
        assert field.unique is True
        assert field.severity == "warning"
        assert field.required_if == ("verified", True)
        assert field.semantic == "url:https"
