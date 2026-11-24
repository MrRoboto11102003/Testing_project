import pytest
import re
import arnio as ar


class TestPhoneNumber:
    """Test arnio.schema.PhoneNumber factory function."""
    
    def test_field_structure(self):
        """PhoneNumber returns Field with dtype='string', semantic='phone'."""
        field = ar.PhoneNumber()
        assert field.dtype == "string"
        assert field.semantic == "phone"
        assert field.nullable is True
        assert field.unique is False
        assert field.severity == "error"
    
    def test_parameters_accepted(self):
        """PhoneNumber accepts and passes parameters to Field."""
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
    
    def test_nullable_must_be_bool(self):
        """nullable parameter must be bool."""
        with pytest.raises(TypeError, match=re.escape("nullable must be a bool")):
            ar.PhoneNumber(nullable=1)
    
    def test_unique_must_be_bool(self):
        """unique parameter must be bool."""
        with pytest.raises(TypeError, match=re.escape("unique must be a bool")):
            ar.PhoneNumber(unique="yes")
    
    def test_severity_type_validation(self):
        """severity parameter must be string."""
        with pytest.raises(TypeError, match=re.escape("severity must be a string")):
            ar.PhoneNumber(severity=1)
    
    def test_severity_value_validation(self):
        """severity parameter must be 'error' or 'warning'."""
        with pytest.raises(ValueError, match=re.escape("severity must be 'error' or 'warning'")):
            ar.PhoneNumber(severity="info")
    
    def test_required_if_validation(self):
        """required_if parameter must be tuple or None."""
        with pytest.raises(TypeError, match=re.escape("required_if must be a tuple or None")):
            ar.PhoneNumber(required_if=["col", "val"])


class TestURL:
    """Test arnio.schema.URL factory function."""
    
    def test_field_structure_default(self):
        """URL with no schemes returns Field with semantic='url'."""
        field = ar.URL()
        assert field.dtype == "string"
        assert field.semantic == "url"
        assert field.nullable is True
        assert field.unique is False
        assert field.severity == "error"
    
    def test_allowed_schemes_single(self):
        """allowed_schemes=['https'] creates semantic='url:https'."""
        field = ar.URL(allowed_schemes=["https"])
        assert field.semantic == "url:https"
    
    def test_allowed_schemes_multiple_sorted(self):
        """Multiple schemes are sorted in semantic string."""
        field = ar.URL(allowed_schemes=["https", "http"])
        assert field.semantic == "url:http|https"
    
    def test_allowed_schemes_bare_string_rejected(self):
        """allowed_schemes as bare string raises TypeError."""
        with pytest.raises(TypeError, match=re.escape("allowed_schemes must be a sequence of scheme names, not a bare string")):
            ar.URL(allowed_schemes="https")
    
    def test_allowed_schemes_non_sequence_rejected(self):
        """allowed_schemes as dict raises TypeError."""
        with pytest.raises(TypeError, match=re.escape("allowed_schemes must be a sequence of scheme names")):
            ar.URL(allowed_schemes={"https": 1})
    
    def test_allowed_schemes_non_string_item_rejected(self):
        """Non-string item in allowed_schemes raises ValueError."""
        with pytest.raises(ValueError, match=re.escape("allowed_schemes must contain non-empty strings")):
            ar.URL(allowed_schemes=[1])
    
    def test_allowed_schemes_invalid_format_rejected(self):
        """Invalid scheme format in allowed_schemes raises ValueError."""
        with pytest.raises(ValueError, match=re.escape("allowed_schemes must contain URL scheme names such as 'http' or 'https'")):
            ar.URL(allowed_schemes=["1http"])
    
    def test_allowed_schemes_empty_sequence_rejected(self):
        """Empty allowed_schemes sequence raises ValueError."""
        with pytest.raises(ValueError, match=re.escape("allowed_schemes must be a non-empty sequence")):
            ar.URL(allowed_schemes=[])
