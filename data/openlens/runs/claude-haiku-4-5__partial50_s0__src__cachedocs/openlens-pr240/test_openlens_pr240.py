import pytest
from openlens.validation import validate_lens_name, validate_parabolic_sag, ValidationError


class TestValidateLensName:
    """Tests for validate_lens_name function."""
    
    def test_valid_simple_name(self):
        """Test that a simple valid name is returned unchanged."""
        assert validate_lens_name("My Lens") == "My Lens"
    
    def test_empty_string_returns_untitled(self):
        """Test that empty string returns 'Untitled'."""
        assert validate_lens_name("") == "Untitled"
    
    def test_whitespace_only_returns_untitled(self):
        """Test that whitespace-only string returns 'Untitled'."""
        assert validate_lens_name("   ") == "Untitled"
    
    def test_strips_leading_trailing_whitespace(self):
        """Test that leading and trailing whitespace is stripped."""
        assert validate_lens_name("  My Lens  ") == "My Lens"
    
    def test_name_exactly_100_chars_is_valid(self):
        """Test that a name of exactly 100 characters is valid."""
        name_100 = "a" * 100
        assert validate_lens_name(name_100) == name_100
    
    def test_name_over_100_chars_raises_error(self):
        """Test that a name over 100 characters raises ValidationError."""
        with pytest.raises(ValidationError):
            validate_lens_name("a" * 101)
    
    def test_non_string_type_raises_error(self):
        """Test that non-string input raises ValidationError."""
        with pytest.raises(ValidationError):
            validate_lens_name(123)
    
    def test_name_with_special_characters(self):
        """Test that names with special characters are accepted."""
        assert validate_lens_name("Lens (50mm) f/2.8") == "Lens (50mm) f/2.8"


class TestValidateParabolicSag:
    """Tests for validate_parabolic_sag function."""
    
    def test_zero_sag_is_valid(self):
        """Test that zero sag is accepted."""
        assert validate_parabolic_sag(0.0) == 0.0
    
    def test_sag_at_positive_legacy_limit(self):
        """Test that sag at exactly 100 mm is valid."""
        assert validate_parabolic_sag(100.0) == 100.0
    
    def test_sag_at_negative_legacy_limit(self):
        """Test that sag at exactly -100 mm is valid."""
        assert validate_parabolic_sag(-100.0) == -100.0
    
    def test_sag_exceeds_positive_legacy_limit_raises_error(self):
        """Test that sag exceeding 100 mm raises ValidationError."""
        with pytest.raises(ValidationError):
            validate_parabolic_sag(100.1)
    
    def test_sag_exceeds_negative_legacy_limit_raises_error(self):
        """Test that sag exceeding -100 mm raises ValidationError."""
        with pytest.raises(ValidationError):
            validate_parabolic_sag(-100.1)
    
    def test_non_numeric_sag_raises_error(self):
        """Test that non-numeric sag raises ValidationError."""
        with pytest.raises(ValidationError):
            validate_parabolic_sag("not_a_number")
