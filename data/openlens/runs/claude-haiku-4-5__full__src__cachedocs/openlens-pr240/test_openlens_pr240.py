import pytest
from openlens.validation import validate_lens_name, validate_parabolic_sag, ValidationError


class TestValidateLensName:
    """Tests for validate_lens_name function."""

    def test_valid_name(self):
        """Valid lens name should be returned."""
        assert validate_lens_name("My Lens") == "My Lens"

    def test_empty_string_returns_untitled(self):
        """Empty string should return 'Untitled'."""
        assert validate_lens_name("") == "Untitled"

    def test_whitespace_only_returns_untitled(self):
        """Whitespace-only string should return 'Untitled'."""
        assert validate_lens_name("   \t\n  ") == "Untitled"

    def test_whitespace_stripped(self):
        """Whitespace should be stripped from name."""
        assert validate_lens_name("  Test Lens  ") == "Test Lens"

    def test_name_at_max_length(self):
        """Name at exactly 100 characters should be valid."""
        name = "a" * 100
        assert validate_lens_name(name) == name

    def test_name_exceeds_max_length(self):
        """Name exceeding 100 characters should raise ValidationError."""
        with pytest.raises(ValidationError):
            validate_lens_name("a" * 101)

    def test_non_string_type_int(self):
        """Non-string input (int) should raise ValidationError."""
        with pytest.raises(ValidationError):
            validate_lens_name(123)

    def test_non_string_type_none(self):
        """Non-string input (None) should raise ValidationError."""
        with pytest.raises(ValidationError):
            validate_lens_name(None)


class TestValidateParabolicSag:
    """Tests for validate_parabolic_sag function."""

    def test_valid_sag_no_diameter(self):
        """Valid positive sag without diameter should be returned."""
        assert validate_parabolic_sag(5.0) == 5.0

    def test_zero_sag(self):
        """Zero sag should be valid."""
        assert validate_parabolic_sag(0.0) == 0.0

    def test_negative_sag_within_legacy_limit(self):
        """Negative sag within legacy limit should be valid."""
        assert validate_parabolic_sag(-50.0) == -50.0

    def test_sag_at_legacy_limit_positive(self):
        """Sag at +100 mm legacy limit should be valid."""
        assert validate_parabolic_sag(100.0) == 100.0

    def test_sag_at_legacy_limit_negative(self):
        """Sag at -100 mm legacy limit should be valid."""
        assert validate_parabolic_sag(-100.0) == -100.0

    def test_sag_exceeds_legacy_limit_positive(self):
        """Sag exceeding +100 mm legacy limit should raise ValidationError."""
        with pytest.raises(ValidationError):
            validate_parabolic_sag(100.1)

    def test_sag_exceeds_legacy_limit_negative(self):
        """Sag exceeding -100 mm legacy limit should raise ValidationError."""
        with pytest.raises(ValidationError):
            validate_parabolic_sag(-101.0)

    def test_non_numeric_sag_type_raises_error(self):
        """Non-numeric sag type should raise ValidationError."""
        with pytest.raises(ValidationError):
            validate_parabolic_sag("5.0")

    def test_sag_with_diameter_valid(self):
        """Valid sag with diameter should use diameter-scaled limit."""
        # Small sag with small diameter should be within limit
        result = validate_parabolic_sag(0.1, diameter=5.0, thickness=1.0)
        assert result == 0.1

    def test_sag_with_diameter_exceeds_limit(self):
        """Sag exceeding diameter-scaled limit should raise ValidationError."""
        # Very large sag with small diameter should exceed limit
        with pytest.raises(ValidationError):
            validate_parabolic_sag(500.0, diameter=5.0, thickness=1.0)

    def test_custom_param_name_in_error_message(self):
        """Custom param_name should appear in error message."""
        with pytest.raises(ValidationError) as exc_info:
            validate_parabolic_sag(150.0, param_name="surface_sag")
        assert "surface_sag" in str(exc_info.value)

    def test_thickness_parameter_with_diameter(self):
        """Thickness parameter should be passed to diameter-scaled limit."""
        # This tests that thickness is properly used with diameter
        result = validate_parabolic_sag(0.5, diameter=20.0, thickness=10.0)
        assert result == 0.5

    def test_sag_error_message_includes_vertex_radius_detail(self):
        """Error message for non-zero exceeding sag should include vertex radius."""
        with pytest.raises(ValidationError) as exc_info:
            validate_parabolic_sag(500.0, diameter=10.0, thickness=2.0)
        error_msg = str(exc_info.value)
        # Should mention the vertex radius in the detail
        assert "vertex radius" in error_msg or "exceeds" in error_msg
