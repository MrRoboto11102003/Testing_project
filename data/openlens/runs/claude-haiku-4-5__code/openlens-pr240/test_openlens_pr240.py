import pytest
import numpy as np
from openlens.validation import validate_lens_name, validate_parabolic_sag, ValidationError


class TestValidateLensName:
    """Tests for validate_lens_name function."""

    def test_valid_name_returned_as_is(self):
        """Valid lens name is returned unchanged (stripped)."""
        assert validate_lens_name("My Lens") == "My Lens"

    def test_empty_string_returns_untitled(self):
        """Empty string returns default 'Untitled'."""
        assert validate_lens_name("") == "Untitled"

    def test_whitespace_only_returns_untitled(self):
        """Whitespace-only strings return default 'Untitled'."""
        assert validate_lens_name("   ") == "Untitled"
        assert validate_lens_name("\t\n") == "Untitled"

    def test_whitespace_stripped_from_valid_name(self):
        """Leading and trailing whitespace is stripped."""
        assert validate_lens_name("  Test  ") == "Test"
        assert validate_lens_name("\tLens\n") == "Lens"

    def test_exactly_100_chars_allowed(self):
        """Exactly 100 characters is the maximum allowed length."""
        name = "x" * 100
        assert validate_lens_name(name) == name

    def test_over_100_chars_raises_error(self):
        """More than 100 characters raises ValidationError."""
        with pytest.raises(ValidationError, match="100 characters or less"):
            validate_lens_name("x" * 101)

    def test_non_string_raises_error(self):
        """Non-string types raise ValidationError."""
        with pytest.raises(ValidationError, match="must be a string"):
            validate_lens_name(123)
        with pytest.raises(ValidationError, match="must be a string"):
            validate_lens_name(None)


class TestValidateParabolicSag:
    """Tests for validate_parabolic_sag function."""

    def test_valid_sag_in_legacy_mode(self):
        """Positive, negative, and zero sag accepted in legacy mode."""
        assert validate_parabolic_sag(5.0) == 5.0
        assert validate_parabolic_sag(-5.0) == -5.0
        assert validate_parabolic_sag(0.0) == 0.0

    def test_legacy_100mm_absolute_cap(self):
        """Magnitude of 100 mm is the absolute limit in legacy mode."""
        assert validate_parabolic_sag(100.0) == 100.0
        assert validate_parabolic_sag(-100.0) == -100.0
        with pytest.raises(ValidationError, match="must be at most 100 mm"):
            validate_parabolic_sag(100.1)
        with pytest.raises(ValidationError, match="must be at most 100 mm"):
            validate_parabolic_sag(-100.1)

    def test_diameter_enables_scaled_limit(self):
        """Diameter parameter uses parabolic_sag_limit instead of absolute cap."""
        with pytest.raises(ValidationError, match="exceeds the"):
            validate_parabolic_sag(5000.0, diameter=50.0)
        assert validate_parabolic_sag(2.0, diameter=50.0, thickness=5.0) == 2.0

    def test_custom_param_name_and_numeric_types(self):
        """Custom param_name appears in errors; numpy scalars and integers accepted."""
        with pytest.raises(ValidationError, match="custom_param"):
            validate_parabolic_sag(150.0, param_name="custom_param")
        assert validate_parabolic_sag(np.float64(5.0)) == 5.0
        assert validate_parabolic_sag(5) == 5
