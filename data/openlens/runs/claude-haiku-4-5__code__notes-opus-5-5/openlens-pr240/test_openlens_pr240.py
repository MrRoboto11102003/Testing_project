import pytest
import re
import numpy as np
from openlens.validation import (
    validate_lens_name,
    validate_parabolic_sag,
    parabolic_sag_limit,
    ValidationError,
)


class TestValidateLensName:
    """Tests for validate_lens_name function."""

    def test_valid_name_unchanged(self):
        """Valid name should be returned unchanged."""
        assert validate_lens_name("Lens A") == "Lens A"

    def test_empty_string_returns_untitled(self):
        """Empty string should return 'Untitled'."""
        assert validate_lens_name("") == "Untitled"

    def test_whitespace_stripped(self):
        """Leading and trailing whitespace should be stripped."""
        assert validate_lens_name("  Lens A  ") == "Lens A"
        assert validate_lens_name("   ") == "Untitled"

    def test_inner_whitespace_preserved(self):
        """Inner whitespace should be preserved."""
        assert validate_lens_name("a b  c") == "a b  c"

    def test_exactly_100_chars_allowed(self):
        """Name with exactly 100 characters should be allowed."""
        name_100 = "a" * 100
        assert validate_lens_name(name_100) == name_100

    def test_101_chars_raises_error(self):
        """Name with 101 characters should raise ValidationError."""
        name_101 = "a" * 101
        with pytest.raises(
            ValidationError,
            match=re.escape("Name must be 100 characters or less"),
        ):
            validate_lens_name(name_101)

    def test_non_string_int_raises(self):
        """Non-string int should raise ValidationError with correct message."""
        with pytest.raises(ValidationError) as excinfo:
            validate_lens_name(123)
        assert str(excinfo.value) == "Name must be a string, got <class 'int'>"

    def test_non_string_none_raises(self):
        """None should raise ValidationError."""
        with pytest.raises(ValidationError) as excinfo:
            validate_lens_name(None)
        assert str(excinfo.value) == "Name must be a string, got <class 'NoneType'>"

    def test_non_string_bool_raises(self):
        """Bool should raise ValidationError."""
        with pytest.raises(ValidationError) as excinfo:
            validate_lens_name(True)
        assert str(excinfo.value) == "Name must be a string, got <class 'bool'>"


class TestValidateParabolicSag:
    """Tests for validate_parabolic_sag function."""

    def test_zero_sag_returns_zero(self):
        """Zero sag should return 0.0."""
        assert validate_parabolic_sag(0) == 0.0

    def test_legacy_limit_100_passes(self):
        """Sag of exactly 100mm magnitude should pass legacy check."""
        assert validate_parabolic_sag(100.0) == 100.0
        assert validate_parabolic_sag(-100.0) == -100.0

    def test_legacy_limit_exceeds_raises(self):
        """Sag exceeding 100mm magnitude should raise in legacy mode."""
        with pytest.raises(
            ValidationError,
            match=re.escape("parabolic sag magnitude must be at most 100 mm"),
        ):
            validate_parabolic_sag(100.5)

    def test_sag_type_string_raises(self):
        """String sag should raise ValidationError."""
        with pytest.raises(
            ValidationError,
            match=re.escape("parabolic sag must be a number, got str"),
        ):
            validate_parabolic_sag("1")

    def test_sag_nan_raises(self):
        """NaN sag should raise ValidationError."""
        with pytest.raises(
            ValidationError,
            match=re.escape("parabolic sag must be a finite number, got nan"),
        ):
            validate_parabolic_sag(float("nan"))

    def test_sag_inf_raises(self):
        """Inf sag should raise ValidationError."""
        with pytest.raises(
            ValidationError,
            match=re.escape("parabolic sag must be a finite number, got inf"),
        ):
            validate_parabolic_sag(float("inf"))

    def test_numpy_float32_accepted(self):
        """numpy.float32 sag should be accepted and converted to float."""
        result = validate_parabolic_sag(np.float32(1.5))
        assert result == pytest.approx(1.5)
        assert isinstance(result, float)

    def test_diameter_bound_at_limit(self):
        """Sag at the diameter-based limit should pass."""
        diameter = 10.0
        limit = parabolic_sag_limit(diameter)
        result = validate_parabolic_sag(limit, diameter=diameter)
        assert result == pytest.approx(limit)

    def test_diameter_bound_exceeds_raises(self):
        """Sag exceeding diameter-based limit should raise."""
        diameter = 10.0
        limit = parabolic_sag_limit(diameter)
        with pytest.raises(ValidationError):
            validate_parabolic_sag(limit * 1.001, diameter=diameter)

    def test_diameter_zero_raises(self):
        """Diameter of 0 should raise ValidationError."""
        with pytest.raises(
            ValidationError,
            match=re.escape("diameter must be positive, got 0.0"),
        ):
            validate_parabolic_sag(1, diameter=0)

    def test_thickness_negative_raises(self):
        """Negative thickness should raise ValidationError."""
        with pytest.raises(
            ValidationError,
            match=re.escape("thickness cannot be negative, got -1.0"),
        ):
            validate_parabolic_sag(1, diameter=10, thickness=-1)

    def test_check_order_sag_before_diameter(self):
        """Sag validation error should be raised before diameter validation."""
        with pytest.raises(
            ValidationError,
            match=re.escape("parabolic sag must be a finite number, got nan"),
        ):
            validate_parabolic_sag(float("nan"), diameter=-1)

    def test_thickness_ignored_without_diameter(self):
        """Thickness should be ignored when diameter is None."""
        result = validate_parabolic_sag(50, diameter=None, thickness=-5)
        assert result == 50.0

    def test_custom_param_name_in_error(self):
        """Custom param_name should appear in error messages."""
        with pytest.raises(ValidationError, match="custom_sag must be a number"):
            validate_parabolic_sag(None, param_name="custom_sag")
