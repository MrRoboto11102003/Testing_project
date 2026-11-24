import pytest
import re
from openlens.validation import (
    validate_lens_name,
    validate_parabolic_sag,
    parabolic_sag_limit,
    ValidationError,
)


class TestValidateLensName:
    """Tests for validate_lens_name function."""

    def test_valid_string_returned_unchanged(self):
        """Valid string should be returned after stripping."""
        assert validate_lens_name("My Lens") == "My Lens"

    def test_type_error_int(self):
        """Non-string int should raise ValidationError with correct type repr."""
        with pytest.raises(ValidationError, match=r"Name must be a string, got <class 'int'>"):
            validate_lens_name(123)

    def test_type_error_none(self):
        """None should raise ValidationError with NoneType."""
        with pytest.raises(ValidationError, match=r"Name must be a string, got <class 'NoneType'>"):
            validate_lens_name(None)

    def test_type_error_bool(self):
        """Bool should raise ValidationError (not accepted even though bool is int subclass)."""
        with pytest.raises(ValidationError, match=r"Name must be a string, got <class 'bool'>"):
            validate_lens_name(True)

    def test_whitespace_stripping(self):
        """Leading/trailing whitespace should be stripped."""
        assert validate_lens_name("  Lens A  ") == "Lens A"
        assert validate_lens_name("\t\nLens\n\t") == "Lens"

    def test_inner_whitespace_preserved(self):
        """Inner whitespace should be preserved."""
        assert validate_lens_name("a b  c") == "a b  c"

    def test_empty_string_returns_untitled(self):
        """Empty or whitespace-only strings should return 'Untitled'."""
        assert validate_lens_name("") == "Untitled"
        assert validate_lens_name("   ") == "Untitled"
        assert validate_lens_name("\t\n") == "Untitled"

    def test_max_length_100_accepted(self):
        """Exactly 100 characters should be accepted."""
        name_100 = "a" * 100
        assert validate_lens_name(name_100) == name_100

    def test_length_101_rejected(self):
        """101 characters should be rejected."""
        with pytest.raises(ValidationError, match="Name must be 100 characters or less"):
            validate_lens_name("a" * 101)

    def test_length_checked_after_stripping(self):
        """Length validation should occur after stripping whitespace."""
        # 100 chars padded should pass
        assert validate_lens_name("  " + "a" * 100 + "  ") == "a" * 100
        # 101 chars padded should fail
        with pytest.raises(ValidationError, match="Name must be 100 characters or less"):
            validate_lens_name("  " + "a" * 101 + "  ")


class TestValidateParabolicSag:
    """Tests for validate_parabolic_sag function."""

    def test_valid_sag_returns_float(self):
        """Valid numbers should be converted to and returned as float."""
        assert validate_parabolic_sag(0.0) == 0.0
        assert validate_parabolic_sag(5.5) == 5.5
        assert validate_parabolic_sag(-3.5) == -3.5
        assert validate_parabolic_sag(5) == 5.0
        assert isinstance(validate_parabolic_sag(5), float)

    def test_type_error_string(self):
        """Non-numeric types should raise ValidationError."""
        with pytest.raises(ValidationError, match="parabolic sag must be a number, got str"):
            validate_parabolic_sag("1")

    def test_type_error_bool(self):
        """Bool should raise ValidationError with specific bool message."""
        with pytest.raises(ValidationError, match="parabolic sag must be a number, got bool"):
            validate_parabolic_sag(True)

    def test_type_error_none(self):
        """None should raise ValidationError."""
        with pytest.raises(ValidationError, match="parabolic sag must be a number, got NoneType"):
            validate_parabolic_sag(None)

    def test_nan_rejected(self):
        """NaN should raise ValidationError for finite number constraint."""
        with pytest.raises(ValidationError, match="parabolic sag must be a finite number, got nan"):
            validate_parabolic_sag(float("nan"))

    def test_inf_rejected(self):
        """Infinity should raise ValidationError."""
        with pytest.raises(ValidationError, match="parabolic sag must be a finite number, got inf"):
            validate_parabolic_sag(float("inf"))

    def test_legacy_branch_magnitude_100_accepted(self):
        """Without diameter, magnitude 100 should be accepted (legacy behavior)."""
        assert validate_parabolic_sag(100.0) == 100.0
        assert validate_parabolic_sag(-100.0) == -100.0

    def test_legacy_branch_magnitude_exceeds_100_rejected(self):
        """Without diameter, magnitude > 100 should be rejected."""
        with pytest.raises(ValidationError, match="magnitude must be at most 100 mm"):
            validate_parabolic_sag(100.5)
        with pytest.raises(ValidationError, match="magnitude must be at most 100 mm"):
            validate_parabolic_sag(-100.5)

    def test_diameter_branch_validates_bound(self):
        """With diameter, sag should be validated against parabolic_sag_limit."""
        limit = parabolic_sag_limit(10.0, 0.0)
        # At limit should pass
        assert validate_parabolic_sag(limit, diameter=10.0) == pytest.approx(limit)
        # Exceeding limit should fail
        with pytest.raises(ValidationError, match="exceeds the .* mm limit"):
            validate_parabolic_sag(limit * 1.01, diameter=10.0, thickness=0.0)

    def test_diameter_branch_validates_diameter_positive(self):
        """With diameter, diameter must be positive."""
        with pytest.raises(ValidationError, match="diameter must be positive, got 0.0"):
            validate_parabolic_sag(1.0, diameter=0.0)
        with pytest.raises(ValidationError, match="diameter must be positive, got -5.0"):
            validate_parabolic_sag(1.0, diameter=-5.0)

    def test_diameter_branch_validates_thickness_non_negative(self):
        """With diameter, thickness cannot be negative."""
        with pytest.raises(ValidationError, match="thickness cannot be negative, got -1.0"):
            validate_parabolic_sag(1.0, diameter=10.0, thickness=-1.0)

    def test_error_message_includes_vertex_radius_for_nonzero_sag(self):
        """Error message should include implied vertex radius calculation."""
        limit = parabolic_sag_limit(20.0, 0.0)
        sag = limit * 1.1
        with pytest.raises(ValidationError) as exc_info:
            validate_parabolic_sag(sag, diameter=20.0, thickness=0.0)
        error_msg = str(exc_info.value)
        assert "implies vertex radius" in error_msg
        assert "D=20.0mm" in error_msg

    def test_custom_param_name_in_error_message(self):
        """Custom param_name should appear in error messages."""
        with pytest.raises(ValidationError, match="my_sag must be a number"):
            validate_parabolic_sag("invalid", param_name="my_sag")
        with pytest.raises(ValidationError, match="my_sag magnitude must be at most 100 mm"):
            validate_parabolic_sag(101.0, param_name="my_sag")

    def test_check_order_sag_validated_before_diameter(self):
        """Sag validation should happen before diameter validation."""
        # If sag is invalid (NaN), that error should be raised even if diameter is also invalid
        with pytest.raises(ValidationError, match="parabolic sag must be a finite number"):
            validate_parabolic_sag(float("nan"), diameter=-1.0, thickness="invalid")
