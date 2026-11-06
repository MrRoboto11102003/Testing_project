import pytest
import arnio as ar


class TestStringField:
    """Test arnio.schema.String field creation and validation."""

    def test_string_basic_creation(self):
        """Test basic String field creation with defaults."""
        field = ar.String()
        assert field.dtype == "string"
        assert field.nullable is True
        assert field.pattern is None
        assert field.allowed is None
        assert field.unique is False
        assert field.severity == "error"
        assert field.min_length is None
        assert field.max_length is None

    def test_string_nullable_false(self):
        """Test String field with nullable=False."""
        field = ar.String(nullable=False)
        assert field.nullable is False

    def test_string_with_valid_pattern(self):
        """Test String field with valid regex pattern."""
        pattern = r"^[A-Z][a-z]+$"
        field = ar.String(pattern=pattern)
        assert field.pattern == pattern

    def test_string_with_invalid_regex_pattern(self):
        """Test that invalid regex pattern raises ValueError."""
        with pytest.raises(ValueError, match="Invalid regex pattern"):
            ar.String(pattern="[unclosed")

    def test_string_pattern_type_validation(self):
        """Test that non-string pattern raises TypeError."""
        with pytest.raises(TypeError, match="pattern must be a string"):
            ar.String(pattern=123)

    def test_string_allowed_list_conversion(self):
        """Test String field with list of allowed values converts to set."""
        allowed = ["active", "inactive", "pending"]
        field = ar.String(allowed=allowed)
        assert field.allowed == set(allowed)

    def test_string_allowed_bare_string_rejected(self):
        """Test that bare string as allowed raises TypeError."""
        with pytest.raises(TypeError, match="allowed must be a sequence"):
            ar.String(allowed="invalid_string")

    def test_string_allowed_bare_bytes_rejected(self):
        """Test that bare bytes as allowed raises TypeError."""
        with pytest.raises(TypeError, match="allowed must be a sequence"):
            ar.String(allowed=b"invalid")

    def test_string_length_constraints(self):
        """Test String field with min_length and max_length."""
        field = ar.String(min_length=3, max_length=50)
        assert field.min_length == 3
        assert field.max_length == 50

    def test_string_min_length_greater_than_max_raises_error(self):
        """Test that min_length > max_length raises ValueError."""
        with pytest.raises(
            ValueError, match="min_length must be less than or equal to max_length"
        ):
            ar.String(min_length=50, max_length=3)

    def test_string_negative_length_raises_error(self):
        """Test that negative length values raise ValueError."""
        with pytest.raises(ValueError, match="must be greater than or equal to 0"):
            ar.String(min_length=-1)

    def test_string_non_integer_length_raises_error(self):
        """Test that non-integer length values raise TypeError."""
        with pytest.raises(TypeError, match="must be an integer"):
            ar.String(max_length="50")

    def test_string_case_sensitivity(self):
        """Test String field with case_sensitive parameter."""
        field = ar.String(case_sensitive=False)
        assert field.case_sensitive is False

    def test_string_unique_and_severity(self):
        """Test String field with unique and severity parameters."""
        field = ar.String(unique=True, severity="warning")
        assert field.unique is True
        assert field.severity == "warning"

    def test_string_required_if_condition(self):
        """Test String field with conditional requirement."""
        field = ar.String(required_if=("status", "active"))
        assert field.required_if == ("status", "active")

    def test_string_all_parameters_combined(self):
        """Test String field with all parameters specified."""
        field = ar.String(
            nullable=False,
            pattern=r"^[a-z]+$",
            allowed=["foo", "bar", "baz"],
            case_sensitive=False,
            unique=True,
            severity="warning",
            min_length=2,
            max_length=100,
            required_if=("role", "admin"),
        )
        assert field.nullable is False
        assert field.pattern == r"^[a-z]+$"
        assert field.allowed == {"foo", "bar", "baz"}
        assert field.case_sensitive is False
        assert field.unique is True
        assert field.severity == "warning"
        assert field.min_length == 2
        assert field.max_length == 100
        assert field.required_if == ("role", "admin")
