import pytest
from arnio.schema import String, Field


class TestStringSchemaField:
    """Test suite for arnio.schema.String function."""

    def test_string_default_parameters(self):
        """Test String with all default parameters."""
        field = String()
        assert isinstance(field, Field)
        assert field.dtype == "string"
        assert field.nullable is True
        assert field.pattern is None
        assert field.allowed is None
        assert field.case_sensitive is True
        assert field.unique is False
        assert field.severity == "error"
        assert field.min_length is None
        assert field.max_length is None

    def test_string_nullable_false(self):
        """Test String with nullable=False."""
        field = String(nullable=False)
        assert field.nullable is False

    def test_string_with_valid_regex_pattern(self):
        """Test String with valid regex pattern."""
        pattern = r"^[a-z]+$"
        field = String(pattern=pattern)
        assert field.pattern == pattern

    def test_string_with_invalid_regex_pattern_raises_error(self):
        """Test String with invalid regex pattern raises ValueError."""
        with pytest.raises(ValueError, match="Invalid regex pattern"):
            String(pattern="[invalid")

    def test_string_pattern_must_be_string(self):
        """Test String with non-string pattern raises TypeError."""
        with pytest.raises(TypeError, match="pattern must be a string"):
            String(pattern=123)

    def test_string_allowed_list_converted_to_set(self):
        """Test String with allowed list is converted to set."""
        allowed_list = ["a", "b", "c"]
        field = String(allowed=allowed_list)
        assert field.allowed == {"a", "b", "c"}

    def test_string_allowed_tuple_converted_to_set(self):
        """Test String with allowed tuple is converted to set."""
        field = String(allowed=("x", "y"))
        assert field.allowed == {"x", "y"}

    def test_string_allowed_bare_string_raises_error(self):
        """Test String with bare string in allowed raises TypeError."""
        with pytest.raises(TypeError, match="allowed must be a sequence"):
            String(allowed="abc")

    def test_string_allowed_bare_bytes_raises_error(self):
        """Test String with bare bytes in allowed raises TypeError."""
        with pytest.raises(TypeError, match="allowed must be a sequence"):
            String(allowed=b"data")

    def test_string_min_max_length_valid(self):
        """Test String with valid min_length and max_length."""
        field = String(min_length=5, max_length=10)
        assert field.min_length == 5
        assert field.max_length == 10

    def test_string_min_length_greater_than_max_raises_error(self):
        """Test String with min_length > max_length raises ValueError."""
        with pytest.raises(ValueError, match="min_length must be less than or equal to max_length"):
            String(min_length=10, max_length=5)

    def test_string_negative_min_length_raises_error(self):
        """Test String with negative min_length raises ValueError."""
        with pytest.raises(ValueError, match="min_length must be greater than or equal to 0"):
            String(min_length=-1)

    def test_string_negative_max_length_raises_error(self):
        """Test String with negative max_length raises ValueError."""
            String(max_length=-1)

    def test_string_min_length_non_integer_raises_error(self):
        """Test String with non-integer min_length raises TypeError."""
        with pytest.raises(TypeError, match="min_length must be an integer"):
            String(min_length="5")

    def test_string_max_length_non_integer_raises_error(self):
        """Test String with non-integer max_length raises TypeError."""
        with pytest.raises(TypeError, match="max_length must be an integer"):
            String(max_length=5.5)

    def test_string_min_length_bool_raises_error(self):
        """Test String with bool min_length raises TypeError (bool is int subclass but should reject)."""
        with pytest.raises(TypeError, match="min_length must be an integer"):
            String(min_length=True)

    def test_string_unique_and_case_sensitivity(self):
        """Test String with unique and case_sensitive flags."""
        field = String(unique=True, case_sensitive=False)
        assert field.unique is True
        assert field.case_sensitive is False

    def test_string_severity_levels(self):
        """Test String with different severity levels."""
        field_error = String(severity="error")
        field_warning = String(severity="warning")
        assert field_error.severity == "error"
        assert field_warning.severity == "warning"

    def test_string_required_if_parameter(self):
        """Test String with required_if conditional parameter."""
        required_if = ("status", "active")
        field = String(required_if=required_if)
        assert field.required_if == required_if

    def test_string_combined_constraints(self):
        """Test String with multiple constraints combined."""
        field = String(
            nullable=False,
            pattern=r"^[A-Z]",
            allowed={"Name", "Title"},
            unique=True,
            min_length=1,
            max_length=50,
            severity="warning"
        )
        assert field.dtype == "string"
        assert field.nullable is False
        assert field.pattern == r"^[A-Z]"
        assert field.allowed == {"Name", "Title"}
        assert field.unique is True
        assert field.min_length == 1
        assert field.max_length == 50
        assert field.severity == "warning"

    def test_string_min_length_equal_to_max_length(self):
        """Test String with min_length equal to max_length."""
        field = String(min_length=5, max_length=5)
        assert field.min_length == 5
        assert field.max_length == 5
