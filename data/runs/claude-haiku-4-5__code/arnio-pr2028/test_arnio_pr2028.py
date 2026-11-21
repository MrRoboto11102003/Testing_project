import pytest
import arnio
from arnio.schema import String, Field


class TestStringSchema:
    """Tests for arnio.schema.String function."""

    def test_basic_string_field(self):
        """Test basic string field creation with defaults."""
        field = String()
        assert isinstance(field, Field)
        assert field.dtype == "string"
        assert field.nullable is True

    def test_nullable_parameter(self):
        """Test nullable parameter."""
        field_nullable = String(nullable=True)
        field_not_nullable = String(nullable=False)
        assert field_nullable.nullable is True
        assert field_not_nullable.nullable is False

    def test_pattern_valid(self):
        """Test valid regex pattern."""
        pattern = r"^\d{3}-\d{4}$"
        field = String(pattern=pattern)
        assert field.pattern == pattern

    def test_pattern_invalid_regex(self):
        """Test that invalid regex pattern raises ValueError."""
        with pytest.raises(ValueError, match="Invalid regex pattern"):
            String(pattern=r"[invalid(")

    def test_pattern_non_string_type_error(self):
        """Test that non-string pattern raises TypeError."""
        with pytest.raises(TypeError, match="pattern must be a string"):
            String(pattern=123)

    def test_allowed_sequence_types(self):
        """Test allowed parameter with list, set, and tuple."""
        field_list = String(allowed=["a", "b", "c"])
        field_set = String(allowed={"x", "y"})
        field_tuple = String(allowed=("p", "q"))
        
        assert field_list.allowed == {"a", "b", "c"}
        assert field_set.allowed == {"x", "y"}
        assert field_tuple.allowed == {"p", "q"}

    def test_allowed_bare_string_type_error(self):
        """Test that bare string for allowed raises TypeError."""
        with pytest.raises(TypeError, match="allowed must be a sequence"):
            String(allowed="invalid")

    def test_case_sensitive_and_unique(self):
        """Test case_sensitive and unique parameters."""
        field = String(case_sensitive=True, unique=True)
        assert field.case_sensitive is True
        assert field.unique is True

    def test_length_constraints_valid(self):
        """Test min_length and max_length parameters."""
        field = String(min_length=2, max_length=10)
        assert field.min_length == 2
        assert field.max_length == 10

    def test_min_length_greater_than_max_value_error(self):
        """Test that min_length > max_length raises ValueError."""
        with pytest.raises(ValueError, match="min_length must be less than or equal to max_length"):
            String(min_length=10, max_length=5)

    def test_length_parameter_type_errors(self):
        """Test that non-integer and boolean length parameters raise TypeError."""
        with pytest.raises(TypeError, match="min_length must be an integer"):
            String(min_length="not_int")
        with pytest.raises(TypeError, match="max_length must be an integer"):
            String(max_length=3.14)
        with pytest.raises(TypeError, match="min_length must be an integer"):
            String(min_length=True)

    def test_length_parameter_negative_value_error(self):
        """Test that negative length parameters raise ValueError."""
        with pytest.raises(ValueError, match="min_length must be greater than or equal to 0"):
            String(min_length=-1)
        with pytest.raises(ValueError, match="max_length must be greater than or equal to 0"):
            String(max_length=-2)

    def test_required_if_parameter(self):
        """Test required_if conditional requirement parameter."""
        condition = ("status", "active")
        field = String(required_if=condition)
        assert field.required_if == condition

    def test_severity_parameter(self):
        """Test severity parameter."""
        field_error = String(severity="error")
        field_warning = String(severity="warning")
        assert field_error.severity == "error"
        assert field_warning.severity == "warning"

    def test_all_parameters_combined(self):
        """Test creating a field with all parameters specified."""
        field = String(
            nullable=False,
            pattern=r"^[A-Z][a-z]+$",
            allowed=["Alice", "Bob", "Charlie"],
            case_sensitive=False,
            unique=True,
            min_length=3,
            max_length=20,
            severity="warning",
            required_if=("verified", True)
        )
        assert field.dtype == "string"
        assert field.nullable is False
        assert field.pattern == r"^[A-Z][a-z]+$"
        assert field.allowed == {"Alice", "Bob", "Charlie"}
        assert field.case_sensitive is False
        assert field.unique is True
        assert field.min_length == 3
        assert field.max_length == 20
        assert field.severity == "warning"
        assert field.required_if == ("verified", True)
