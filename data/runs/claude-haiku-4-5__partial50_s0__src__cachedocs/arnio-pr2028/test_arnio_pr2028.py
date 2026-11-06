import pytest
import arnio as ar
from arnio.schema import String, Field


class TestStringSchemaField:
    """Tests for arnio.schema.String field factory."""

    def test_string_creates_field_with_defaults(self):
        """String() returns a Field with correct default values."""
        field = String()
        assert isinstance(field, Field)
        assert field.dtype == "string"
        assert field.nullable is True
        assert field.pattern is None
        assert field.allowed is None
        assert field.case_sensitive is True
        assert field.unique is False
        assert field.severity == "error"
        assert field.required_if is None

    def test_nullable_parameter(self):
        """nullable parameter controls whether nulls are allowed."""
        assert String(nullable=True).nullable is True
        assert String(nullable=False).nullable is False

    def test_pattern_with_valid_regex(self):
        """pattern parameter accepts valid regex and stores it."""
        pattern = r"^[A-Z][a-z]+$"
        field = String(pattern=pattern)
        assert field.pattern == pattern

    def test_pattern_with_invalid_regex_raises_valueerror(self):
        """Invalid regex pattern raises ValueError."""
        with pytest.raises(ValueError, match="Invalid regex pattern"):
            String(pattern="[unclosed")

    def test_pattern_must_be_string_not_other_type(self):
        """pattern must be string type, not int or other."""
        with pytest.raises(TypeError, match="pattern must be a string or None"):
            String(pattern=123)

    def test_allowed_accepts_list_tuple_set(self):
        """allowed parameter accepts list, tuple, or set and converts to set."""
        assert String(allowed=["a", "b"]).allowed == {"a", "b"}
        assert String(allowed=("x", "y")).allowed == {"x", "y"}
        assert String(allowed={"red", "blue"}).allowed == {"red", "blue"}

    def test_allowed_bare_string_raises_typeerror(self):
        """bare string as allowed raises TypeError (must be sequence)."""
        with pytest.raises(TypeError, match="allowed must be a sequence"):
            String(allowed="notvalid")

    def test_allowed_bytes_raises_typeerror(self):
        """bare bytes as allowed raises TypeError."""
        with pytest.raises(TypeError, match="allowed must be a sequence"):
            String(allowed=b"bytes")

    def test_case_sensitive_parameter(self):
        """case_sensitive parameter sets field behavior."""
        assert String(case_sensitive=True).case_sensitive is True
        assert String(case_sensitive=False).case_sensitive is False

    def test_unique_parameter(self):
        """unique parameter indicates values must be unique."""
        assert String(unique=False).unique is False
        assert String(unique=True).unique is True

    def test_min_and_max_length_constraints(self):
        """min_length and max_length parameters set length bounds."""
        field = String(min_length=2, max_length=100)
        assert field.min_length == 2
        assert field.max_length == 100

    def test_min_length_exceeding_max_length_raises_error(self):
        """min_length > max_length raises ValueError."""
        with pytest.raises(ValueError, match="min_length must be less than or equal to max_length"):
            String(min_length=10, max_length=5)

    def test_negative_length_values_raise_error(self):
        """Negative min_length or max_length raises ValueError."""
        with pytest.raises(ValueError, match="greater than or equal to 0"):
            String(min_length=-1)
        with pytest.raises(ValueError, match="greater than or equal to 0"):
            String(max_length=-5)

    def test_non_integer_length_raises_error(self):
        """Non-integer min_length or max_length raises TypeError."""
        with pytest.raises(TypeError, match="must be an integer"):
            String(min_length="10")
        with pytest.raises(TypeError, match="must be an integer"):
            String(max_length=3.5)

    def test_severity_and_required_if_parameters(self):
        """severity and required_if parameters are stored correctly."""
        field = String(
            severity="warning",
            required_if=("status", "active"),
        )
        assert field.severity == "warning"
        assert field.required_if == ("status", "active")
