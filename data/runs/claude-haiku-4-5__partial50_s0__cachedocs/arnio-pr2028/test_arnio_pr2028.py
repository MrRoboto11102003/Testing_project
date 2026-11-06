import pytest
import pandas as pd
import arnio as ar


class TestStringField:
    """Test the arnio.schema.String field validator."""

    def test_string_field_creation(self):
        """String field should be created successfully."""
        field = ar.String()
        assert field is not None

    def test_string_in_schema(self):
        """String field should work in a schema."""
        schema = ar.Schema({"name": ar.String()})
        assert schema is not None

    def test_string_nullable_default_allows_nulls(self):
        """By default (nullable=True), String allows null values."""
        frame = ar.from_dict({
            "name": ["Alice", None, "Bob"]
        })
        schema = ar.Schema({"name": ar.String()})
        result = ar.validate(frame, schema)
        assert result.passed

    def test_string_nullable_false_rejects_nulls(self):
        """String with nullable=False should reject null values."""
        frame = ar.from_dict({
            "name": ["Alice", None, "Bob"]
        })
        schema = ar.Schema({"name": ar.String(nullable=False)})
        result = ar.validate(frame, schema)
        assert not result.passed
        assert len(result.issues) > 0

    def test_string_min_length_constraint(self):
        """String with min_length should reject shorter strings."""
        frame = ar.from_dict({
            "username": ["alice", "ab", "bob"]
        })
        schema = ar.Schema({"username": ar.String(min_length=3)})
        result = ar.validate(frame, schema)
        assert not result.passed

    def test_string_max_length_constraint(self):
        """String with max_length should reject longer strings."""
        frame = ar.from_dict({
            "username": ["alice", "bob", "verylongusername"]
        })
        schema = ar.Schema({"username": ar.String(max_length=10)})
        result = ar.validate(frame, schema)
        assert not result.passed

    def test_string_allowed_values_list(self):
        """String with allowed list should restrict to specified values."""
        frame = ar.from_dict({
            "status": ["active", "inactive", "unknown"]
        })
        schema = ar.Schema({
            "status": ar.String(allowed=["active", "inactive"])
        })
        result = ar.validate(frame, schema)
        assert not result.passed

    def test_string_allowed_values_set(self):
        """String should accept allowed as a set."""
        frame = ar.from_dict({
            "status": ["active", "inactive"]
        })
        schema = ar.Schema({
            "status": ar.String(allowed={"active", "inactive"})
        })
        result = ar.validate(frame, schema)
        assert result.passed

    def test_string_allowed_values_tuple(self):
        """String should accept allowed as a tuple."""
        frame = ar.from_dict({
            "status": ["active", "inactive"]
        })
        schema = ar.Schema({
            "status": ar.String(allowed=("active", "inactive"))
        })
        result = ar.validate(frame, schema)
        assert result.passed

    def test_string_case_sensitive_default(self):
        """By default (case_sensitive=True), String is case-sensitive."""
        frame = ar.from_dict({
            "status": ["Active", "inactive"]
        })
        schema = ar.Schema({
            "status": ar.String(
                allowed=["active", "inactive"],
                case_sensitive=True
            )
        })
        result = ar.validate(frame, schema)
        assert not result.passed

    def test_string_case_insensitive(self):
        """String with case_sensitive=False should ignore case."""
        frame = ar.from_dict({
            "status": ["Active", "INACTIVE"]
        })
        schema = ar.Schema({
            "status": ar.String(
                allowed=["active", "inactive"],
                case_sensitive=False
            )
        })
        result = ar.validate(frame, schema)
        assert result.passed

    def test_string_unique_constraint(self):
        """String with unique=True should reject duplicate values."""
        frame = ar.from_dict({
            "email": ["alice@example.com", "bob@example.com", "alice@example.com"]
        })
        schema = ar.Schema({"email": ar.String(unique=True)})
        result = ar.validate(frame, schema)
        assert not result.passed

    def test_string_severity_warning(self):
        """String with severity='warning' should pass but report issues."""
        frame = ar.from_dict({
            "name": ["Alice", "ab"]
        })
        schema = ar.Schema({
            "name": ar.String(min_length=3, severity="warning")
        })
        result = ar.validate(frame, schema)
        assert result.passed  # Warnings don't fail validation
        assert result.issue_count >= 1  # But issues are still reported

    def test_string_required_if_conditional(self):
        """String with required_if should require field when condition matches."""
        frame = ar.from_dict({
            "user_type": ["international", "domestic"],
            "country": [None, None]
        })
        schema = ar.Schema({
            "user_type": ar.String(),
            "country": ar.String(
                nullable=True,
                required_if=("user_type", "international")
            )
        })
        result = ar.validate(frame, schema)
        assert not result.passed

    def test_string_multiple_constraints_combined(self):
        """String should support combining multiple constraints."""
        frame = ar.from_dict({
            "username": ["alice", "ab", "Bob", "alice"]
        })
        schema = ar.Schema({
            "username": ar.String(
                min_length=3,
                max_length=20,
                case_sensitive=True,
                unique=True,
                nullable=False
            )
        })
        result = ar.validate(frame, schema)
        assert not result.passed  # Multiple violations expected
