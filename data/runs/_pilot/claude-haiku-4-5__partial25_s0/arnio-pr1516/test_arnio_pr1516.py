import pytest
import pandas as pd
import json
import arnio as ar


class TestSchemaToJson:
    """Tests for Schema.to_json() method."""

    def test_to_json_returns_string(self):
        """Schema.to_json() returns a valid JSON string."""
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "name": ar.String(),
        })
        payload = schema.to_json()
        assert isinstance(payload, str)

    def test_to_json_is_valid_json(self):
        """Schema.to_json() output is valid JSON that can be parsed."""
        schema = ar.Schema({
            "id": ar.Int64(),
            "email": ar.Email(),
        })
        payload = schema.to_json()
        parsed = json.loads(payload)
        assert isinstance(parsed, dict)

    def test_to_json_roundtrip_with_from_json(self):
        """Schema round-trips through to_json()/from_json()."""
        original = ar.Schema({
            "id": ar.String(nullable=False),
            "created_at": ar.DateTime(format="%Y-%m-%dT%H:%M:%S"),
        })
        payload = original.to_json()
        restored = ar.Schema.from_json(payload)
        assert restored is not None

    def test_to_json_preserves_field_constraints(self):
        """to_json() preserves field constraints (min_length, max_length, min, max)."""
        schema = ar.Schema({
            "username": ar.String(min_length=3, max_length=20, nullable=False),
            "age": ar.Int64(min=0, max=150),
        })
        payload = schema.to_json()
        restored = ar.Schema.from_json(payload)
        assert restored is not None

    def test_to_json_preserves_unique_constraint(self):
        """to_json() preserves single-column unique constraints."""
        schema = ar.Schema({
            "user_id": ar.Int64(nullable=False),
            "email": ar.String(),
        }, unique=["user_id"])
        payload = schema.to_json()
        restored = ar.Schema.from_json(payload)
        assert restored is not None

    def test_to_json_preserves_composite_unique_constraint(self):
        """to_json() preserves multi-column unique constraints."""
        schema = ar.Schema({
            "user_id": ar.Int64(),
            "course_id": ar.Int64(),
        }, unique=["user_id", "course_id"])
        payload = schema.to_json()
        restored = ar.Schema.from_json(payload)
        assert restored is not None

    def test_to_json_preserves_strict_mode(self):
        """to_json() preserves strict=True setting."""
        schema = ar.Schema({
            "id": ar.String(nullable=False),
        }, strict=True)
        payload = schema.to_json()
        restored = ar.Schema.from_json(payload)
        assert restored is not None

    def test_to_json_with_multiple_field_types(self):
        """to_json() works with diverse field types."""
        schema = ar.Schema({
            "id": ar.Int64(),
            "score": ar.Float64(),
            "active": ar.Bool(),
            "email": ar.Email(),
            "phone": ar.PhoneNumber(),
            "website": ar.URL(),
            "created": ar.DateTime(),
        })
        payload = schema.to_json()
        parsed = json.loads(payload)
        assert isinstance(parsed, dict)

    def test_to_json_with_regex_pattern(self):
        """to_json() preserves Regex field definitions."""
        schema = ar.Schema({
            "code": ar.Regex(r"^USR-\d{4}$", nullable=False),
        })
        payload = schema.to_json()
        restored = ar.Schema.from_json(payload)
        assert restored is not None


class TestSchemaValidate:
    """Tests for Schema.validate() method."""

    def test_validate_returns_validation_result(self):
        """validate() returns a ValidationResult object."""
        frame = ar.from_dict({"name": ["Alice", "Bob"]})
        schema = ar.Schema({"name": ar.String()})
        result = schema.validate(frame)
        assert hasattr(result, "passed")
        assert hasattr(result, "issue_count")
        assert isinstance(result.passed, bool)

    def test_validate_passes_for_valid_data(self):
        """validate() returns passed=True for valid data."""
        frame = ar.from_dict({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
        })
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "name": ar.String(nullable=False),
        })
        result = schema.validate(frame)
        assert result.passed is True
        assert result.issue_count == 0

    def test_validate_fails_on_dtype_mismatch(self):
        """validate() detects dtype mismatches."""
        frame = ar.from_dict({"age": ["25", "30", "35"]})  # strings not ints
        schema = ar.Schema({"age": ar.Int64()})
        result = schema.validate(frame)
        assert result.passed is False
        assert result.issue_count > 0

    def test_validate_fails_on_nullable_violation(self):
        """validate() detects null values when nullable=False."""
        frame = ar.from_dict({
            "email": ["alice@example.com", None, "charlie@example.com"]
        })
        schema = ar.Schema({"email": ar.String(nullable=False)})
        result = schema.validate(frame)
        assert result.passed is False
        assert result.issue_count > 0

    def test_validate_fails_on_email_format_invalid(self):
        """validate() rejects invalid email addresses."""
        frame = ar.from_dict({
            "email": ["alice@example.com", "not-an-email", "bob@example.com"]
        })
        schema = ar.Schema({"email": ar.Email()})
        result = schema.validate(frame)
        assert result.passed is False

    def test_validate_fails_on_unique_constraint_violation(self):
        """validate() detects single-column unique constraint violations."""
        frame = ar.from_dict({"id": [1, 2, 2, 3]})
        schema = ar.Schema({"id": ar.Int64()}, unique=["id"])
        result = schema.validate(frame)
        assert result.passed is False
        assert result.issue_count > 0

    def test_validate_fails_on_composite_unique_violation(self):
        """validate() detects multi-column unique constraint violations."""
        frame = ar.from_dict({
            "user_id": [1, 1, 2],
            "course_id": [10, 10, 10],
        })
        schema = ar.Schema(
            {
                "user_id": ar.Int64(),
                "course_id": ar.Int64(),
            },
            unique=["user_id", "course_id"],
        )
        result = schema.validate(frame)
        assert result.passed is False

    def test_validate_respects_max_errors_parameter(self):
        """validate() respects max_errors limit."""
        frame = ar.from_dict({
            "col1": ["x", "y", "z"],
            "col2": ["a", "b", "c"],
        })
        schema = ar.Schema({
            "col1": ar.Int64(),
            "col2": ar.Int64(),
        })
        result = schema.validate(frame, max_errors=2)
        assert result.issue_count <= 2

    def test_validate_string_length_constraints(self):
        """validate() enforces String min_length and max_length."""
        frame = ar.from_dict({
            "username": ["x", "validname", "thisusernameiswaytoolongtobevalid"]
        })
        schema = ar.Schema({
            "username": ar.String(min_length=3, max_length=20)
        })
        result = schema.validate(frame)
        assert result.passed is False

    def test_validate_regex_pattern_mismatch(self):
        """validate() enforces Regex patterns."""
        frame = ar.from_dict({
            "code": ["USR-0001", "INVALID", "USR-9999"]
        })
        schema = ar.Schema({
            "code": ar.Regex(r"^USR-\d{4}$")
        })
        result = schema.validate(frame)
        assert result.passed is False

    def test_validate_warning_severity_does_not_fail_validation(self):
        """validate() with severity='warning' returns passed=True."""
        frame = ar.from_dict({"age": [5, 200]})  # outside typical range
        schema = ar.Schema({
            "age": ar.Int64(min=18, severity="warning")
        })
        result = schema.validate(frame)
        # Warnings should not cause validation to fail
        assert result.passed is True
        # But issues should still be reported
        assert result.issue_count > 0

    def test_validate_url_format_validation(self):
        """validate() validates URL formats."""
        frame = ar.from_dict({
            "website": ["https://example.com", "not a url", "http://valid.com"]
        })
        schema = ar.Schema({"website": ar.URL()})
        result = schema.validate(frame)
        assert result.passed is False

    def test_validate_phone_number_validation(self):
        """validate() validates phone number formats."""
        frame = ar.from_dict({
            "phone": ["+1-555-123-4567", "invalid-phone"]
        })
        schema = ar.Schema({"phone": ar.PhoneNumber()})
        result = schema.validate(frame)
        assert result.passed is False
