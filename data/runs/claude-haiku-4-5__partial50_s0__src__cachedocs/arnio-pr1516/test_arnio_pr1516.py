import pytest
import json
import warnings
import pandas as pd
import arnio as ar


class TestSchemaToJson:
    """Tests for Schema.to_json() serialization."""

    def test_to_json_basic_returns_string(self):
        """to_json() should return a valid JSON string."""
        schema = ar.Schema({
            "id": ar.Int64(),
            "name": ar.String(),
        })
        json_str = schema.to_json()
        assert isinstance(json_str, str)
        payload = json.loads(json_str)
        assert isinstance(payload, dict)

    def test_to_json_includes_all_fields(self):
        """JSON payload should include all schema fields."""
        schema = ar.Schema({
            "id": ar.Int64(),
            "email": ar.Email(),
            "active": ar.Bool(),
        })
        payload = json.loads(schema.to_json())
        assert "fields" in payload
        assert set(payload["fields"].keys()) == {"id", "email", "active"}

    def test_to_json_fields_sorted_alphabetically(self):
        """Field names in JSON should be sorted alphabetically."""
        schema = ar.Schema({
            "zebra": ar.String(),
            "apple": ar.String(),
            "monkey": ar.String(),
        })
        payload = json.loads(schema.to_json())
        field_keys = list(payload["fields"].keys())
        assert field_keys == ["apple", "monkey", "zebra"]

    def test_to_json_includes_strict_mode(self):
        """JSON should include the strict mode flag."""
        schema_strict = ar.Schema({"id": ar.Int64()}, strict=True)
        assert json.loads(schema_strict.to_json())["strict"] is True

        schema_lenient = ar.Schema({"id": ar.Int64()}, strict=False)
        assert json.loads(schema_lenient.to_json())["strict"] is False

    def test_to_json_unique_constraints(self):
        """JSON should include unique constraints as a list."""
        schema = ar.Schema(
            {"user_id": ar.Int64(), "course_id": ar.Int64()},
            unique=["user_id", "course_id"]
        )
        payload = json.loads(schema.to_json())
        assert payload["unique"] == ["user_id", "course_id"]

    def test_to_json_null_unique_when_absent(self):
        """JSON should have null unique field when no constraints exist."""
        schema = ar.Schema({"id": ar.Int64()})
        payload = json.loads(schema.to_json())
        assert payload["unique"] is None

    def test_to_json_warns_on_cross_field_rules(self):
        """to_json() should emit UserWarning when cross-field rules exist."""
        schema = ar.Schema({"id": ar.Int64(), "name": ar.String()})
        schema.rules = [("example_rule", lambda df: True)]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            schema.to_json()

            assert len(w) >= 1
            assert issubclass(w[-1].category, UserWarning)
            assert "rules" in str(w[-1].message).lower()

    def test_to_json_rules_omitted_flag_set_with_rules(self):
        """JSON should include rules_omitted=true when rules are present."""
        schema = ar.Schema({"id": ar.Int64()})
        schema.rules = [("rule", lambda df: True)]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            payload = json.loads(schema.to_json())

        assert payload.get("rules_omitted") is True

    def test_to_json_no_rules_omitted_flag_without_rules(self):
        """JSON should not have rules_omitted key when no rules exist."""
        schema = ar.Schema({"id": ar.Int64()})
        payload = json.loads(schema.to_json())
        assert "rules_omitted" not in payload


class TestSchemaValidate:
    """Tests for Schema.validate() method."""

    def test_validate_returns_validation_result(self):
        """validate() should return a ValidationResult object."""
        schema = ar.Schema({"id": ar.Int64()})
        frame = ar.from_pandas(pd.DataFrame({"id": [1, 2, 3]}))
        result = schema.validate(frame)

        assert hasattr(result, "passed")
        assert hasattr(result, "issues")
        assert hasattr(result, "issue_count")

    def test_validate_passes_valid_data(self):
        """Valid data should pass validation."""
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "name": ar.String(nullable=False),
        })
        frame = ar.from_pandas(pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
        }))

        result = schema.validate(frame)
        assert result.passed is True
        assert result.issue_count == 0

    def test_validate_fails_on_dtype_mismatch(self):
        """Validation should fail when column dtype doesn't match schema."""
        schema = ar.Schema({"age": ar.Int64(nullable=False)})
        frame = ar.from_pandas(pd.DataFrame({
            "age": ["not", "numeric", "values"],
        }))

        result = schema.validate(frame)
        assert result.passed is False
        assert result.issue_count > 0

    def test_validate_enforces_nullable_false(self):
        """Validation should fail when nullable=False and nulls are present."""
        schema = ar.Schema({"id": ar.Int64(nullable=False)})
        frame = ar.from_pandas(pd.DataFrame({"id": [1, None, 3]}))

        result = schema.validate(frame)
        assert result.passed is False
        assert result.issue_count > 0

    def test_validate_allows_nulls_when_nullable_true(self):
        """Validation should pass when nullable=True and nulls are present."""
        schema = ar.Schema({"id": ar.Int64(nullable=True)})
        frame = ar.from_pandas(pd.DataFrame({"id": [1, None, 3]}))

        result = schema.validate(frame)
        assert result.passed is True

    def test_validate_max_errors_limits_issues(self):
        """max_errors parameter should limit the number of returned issues."""
        schema = ar.Schema({"age": ar.Int64(min=0, max=100)})
        frame = ar.from_pandas(pd.DataFrame({
            "age": [-1, -2, -3, -4, -5],
        }))

        result = schema.validate(frame, max_errors=2)
        assert result.issue_count <= 2

    def test_validate_multiple_fields_checked(self):
        """Validation should check all fields in schema."""
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "email": ar.Email(nullable=False),
        })
        frame = ar.from_pandas(pd.DataFrame({
            "id": [1, 2],
            "email": ["invalid-email", "also-bad"],
        }))

        result = schema.validate(frame)
        assert result.passed is False
