import pytest
import pandas as pd
import arnio as ar
import json
import warnings


class TestSchemaToJson:
    """Tests for Schema.to_json() serialization method."""

    def test_simple_schema_serializes_to_valid_json(self):
        """A simple schema should serialize to valid JSON."""
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "name": ar.String(),
        })
        json_str = schema.to_json()

        # Should be valid JSON
        data = json.loads(json_str)
        assert isinstance(data, dict)

    def test_json_serialization_is_stable(self):
        """Multiple to_json() calls should produce identical output."""
        schema = ar.Schema({
            "id": ar.Int64(),
            "value": ar.Float64(nullable=True),
        })
        json1 = schema.to_json()
        json2 = schema.to_json()

        assert json1 == json2

    def test_schema_with_rules_emits_user_warning(self):
        """Schema with cross-field rules should emit UserWarning when serializing."""
        def cross_field_rule(df):
            return []

        schema = ar.Schema(
            {"a": ar.String(), "b": ar.String()},
            rules=[cross_field_rule],
        )

        with pytest.warns(UserWarning):
            schema.to_json()

    def test_schema_with_rules_sets_rules_omitted_flag(self):
        """Serialized schema with rules should set 'rules_omitted': true."""
        def cross_field_rule(df):
            return []

        schema = ar.Schema(
            {"a": ar.String(), "b": ar.String()},
            rules=[cross_field_rule],
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            json_str = schema.to_json()

        data = json.loads(json_str)
        assert data.get("rules_omitted") is True

    def test_schema_without_rules_no_warning(self):
        """Schema without rules should not emit warning."""
        schema = ar.Schema({"id": ar.Int64()})

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            schema.to_json()

            rule_warnings = [x for x in w
                           if issubclass(x.category, UserWarning)
                           and "rule" in str(x.message).lower()]
            assert len(rule_warnings) == 0

    def test_round_trip_preserves_field_definitions(self):
        """to_json() -> from_json() should preserve field definitions."""
        schema = ar.Schema({
            "id": ar.Int64(nullable=False, unique=True),
            "email": ar.Email(nullable=False),
            "age": ar.Int64(nullable=True, min=0, max=150),
        })

        json_str = schema.to_json()
        restored = ar.Schema.from_json(json_str)

        # Both should validate identical valid frames
        frame = ar.from_pandas(pd.DataFrame({
            "id": [1, 2, 3],
            "email": ["a@example.com", "b@example.com", "c@example.com"],
            "age": [25, 30, 35],
        }))

        result1 = schema.validate(frame)
        result2 = restored.validate(frame)

        assert result1.passed is True
        assert result2.passed is True


class TestSchemaValidate:
    """Tests for Schema.validate() method."""

    def test_valid_frame_passes_validation(self):
        """Frame matching schema should pass validation."""
        schema = ar.Schema({
            "id": ar.Int64(),
            "name": ar.String(),
        })
        frame = ar.from_pandas(pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
        }))

        result = schema.validate(frame)
        assert result.passed is True
        assert result.issue_count == 0

    def test_invalid_dtype_produces_issue(self):
        """Frame with wrong dtype should produce validation issue."""
        schema = ar.Schema({"id": ar.Int64()})
        frame = ar.from_pandas(pd.DataFrame({
            "id": ["not", "integers"],
        }))

        result = schema.validate(frame)
        assert result.passed is False
        assert result.issue_count > 0

    def test_nullable_false_with_nulls_fails(self):
        """Non-nullable field with nulls should fail validation."""
        schema = ar.Schema({
            "required_field": ar.String(nullable=False),
        })
        frame = ar.from_pandas(pd.DataFrame({
            "required_field": ["value", None, "another"],
        }))

        result = schema.validate(frame)
        assert result.passed is False

    def test_nullable_true_with_nulls_passes(self):
        """Nullable field with nulls should pass validation."""
        schema = ar.Schema({
            "optional_field": ar.String(nullable=True),
        })
        frame = ar.from_pandas(pd.DataFrame({
            "optional_field": ["value", None, "another"],
        }))

        result = schema.validate(frame)
        assert result.passed is True

    def test_max_errors_limits_issue_count(self):
        """max_errors parameter should limit returned issues."""
        schema = ar.Schema({
            "value": ar.Int64(min=100),
        })
        frame = ar.from_pandas(pd.DataFrame({
            "value": [1, 2, 3, 4, 5],  # All violate min constraint
        }))

        result_all = schema.validate(frame)
        result_limited = schema.validate(frame, max_errors=2)

        assert result_limited.issue_count <= 2
        assert result_limited.issue_count <= result_all.issue_count

    def test_max_errors_none_returns_all_issues(self):
        """max_errors=None should return all validation issues."""
        schema = ar.Schema({
            "value": ar.Int64(min=100),
        })
        frame = ar.from_pandas(pd.DataFrame({
            "value": [1, 2, 3, 4, 5],  # All fail min constraint
        }))

        result = schema.validate(frame, max_errors=None)
        assert result.issue_count > 1

    def test_unique_constraint_violation(self):
        """Duplicate values in unique field should fail."""
        schema = ar.Schema({
            "id": ar.Int64(unique=True),
        })
        frame = ar.from_pandas(pd.DataFrame({
            "id": [1, 2, 2, 3],  # 2 is duplicated
        }))

        result = schema.validate(frame)
        assert result.passed is False

    def test_email_validation(self):
        """Email field should validate email format."""
        schema = ar.Schema({
            "email": ar.Email(),
        })
        frame = ar.from_pandas(pd.DataFrame({
            "email": ["valid@example.com", "not-an-email"],
        }))

        result = schema.validate(frame)
        assert result.passed is False

    def test_numeric_min_constraint(self):
        """Numeric field should validate min constraint."""
        schema = ar.Schema({
            "age": ar.Int64(min=0),
        })
        frame = ar.from_pandas(pd.DataFrame({
            "age": [-1, 25, 50],
        }))

        result = schema.validate(frame)
        assert result.passed is False

    def test_numeric_max_constraint(self):
        """Numeric field should validate max constraint."""
        schema = ar.Schema({
            "age": ar.Int64(max=150),
        })
        frame = ar.from_pandas(pd.DataFrame({
            "age": [25, 200, 75],
        }))

        result = schema.validate(frame)
        assert result.passed is False

    def test_validation_issue_has_row_index(self):
        """ValidationIssue should include row_index (1-based)."""
        schema = ar.Schema({
            "value": ar.Int64(min=10),
        })
        frame = ar.from_pandas(pd.DataFrame({
            "value": [5],  # Violates min constraint
        }))

        result = schema.validate(frame)
        assert len(result.issues) > 0
        issue = result.issues[0]
        assert hasattr(issue, 'row_index')
        assert issue.row_index >= 1

    def test_empty_frame_validation(self):
        """Empty frame should pass validation if structure matches."""
        schema = ar.Schema({
            "id": ar.Int64(),
            "name": ar.String(),
        })
        frame = ar.from_pandas(pd.DataFrame({
            "id": pd.Series([], dtype='int64'),
            "name": pd.Series([], dtype='object'),
        }))

        result = schema.validate(frame)
        assert result.passed is True

    def test_all_fields_with_constraints(self):
        """Complex schema with multiple field types and constraints."""
        schema = ar.Schema({
            "id": ar.Int64(nullable=False, unique=True),
            "email": ar.Email(nullable=False),
            "score": ar.Float64(nullable=True, min=0.0, max=100.0),
            "active": ar.Bool(nullable=False),
        })
        frame = ar.from_pandas(pd.DataFrame({
            "id": [1, 2, 3],
            "email": ["a@test.com", "b@test.com", "c@test.com"],
            "score": [85.5, 90.0, None],
            "active": [True, True, False],
        }))

        result = schema.validate(frame)
        assert result.passed is True
