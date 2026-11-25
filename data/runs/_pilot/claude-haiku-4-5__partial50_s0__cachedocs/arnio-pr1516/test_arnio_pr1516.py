import pytest
import pandas as pd
import arnio as ar
import json


class TestSchemaToJson:
    """Tests for Schema.to_json() method."""

    def test_to_json_returns_string(self):
        """to_json() should return a string."""
        schema = ar.Schema({"id": ar.Int64()})
        result = schema.to_json()
        assert isinstance(result, str)

    def test_to_json_valid_json(self):
        """to_json() should return valid JSON."""
        schema = ar.Schema({"id": ar.Int64()})
        result = schema.to_json()
        payload = json.loads(result)
        assert isinstance(payload, dict)

    def test_to_json_includes_fields(self):
        """to_json() should include defined fields in output."""
        schema = ar.Schema({
            "id": ar.Int64(),
            "name": ar.String(),
            "active": ar.Bool(),
        })
        payload = json.loads(schema.to_json())
        assert "id" in payload
        assert "name" in payload
        assert "active" in payload

    def test_to_json_preserves_field_constraints(self):
        """to_json() should preserve field-level constraints."""
        schema = ar.Schema({
            "id": ar.Int64(nullable=False, unique=True),
            "username": ar.String(min_length=3, max_length=20),
        })
        payload = json.loads(schema.to_json())
        assert payload is not None
        assert "id" in payload
        assert "username" in payload

    def test_to_json_roundtrip_basic(self):
        """to_json() followed by from_json() should restore schema functionality."""
        original = ar.Schema({
            "id": ar.Int64(nullable=False),
            "name": ar.String(nullable=False),
        })
        json_str = original.to_json()
        restored = ar.Schema.from_json(json_str)
        
        df = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})
        frame = ar.from_pandas(df)
        result = restored.validate(frame)
        assert result.passed

    def test_to_json_roundtrip_with_datetime(self):
        """to_json() should preserve DateTime field format during round-trip."""
        original = ar.Schema({
            "created_at": ar.DateTime(format="%Y-%m-%d"),
        })
        json_str = original.to_json()
        restored = ar.Schema.from_json(json_str)
        
        df = pd.DataFrame({"created_at": ["2025-01-01", "2025-12-31"]})
        frame = ar.from_pandas(df)
        result = restored.validate(frame)
        assert result.passed

    def test_to_json_with_schema_metadata(self):
        """to_json() should preserve schema-level metadata (strict, unique)."""
        schema = ar.Schema(
            {"user_id": ar.Int64(), "course_id": ar.Int64()},
            strict=True,
            unique=["user_id", "course_id"],
        )
        payload = json.loads(schema.to_json())
        assert payload is not None


class TestSchemaValidate:
    """Tests for Schema.validate() method."""

    def test_validate_returns_validation_result(self):
        """validate() should return a ValidationResult with passed, issues, and issue_count."""
        schema = ar.Schema({"id": ar.Int64()})
        frame = ar.from_pandas(pd.DataFrame({"id": [1, 2, 3]}))
        result = schema.validate(frame)
        assert hasattr(result, "passed")
        assert hasattr(result, "issues")
        assert hasattr(result, "issue_count")

    def test_validate_passing_data(self):
        """validate() should return passed=True for data matching the schema."""
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

    def test_validate_dtype_mismatch(self):
        """validate() should detect dtype mismatches."""
        schema = ar.Schema({"id": ar.Int64()})
        frame = ar.from_pandas(pd.DataFrame({"id": ["text", "data"]}))
        result = schema.validate(frame)
        assert result.passed is False
        assert result.issue_count > 0

    def test_validate_nullable_constraint_violated(self):
        """validate() should reject null values when nullable=False."""
        schema = ar.Schema({"id": ar.Int64(nullable=False)})
        frame = ar.from_pandas(pd.DataFrame({"id": [1, None, 3]}))
        result = schema.validate(frame)
        assert result.passed is False
        assert result.issue_count > 0

    def test_validate_nullable_constraint_satisfied(self):
        """validate() should allow nulls when nullable=True."""
        schema = ar.Schema({"id": ar.Int64(nullable=True)})
        frame = ar.from_pandas(pd.DataFrame({"id": [1, None, 3]}))
        result = schema.validate(frame)
        assert result.passed is True

    def test_validate_unique_constraint_violated(self):
        """validate() should detect duplicate values in unique columns."""
        schema = ar.Schema({"id": ar.Int64(unique=True)})
        frame = ar.from_pandas(pd.DataFrame({"id": [1, 2, 2, 3]}))
        result = schema.validate(frame)
        assert result.passed is False

    def test_validate_min_max_constraints(self):
        """validate() should enforce numeric min/max constraints."""
        schema = ar.Schema({
            "age": ar.Int64(min=0, max=150),
            "score": ar.Float64(min=0.0, max=100.0),
        })
        frame = ar.from_pandas(pd.DataFrame({
            "age": [25, -5, 30],
            "score": [95.5, 105.0, 88.0],
        }))
        result = schema.validate(frame)
        assert result.passed is False

    def test_validate_email_field(self):
        """validate() should validate Email field format."""
        schema = ar.Schema({"email": ar.Email(nullable=False)})
        frame = ar.from_pandas(pd.DataFrame({
            "email": ["alice@example.com", "invalid-email"],
        }))
        result = schema.validate(frame)
        assert result.passed is False

    def test_validate_with_max_errors_parameter(self):
        """validate(max_errors=n) should limit the number of reported issues."""
        schema = ar.Schema({"id": ar.Int64(min=0)})
        frame = ar.from_pandas(pd.DataFrame({"id": [-1, -2, -3, -4, -5]}))
        result = schema.validate(frame, max_errors=2)
        assert result.issue_count <= 2

    def test_validate_warning_severity_does_not_fail(self):
        """validate() should report warning-level issues without failing validation."""
        schema = ar.Schema({
            "age": ar.Int64(min=18, severity="warning"),
        })
        frame = ar.from_pandas(pd.DataFrame({"age": [10, 25, 30]}))
        result = schema.validate(frame)
        assert result.passed is True
        assert result.issue_count > 0

    def test_validate_issue_row_index_one_based(self):
        """ValidationIssue.row_index should use 1-based indexing (header not counted)."""
        schema = ar.Schema({"id": ar.Int64(min=0)})
        frame = ar.from_pandas(pd.DataFrame({"id": [1, -2, 3]}))
        result = schema.validate(frame)
        assert len(result.issues) > 0
        row_indices = {issue.row_index for issue in result.issues}
        assert 2 in row_indices  # Row 2 (1-based) contains -2

    def test_validate_issue_includes_column_rule_message(self):
        """ValidationIssue should contain column, rule, and message fields."""
        schema = ar.Schema({"id": ar.Int64()})
        frame = ar.from_pandas(pd.DataFrame({"id": ["not-int"]}))
        result = schema.validate(frame)
        assert len(result.issues) > 0
        issue = result.issues[0]
        assert issue.column == "id"
        assert hasattr(issue, "rule")
        assert hasattr(issue, "message")
        assert isinstance(issue.message, str)
        assert len(issue.message) > 0
