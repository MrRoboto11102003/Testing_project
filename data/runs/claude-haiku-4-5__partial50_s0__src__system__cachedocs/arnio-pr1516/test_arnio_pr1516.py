import pytest
import pandas as pd
import numpy as np
import arnio as ar
import json
from pathlib import Path


class TestSchemaToJson:
    """Tests for Schema.to_json() serialization."""

    def test_basic_schema_to_json_roundtrip(self):
        """Basic schema serialization and deserialization."""
        schema = ar.Schema({
            "id": ar.Int64(nullable=False, unique=True),
            "name": ar.String(nullable=False),
        })
        
        json_str = schema.to_json()
        payload = json.loads(json_str)
        
        assert isinstance(json_str, str)
        assert "fields" in payload
        assert "id" in payload["fields"]
        assert "name" in payload["fields"]
        
        restored = ar.Schema.from_json(json_str)
        assert len(restored.fields) == 2

    def test_schema_to_json_with_unique_constraint(self):
        """Schema JSON preserves unique column constraints."""
        schema = ar.Schema(
            {"user_id": ar.Int64(), "email": ar.String()},
            unique=["user_id"]
        )
        
        json_str = schema.to_json()
        payload = json.loads(json_str)
        
        assert payload["unique"] == ["user_id"]

    def test_schema_to_json_with_strict_mode(self):
        """Schema JSON includes strict flag."""
        schema_strict = ar.Schema({"id": ar.Int64()}, strict=True)
        schema_loose = ar.Schema({"id": ar.Int64()}, strict=False)
        
        payload_strict = json.loads(schema_strict.to_json())
        payload_loose = json.loads(schema_loose.to_json())
        
        assert payload_strict["strict"] is True
        assert payload_loose["strict"] is False

    def test_schema_to_json_multiple_field_types(self):
        """Schema JSON handles various field types."""
        schema = ar.Schema({
            "id": ar.Int64(),
            "score": ar.Float64(),
            "name": ar.String(),
            "active": ar.Bool(),
            "email": ar.Email(),
            "website": ar.URL(),
        })
        
        json_str = schema.to_json()
        payload = json.loads(json_str)
        
        assert len(payload["fields"]) == 6

    def test_schema_from_json_restores_field_constraints(self):
        """Schema restored from JSON preserves field constraints."""
        schema = ar.Schema({
            "age": ar.Int64(min=0, max=150, nullable=False),
            "email": ar.Email(nullable=False),
        })
        
        json_str = schema.to_json()
        restored = ar.Schema.from_json(json_str)
        
        # Validate that constraints are preserved
        df = pd.DataFrame({"age": [25, 30], "email": ["a@b.com", "c@d.com"]})
        frame = ar.from_pandas(df)
        result = restored.validate(frame)
        assert result.passed is True


class TestSchemaValidation:
    """Tests for Schema.validate() with various scenarios."""

    def test_validate_valid_dataframe(self):
        """Validate a valid DataFrame passes validation."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
        })
        
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "name": ar.String(nullable=False),
        })
        
        result = schema.validate(frame)
        
        assert result.passed is True
        assert result.issue_count == 0

    def test_validate_invalid_email(self):
        """Validation catches invalid email format."""
        df = pd.DataFrame({
            "email": ["valid@example.com", "not-an-email", "also@valid.com"],
        })
        
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "email": ar.Email(nullable=False),
        })
        
        result = schema.validate(frame)
        
        assert result.passed is False
        assert result.issue_count >= 1
        assert any(issue.row_index == 2 for issue in result.issues)
        assert any(issue.column == "email" for issue in result.issues)

    def test_validate_null_violation(self):
        """Validation detects null when not nullable."""
        df = pd.DataFrame({
            "id": [1, None, 3],
        })
        
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
        })
        
        result = schema.validate(frame)
        
        assert result.passed is False
        assert any(issue.row_index == 2 for issue in result.issues)

    def test_validate_numeric_min_max_constraints(self):
        """Validation checks numeric min/max constraints."""
        df = pd.DataFrame({
            "score": [50, 150, 25],
        })
        
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "score": ar.Int64(min=0, max=100),
        })
        
        result = schema.validate(frame)
        
        assert result.passed is False
        assert any(issue.row_index == 2 for issue in result.issues)

    def test_validate_unique_constraint(self):
        """Validation checks unique constraint."""
        df = pd.DataFrame({
            "user_id": [1, 2, 1],
            "name": ["Alice", "Bob", "Alice2"],
        })
        
        frame = ar.from_pandas(df)
        schema = ar.Schema(
            {"user_id": ar.Int64(), "name": ar.String()},
            unique=["user_id"]
        )
        
        result = schema.validate(frame)
        assert result.passed is False

    def test_validate_url_format(self):
        """Validation checks URL format."""
        df = pd.DataFrame({
            "website": ["https://example.com", "invalid_url", "http://valid.org"],
        })
        
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "website": ar.URL(nullable=False),
        })
        
        result = schema.validate(frame)
        
        assert result.passed is False
        assert any(issue.row_index == 2 for issue in result.issues)

    def test_validate_max_errors_limit(self):
        """Validation respects max_errors parameter."""
        df = pd.DataFrame({
            "email": ["invalid1", "invalid2", "invalid3", "invalid4"],
        })
        
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "email": ar.Email(),
        })
        
        result = schema.validate(frame, max_errors=2)
        
        assert result.issue_count <= 2

    def test_validation_result_attributes(self):
        """ValidationResult has expected attributes and methods."""
        df = pd.DataFrame({
            "email": ["bad"],
        })
        
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "email": ar.Email(),
        })
        
        result = schema.validate(frame)
        
        assert hasattr(result, "passed")
        assert hasattr(result, "issues")
        assert hasattr(result, "issue_count")
        assert callable(getattr(result, "summary", None))
        assert callable(getattr(result, "to_markdown", None))

    def test_validation_issue_row_index_convention(self):
        """ValidationIssue row_index is 1-based for data rows."""
        df = pd.DataFrame({
            "value": ["a", "b", "not_int"],
        })
        
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "value": ar.Int64(),
        })
        
        result = schema.validate(frame)
        
        # row_index should be 3 (1-based, third data row)
        assert any(issue.row_index == 3 for issue in result.issues)

    def test_csv_to_schema_validation_workflow(self, tmp_path):
        """End-to-end CSV load and validation workflow."""
        csv_path = tmp_path / "customers.csv"
        csv_path.write_text(
            "id,email,score\n1,alice@example.com,95\n2,bad_email,105\n"
        )
        
        frame = ar.read_csv(str(csv_path))
        schema = ar.Schema({
            "id": ar.Int64(),
            "email": ar.Email(),
            "score": ar.Int64(min=0, max=100),
        })
        
        result = schema.validate(frame)
        
        assert result.passed is False
        assert result.issue_count >= 2

    def test_schema_roundtrip_preserves_validation_behavior(self):
        """Schema JSON roundtrip preserves validation behavior."""
        original_schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "email": ar.Email(),
        })
        
        json_str = original_schema.to_json()
        restored_schema = ar.Schema.from_json(json_str)
        
        # Test with valid data
        df = pd.DataFrame({
            "id": [1, 2],
            "email": ["alice@example.com", "bob@example.com"],
        })
        frame = ar.from_pandas(df)
        
        result1 = original_schema.validate(frame)
        result2 = restored_schema.validate(frame)
        
        assert result1.passed == result2.passed

    def test_validation_summary_output(self):
        """ValidationResult summary has expected structure."""
        df = pd.DataFrame({
            "value": ["a", "b"],
        })
        
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "value": ar.Int64(),
        })
        
        result = schema.validate(frame)
        summary = result.summary()
        
        assert isinstance(summary, dict)
        assert result.issue_count > 0

    def test_validation_with_multiple_field_types(self):
        """Validate schema with multiple field types simultaneously."""
        df = pd.DataFrame({
            "id": [1, 2],
            "name": ["Alice", "Bob"],
            "active": [True, False],
            "score": [95.5, 87.3],
            "email": ["alice@test.com", "bob@test.com"],
        })
        
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "id": ar.Int64(),
            "name": ar.String(),
            "active": ar.Bool(),
            "score": ar.Float64(),
            "email": ar.Email(),
        })
        
        result = schema.validate(frame)
        assert result.passed is True

    def test_validation_with_nullable_fields(self):
        """Validate nullable fields allow None values."""
        df = pd.DataFrame({
            "id": [1, 2],
            "notes": [None, "has notes"],
        })
        
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "notes": ar.String(nullable=True),
        })
        
        result = schema.validate(frame)
        assert result.passed is True
