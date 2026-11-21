import pytest
import json
import warnings
import pandas as pd
import arnio as ar
from arnio.schema import (
    Schema, ValidationResult, ValidationIssue,
    Int64, Float64, String, Bool, Email, URL, Date, UUID, IPv4
)


class TestSchemaToJson:
    """Tests for Schema.to_json() serialization."""
    
    def test_to_json_basic_schema(self):
        """Test serialization of a basic schema to valid JSON."""
        schema = Schema(
            fields={
                "id": Int64(nullable=False),
                "name": String(nullable=False),
            }
        )
        json_str = schema.to_json()
        
        # Verify valid JSON
        payload = json.loads(json_str)
        assert "fields" in payload
        assert "strict" in payload
        assert "unique" in payload
        assert len(payload["fields"]) == 2
    
    def test_to_json_with_rules_warns_and_marks_omitted(self):
        """Test that to_json warns when rules are present and marks them omitted."""
        def custom_rule(df):
            return []
        
        schema = Schema(
            fields={"id": Int64()},
            rules=[custom_rule]
        )
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            json_str = schema.to_json()
            
            # Should emit UserWarning about rules
            assert any("rules cannot be serialized" in str(warning.message) 
                      for warning in w)
        
        payload = json.loads(json_str)
        assert payload.get("rules_omitted") is True
    
    def test_to_json_without_rules_no_warning(self):
        """Test that to_json does not warn when no rules present."""
        schema = Schema(
            fields={"id": Int64(), "name": String()}
        )
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            json_str = schema.to_json()
            
            rule_warnings = [x for x in w if "rules" in str(x.message).lower()]
            assert len(rule_warnings) == 0
        
        payload = json.loads(json_str)
        assert payload.get("rules_omitted") is not True
    
    def test_to_json_preserves_strict_mode(self):
        """Test that to_json preserves strict mode setting."""
        schema_strict = Schema(
            fields={"id": Int64()},
            strict=True
        )
        schema_not_strict = Schema(
            fields={"id": Int64()},
            strict=False
        )
        
        assert json.loads(schema_strict.to_json())["strict"] is True
        assert json.loads(schema_not_strict.to_json())["strict"] is False
    
    def test_to_json_preserves_unique_constraints(self):
        """Test that to_json preserves unique column constraints."""
        schema = Schema(
            fields={"id": Int64(), "email": Email()},
            unique=["id", "email"]
        )
        
        payload = json.loads(schema.to_json())
        assert set(payload["unique"]) == {"id", "email"}
    
    def test_to_json_fields_sorted_alphabetically(self):
        """Test that to_json outputs fields in sorted order."""
        schema = Schema(
            fields={
                "zebra": String(),
                "apple": Int64(),
                "middle": Float64(),
            }
        )
        
        payload = json.loads(schema.to_json())
        field_keys = list(payload["fields"].keys())
        assert field_keys == ["apple", "middle", "zebra"]
    
    def test_to_json_roundtrip_preserves_schema(self):
        """Test that schema survives JSON serialization roundtrip."""
        original = Schema(
            fields={
                "id": Int64(nullable=False, unique=True, min=1, max=1000),
                "email": Email(nullable=False),
                "age": Int64(min=0, max=150),
            },
            strict=True,
            unique=["id"]
        )
        
        json_str = original.to_json()
        restored = Schema.from_json(json_str)
        
        assert restored.strict == original.strict
        assert restored.unique == original.unique
        assert len(restored.fields) == len(original.fields)
        for field_name in original.fields:
            assert field_name in restored.fields


class TestSchemaValidate:
    """Tests for Schema.validate() method."""
    
    def test_validate_passing_data_returns_passed_result(self):
        """Test validation returns passed result for valid data."""
        schema = Schema(
            fields={
                "id": Int64(nullable=False),
                "name": String(nullable=False),
            }
        )
        
        frame = ar.from_dict({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"]
        })
        
        result = schema.validate(frame)
        assert isinstance(result, ValidationResult)
        assert result.passed is True
        assert result.issue_count == 0
        assert len(result.issues) == 0
    
    def test_validate_non_nullable_field_with_null(self):
        """Test validation fails when non-nullable field contains null."""
        schema = Schema(
            fields={"id": Int64(nullable=False)}
        )
        
        frame = ar.from_dict({"id": [1, None, 3]})
        result = schema.validate(frame)
        
        assert result.passed is False
        assert result.issue_count > 0
        assert any(issue.rule == "non_nullable" for issue in result.issues)
    
    def test_validate_numeric_min_max_constraint(self):
        """Test validation enforces numeric min/max constraints."""
        schema = Schema(
            fields={"age": Int64(min=0, max=150)}
        )
        
        # Valid data
        valid_result = schema.validate(ar.from_dict({"age": [0, 50, 150]}))
        assert valid_result.passed is True
        
        # Invalid data
        invalid_result = schema.validate(ar.from_dict({"age": [-1, 200]}))
        assert invalid_result.passed is False
        assert invalid_result.issue_count > 0
    
    def test_validate_unique_constraint_on_field(self):
        """Test validation enforces unique constraint on field."""
        schema = Schema(
            fields={"id": Int64(unique=True)}
        )
        
        # Valid - unique values
        valid_result = schema.validate(ar.from_dict({"id": [1, 2, 3]}))
        assert valid_result.passed is True
        
        # Invalid - duplicate values
        invalid_result = schema.validate(ar.from_dict({"id": [1, 2, 1]}))
        assert invalid_result.passed is False
    
    def test_validate_email_semantic_type(self):
        """Test validation with Email semantic type."""
        schema = Schema(
            fields={"email": Email()}
        )
        
        # Valid emails
        valid_result = schema.validate(ar.from_dict({
            "email": ["user@example.com", "test@domain.org"]
        }))
        assert valid_result.passed is True
        
        # Invalid emails
        invalid_result = schema.validate(ar.from_dict({
            "email": ["not_an_email", "@nodomain.com"]
        }))
        assert invalid_result.passed is False
    
    def test_validate_max_errors_limits_issues(self):
        """Test that max_errors parameter limits number of issues returned."""
        schema = Schema(
            fields={"value": Int64(min=0, max=10)}
        )
        
        frame = ar.from_dict({
            "value": [-1, 20, -5, 15, 30]
        })
        
        unlimited_result = schema.validate(frame)
        limited_result = schema.validate(frame, max_errors=2)
        
        assert limited_result.issue_count <= 2
        assert limited_result.issue_count < unlimited_result.issue_count
    
    def test_validate_result_contains_validation_issues(self):
        """Test that ValidationResult.issues contains ValidationIssue objects."""
        schema = Schema(
            fields={"id": Int64(nullable=False)}
        )
        
        frame = ar.from_dict({"id": [1, None, 3]})
        result = schema.validate(frame)
        
        assert isinstance(result.issues, list)
        assert len(result.issues) > 0
        assert all(isinstance(issue, ValidationIssue) for issue in result.issues)
        
        # Check ValidationIssue structure
        issue = result.issues[0]
        assert hasattr(issue, "column")
        assert hasattr(issue, "rule")
        assert hasattr(issue, "message")
    
    def test_validate_row_count_in_result(self):
        """Test that ValidationResult reports correct row count."""
        schema = Schema(
            fields={"id": Int64()}
        )
        
        frame = ar.from_dict({"id": [1, 2, 3, 4, 5]})
        result = schema.validate(frame)
        
        assert result.row_count == 5
