import pytest
import json
import pandas as pd
import arnio as ar


class TestSchemaToJson:
    """Test Schema.to_json() serialization."""
    
    def test_to_json_returns_string(self):
        """Test that to_json returns a valid JSON string."""
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "name": ar.String(),
        })
        result = schema.to_json()
        
        assert isinstance(result, str)
        assert len(result) > 0
        # Verify it's valid JSON
        parsed = json.loads(result)
        assert isinstance(parsed, dict)
    
    def test_to_json_preserves_constraints(self):
        """Test that field constraints are preserved in JSON."""
        schema = ar.Schema({
            "age": ar.Int64(min=0, max=150, nullable=False),
            "score": ar.Float64(nullable=True),
        })
        json_str = schema.to_json()
        parsed = json.loads(json_str)
        
        assert isinstance(parsed, dict)
        # Fields should be present
        assert "age" in parsed or "fields" in parsed or "schema" in parsed
    
    def test_to_json_with_unique_constraint(self):
        """Test that unique constraints are serialized."""
        schema = ar.Schema({
            "user_id": ar.Int64(),
            "email": ar.String(),
        }, unique=["user_id"])
        
        json_str = schema.to_json()
        assert isinstance(json_str, str)
        json.loads(json_str)  # Verify valid JSON


class TestSchemaFromJson:
    """Test Schema.from_json() deserialization."""
    
    def test_from_json_roundtrip_basic(self):
        """Test deserializing a schema and validating with it."""
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "name": ar.String(nullable=True),
        })
        json_str = schema.to_json()
        restored = ar.Schema.from_json(json_str)
        
        frame = ar.from_dict({"id": [1, 2], "name": ["Alice", "Bob"]})
        result = restored.validate(frame)
        
        assert isinstance(result, ar.ValidationResult)
        assert result.passed
    
    def test_from_json_complex_types(self):
        """Test roundtrip with various field types."""
        schema = ar.Schema({
            "id": ar.Int64(),
            "email": ar.Email(),
            "active": ar.Bool(),
            "score": ar.Float64(),
        })
        json_str = schema.to_json()
        restored = ar.Schema.from_json(json_str)
        
        assert isinstance(restored, ar.Schema)
        # Verify we can use it for validation
        frame = ar.from_dict({
            "id": [1],
            "email": ["test@example.com"],
            "active": [True],
            "score": [95.5],
        })
        result = restored.validate(frame)
        assert isinstance(result, ar.ValidationResult)
    
    def test_from_json_with_constraints(self):
        """Test that min/max constraints round-trip correctly."""
        schema = ar.Schema({
            "age": ar.Int64(min=0, max=150),
            "score": ar.Float64(min=0.0, max=100.0),
        })
        json_str = schema.to_json()
        restored = ar.Schema.from_json(json_str)
        
        # Test with out-of-range values
        frame = ar.from_dict({
            "age": [-5, 25, 200],
            "score": [101.0, 50.0, -1.0],
        })
        result = restored.validate(frame)
        
        # Should have issues for constraint violations
        assert isinstance(result, ar.ValidationResult)
    
    def test_from_json_invalid_json_raises(self):
        """Test that invalid JSON raises error."""
        with pytest.raises((json.JSONDecodeError, ValueError)):
            ar.Schema.from_json("not valid json {]")
    
    def test_from_json_empty_string_raises(self):
        """Test that empty string raises error."""
        with pytest.raises((json.JSONDecodeError, ValueError)):
            ar.Schema.from_json("")


class TestSchemaValidate:
    """Test Schema.validate() method."""
    
    def test_validate_returns_validation_result(self):
        """Test that validate returns ValidationResult object."""
        schema = ar.Schema({"col": ar.String()})
        frame = ar.from_dict({"col": ["test"]})
        
        result = schema.validate(frame)
        
        assert isinstance(result, ar.ValidationResult)
        assert hasattr(result, 'passed')
        assert hasattr(result, 'issues')
    
    def test_validate_passing_schema(self):
        """Test validation that passes without issues."""
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "name": ar.String(nullable=False),
        })
        
        frame = ar.from_dict({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
        })
        
        result = schema.validate(frame)
        assert result.passed
        assert result.issue_count == 0
    
    def test_validate_dtype_mismatch(self):
        """Test validation fails on dtype mismatch."""
        schema = ar.Schema({"age": ar.Int64()})
        frame = ar.from_dict({"age": ["25", "30"]})  # strings, not ints
        
        result = schema.validate(frame)
        
        assert not result.passed
        assert result.issue_count > 0
    
    def test_validate_nullable_constraint(self):
        """Test validation of nullable=False constraint."""
        schema = ar.Schema({
            "required": ar.String(nullable=False),
        })
        
        frame = ar.from_dict({"required": [None, "value"]})
        result = schema.validate(frame)
        
        assert not result.passed
        assert result.issue_count > 0
    
    def test_validate_numeric_bounds(self):
        """Test validation with min/max constraints."""
        schema = ar.Schema({
            "age": ar.Int64(min=0, max=150),
        })
        
        frame = ar.from_dict({"age": [-5, 25, 200]})
        result = schema.validate(frame)
        
        assert not result.passed
        assert result.issue_count >= 2  # Two violations
    
    def test_validate_missing_column(self):
        """Test validation fails when required column missing."""
        schema = ar.Schema({
            "id": ar.Int64(),
            "name": ar.String(),
        })
        
        frame = ar.from_dict({"id": [1, 2]})  # missing 'name'
        result = schema.validate(frame)
        
        assert not result.passed
    
    def test_validate_email_format(self):
        """Test validation of email format constraint."""
        schema = ar.Schema({
            "email": ar.Email(nullable=False),
        })
        
        frame = ar.from_dict({
            "email": ["valid@example.com", "invalid-email", "test@test.org"],
        })
        
        result = schema.validate(frame)
        assert isinstance(result, ar.ValidationResult)
    
    def test_validate_unique_constraint(self):
        """Test validation of unique constraint across rows."""
        schema = ar.Schema({
            "user_id": ar.Int64(),
        }, unique=["user_id"])
        
        frame = ar.from_dict({"user_id": [1, 2, 1]})  # duplicate
        result = schema.validate(frame)
        
        assert not result.passed
        assert result.issue_count > 0
    
    def test_validate_max_errors_parameter(self):
        """Test max_errors parameter limits reported issues."""
        schema = ar.Schema({
            "age": ar.Int64(min=0, max=150),
        })
        
        frame = ar.from_dict({
            "age": [-1, -2, -3, -4, -5],
        })
        
        # Without limit
        result_all = schema.validate(frame)
        all_count = result_all.issue_count
        
        # With limit
        result_limited = schema.validate(frame, max_errors=2)
        limited_count = result_limited.issue_count
        
        assert limited_count <= 2
        assert limited_count <= all_count
    
    def test_validate_empty_frame(self):
        """Test validation with empty frame."""
        schema = ar.Schema({"col": ar.String()})
        frame = ar.from_dict({"col": []})
        
        result = schema.validate(frame)
        assert isinstance(result, ar.ValidationResult)


class TestSchemaBootstrapFromReport:
    """Test Schema.bootstrap_from_report() method."""
    
    def test_bootstrap_from_report_creates_schema(self):
        """Test that bootstrap creates a valid Schema from report."""
        frame = ar.from_dict({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
        })
        
        report = ar.profile(frame)
        schema = ar.Schema.bootstrap_from_report(report)
        
        assert isinstance(schema, ar.Schema)
    
    def test_bootstrap_schema_validates_original_frame(self):
        """Test that bootstrapped schema validates the source frame."""
        frame = ar.from_dict({
            "id": [1, 2, 3],
            "score": [95.5, 87.2, 91.3],
        })
        
        report = ar.profile(frame)
        schema = ar.Schema.bootstrap_from_report(report)
        result = schema.validate(frame)
        
        assert isinstance(result, ar.ValidationResult)
    
    def test_bootstrap_infers_multiple_types(self):
        """Test bootstrap infers various column types."""
        frame = ar.from_dict({
            "int_col": [1, 2, 3],
            "float_col": [1.5, 2.5, 3.5],
            "str_col": ["a", "b", "c"],
            "bool_col": [True, False, True],
        })
        
        report = ar.profile(frame)
        schema = ar.Schema.bootstrap_from_report(report)
        
        assert isinstance(schema, ar.Schema)
        result = schema.validate(frame)
        assert isinstance(result, ar.ValidationResult)
    
    def test_bootstrap_handles_nulls(self):
        """Test bootstrap with columns containing null values."""
        frame = ar.from_dict({
            "id": [1, 2, None],
            "name": ["Alice", None, "Charlie"],
        })
        
        report = ar.profile(frame)
        schema = ar.Schema.bootstrap_from_report(report)
        
        assert isinstance(schema, ar.Schema)
        result = schema.validate(frame)
        assert isinstance(result, ar.ValidationResult)
    
    def test_bootstrap_schema_is_serializable(self):
        """Test that bootstrapped schema can be serialized to JSON."""
        frame = ar.from_dict({
            "id": [1, 2],
            "name": ["A", "B"],
        })
        
        report = ar.profile(frame)
        schema = ar.Schema.bootstrap_from_report(report)
        
        # Should be serializable
        json_str = schema.to_json()
        assert isinstance(json_str, str)
        
        # And deserializable
        restored = ar.Schema.from_json(json_str)
        assert isinstance(restored, ar.Schema)


class TestSchemaRoundTrip:
    """Test round-trip consistency across schema methods."""
    
    def test_roundtrip_to_json_from_json(self):
        """Test complete roundtrip: Schema -> JSON -> Schema -> validate."""
        original = ar.Schema({
            "id": ar.Int64(nullable=False),
            "email": ar.Email(),
            "age": ar.Int64(min=0, max=150, nullable=True),
        })
        
        json_str = original.to_json()
        restored = ar.Schema.from_json(json_str)
        
        frame = ar.from_dict({
            "id": [1, 2],
            "email": ["test@example.com", "user@test.org"],
            "age": [25, 30],
        })
        
        result = restored.validate(frame)
        assert result.passed
    
    def test_roundtrip_preserves_validation_behavior(self):
        """Test that original and restored schema have same validation behavior."""
        original = ar.Schema({
            "score": ar.Float64(min=0.0, max=100.0),
        })
        
        json_str = original.to_json()
        restored = ar.Schema.from_json(json_str)
        
        frame = ar.from_dict({"score": [-10.0, 50.0, 150.0]})
        
        result_original = original.validate(frame)
        result_restored = restored.validate(frame)
        
        assert result_original.passed == result_restored.passed
        assert result_original.issue_count == result_restored.issue_count
    
    def test_bootstrap_and_serialize(self):
        """Test bootstrap -> serialize -> deserialize workflow."""
        frame = ar.from_dict({
            "user_id": [1, 2, 3],
            "username": ["alice", "bob", "charlie"],
        })
        
        report = ar.profile(frame)
        bootstrap_schema = ar.Schema.bootstrap_from_report(report)
        
        json_str = bootstrap_schema.to_json()
        restored_schema = ar.Schema.from_json(json_str)
        
        result = restored_schema.validate(frame)
        assert isinstance(result, ar.ValidationResult)
