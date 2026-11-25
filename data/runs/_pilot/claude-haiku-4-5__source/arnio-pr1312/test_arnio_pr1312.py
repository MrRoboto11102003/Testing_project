import pytest
import json
import warnings
import arnio as ar
from arnio.schema import Schema, Field


class TestBootstrapFromReport:
    """Test Schema.bootstrap_from_report"""

    def test_bootstrap_from_valid_report(self):
        """Test creating schema from a valid DataQualityReport"""
        frame = ar.from_dict({"name": ["Alice", "Bob"], "age": [25, 30]})
        report = ar.profile(frame)
        
        schema = Schema.bootstrap_from_report(report)
        
        assert isinstance(schema, Schema)
        assert "name" in schema.fields
        assert "age" in schema.fields

    def test_bootstrap_detects_nullable_from_nulls(self):
        """Test that nullable is detected from null_count in report"""
        frame = ar.from_dict({
            "complete": [1, 2, 3],
            "with_nulls": [1, None, 3]
        })
        report = ar.profile(frame)
        schema = Schema.bootstrap_from_report(report)
        
        assert schema.fields["with_nulls"].nullable

    def test_bootstrap_raises_type_error_on_non_report(self):
        """Test TypeError when input is not a DataQualityReport"""
        with pytest.raises(TypeError) as exc:
            Schema.bootstrap_from_report("not a report")
        assert "DataQualityReport" in str(exc.value)


class TestFromJson:
    """Test Schema.from_json"""

    def test_from_json_deserializes_valid_schema(self):
        """Test deserializing a valid schema from JSON"""
        json_str = json.dumps({
            "fields": {
                "id": {"dtype": "integer", "nullable": False},
                "name": {"dtype": "string", "nullable": False}
            },
            "strict": False,
            "unique": None
        })
        
        schema = Schema.from_json(json_str)
        
        assert isinstance(schema, Schema)
        assert "id" in schema.fields
        assert "name" in schema.fields
        assert schema.strict is False

    def test_from_json_with_unique_constraint(self):
        """Test deserializing schema with unique constraint"""
        json_str = json.dumps({
            "fields": {"email": {"dtype": "string", "nullable": False}},
            "strict": True,
            "unique": ["email"]
        })
        
        schema = Schema.from_json(json_str)
        
        assert schema.unique == ["email"]
        assert schema.strict is True

    def test_from_json_raises_on_invalid_json(self):
        """Test ValueError on malformed JSON"""
        with pytest.raises(ValueError) as exc:
            Schema.from_json("{invalid json")
        assert "Invalid schema JSON" in str(exc.value)

    def test_from_json_raises_on_non_dict_payload(self):
        """Test TypeError when JSON decodes to non-dict"""
        with pytest.raises(TypeError) as exc:
            Schema.from_json('["array"]')
        assert "object" in str(exc.value).lower()

    def test_from_json_raises_on_unknown_keys(self):
        """Test ValueError when schema has unknown keys"""
        json_str = json.dumps({
            "fields": {},
            "strict": False,
            "bad_key": "value"
        })
        
        with pytest.raises(ValueError) as exc:
            Schema.from_json(json_str)
        assert "unknown key" in str(exc.value).lower()

    def test_from_json_raises_on_invalid_types(self):
        """Test TypeError for invalid field types"""
        # strict not boolean
        with pytest.raises(TypeError) as exc:
            Schema.from_json(json.dumps({"fields": {}, "strict": "not bool"}))
        assert "strict" in str(exc.value).lower()
        
        # unique not list
        with pytest.raises(TypeError) as exc:
            Schema.from_json(json.dumps({"fields": {}, "strict": False, "unique": "not list"}))
        assert "unique" in str(exc.value).lower()

    def test_from_json_ignores_rules_omitted_marker(self):
        """Test that rules_omitted marker is silently ignored"""
        json_str = json.dumps({
            "fields": {"id": {"dtype": "integer", "nullable": False}},
            "strict": False,
            "rules_omitted": True
        })
        
        schema = Schema.from_json(json_str)
        assert isinstance(schema, Schema)


class TestToJson:
    """Test Schema.to_json"""

    def test_to_json_produces_valid_json(self):
        """Test that to_json returns valid JSON"""
        schema = Schema(
            fields={
                "name": Field(dtype="string", nullable=False),
                "age": Field(dtype="integer", nullable=True)
            },
            strict=False
        )
        
        json_str = schema.to_json()
        
        assert isinstance(json_str, str)
        payload = json.loads(json_str)
        assert "fields" in payload
        assert "strict" in payload

    def test_to_json_preserves_schema_attributes(self):
        """Test that schema attributes are preserved in JSON"""
        schema = Schema(
            fields={"email": Field(dtype="string", nullable=False)},
            strict=True,
            unique=["email"]
        )
        
        payload = json.loads(schema.to_json())
        
        assert payload["strict"] is True
        assert payload["unique"] == ["email"]

    def test_to_json_emits_warning_when_rules_present(self):
        """Test that UserWarning is emitted when rules exist"""
        schema = Schema(
            fields={"id": Field(dtype="integer", nullable=False)},
            strict=False
        )
        schema.rules = [lambda x: x]
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            json_str = schema.to_json()
            
            assert any(issubclass(warning.category, UserWarning) for warning in w)
        
        payload = json.loads(json_str)
        assert payload.get("rules_omitted") is True

    def test_to_json_roundtrip(self):
        """Test schema survives JSON roundtrip"""
        original = Schema(
            fields={
                "name": Field(dtype="string", nullable=False),
                "age": Field(dtype="integer", nullable=True),
            },
            strict=True,
            unique=["name"]
        )
        
        json_str = original.to_json()
        restored = Schema.from_json(json_str)
        
        assert set(restored.fields.keys()) == set(original.fields.keys())
        assert restored.strict == original.strict
        assert restored.unique == original.unique


class TestValidate:
    """Test Schema.validate"""

    def test_validate_returns_result(self):
        """Test that validate returns a ValidationResult"""
        frame = ar.from_dict({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"]
        })
        
        schema = Schema(
            fields={
                "id": Field(dtype="integer", nullable=False),
                "name": Field(dtype="string", nullable=False)
            }
        )
        
        result = schema.validate(frame)
        assert result is not None

    def test_validate_with_max_errors_parameter(self):
        """Test that validate accepts max_errors parameter"""
        frame = ar.from_dict({"x": [1, 2, 3]})
        schema = Schema(
            fields={"x": Field(dtype="integer", nullable=False)}
        )
        
        result = schema.validate(frame, max_errors=10)
        assert result is not None
        
        result = schema.validate(frame, max_errors=None)
        assert result is not None


class TestSchemaIntegration:
    """Integration tests combining schema operations"""

    def test_profile_bootstrap_validate_workflow(self):
        """Test complete workflow: profile -> bootstrap -> validate"""
        frame = ar.from_dict({
            "id": [1, 2, 3],
            "name": ["A", "B", "C"],
            "score": [10.5, 20.3, 30.1]
        })
        
        report = ar.profile(frame)
        schema = Schema.bootstrap_from_report(report)
        result = schema.validate(frame)
        
        assert isinstance(schema, Schema)
        assert result is not None

    def test_schema_json_roundtrip_with_validation(self):
        """Test creating, serializing, deserializing, and validating"""
        original_schema = Schema(
            fields={
                "user_id": Field(dtype="integer", nullable=False),
                "email": Field(dtype="string", nullable=False),
                "score": Field(dtype="float", nullable=True)
            },
            strict=True
        )
        
        json_str = original_schema.to_json()
        restored_schema = Schema.from_json(json_str)
        
        frame = ar.from_dict({
            "user_id": [1, 2, 3],
            "email": ["a@b.com", "c@d.com", "e@f.com"],
            "score": [10.5, None, 30.1]
        })
        
        result = restored_schema.validate(frame)
        assert result is not None
