import pytest
import json
import warnings
import pandas as pd
import arnio as ar


class TestSchemaToJson:
    """Tests for Schema.to_json() method."""
    
    def test_to_json_returns_string(self):
        """Test that to_json returns a string."""
        schema = ar.Schema({"col": ar.String()})
        result = schema.to_json()
        assert isinstance(result, str)
    
    def test_to_json_valid_json_format(self):
        """Test that to_json produces valid, parseable JSON."""
        schema = ar.Schema({"col": ar.String()})
        json_str = schema.to_json()
        payload = json.loads(json_str)
        assert isinstance(payload, dict)
    
    def test_to_json_contains_required_keys(self):
        """Test that JSON payload contains required keys."""
        schema = ar.Schema({"name": ar.String()})
        payload = json.loads(schema.to_json())
        assert "fields" in payload
        assert "strict" in payload
        assert "unique" in payload
    
    def test_to_json_fields_sorted_alphabetically(self):
        """Test that fields are sorted alphabetically in JSON."""
        schema = ar.Schema({
            "zebra": ar.String(),
            "apple": ar.String(),
            "middle": ar.String(),
        })
        payload = json.loads(schema.to_json())
        field_keys = list(payload["fields"].keys())
        assert field_keys == ["apple", "middle", "zebra"]
    
    def test_to_json_preserves_strict_setting(self):
        """Test that strict mode setting is preserved."""
        schema_strict = ar.Schema({"col": ar.String()}, strict=True)
        schema_lenient = ar.Schema({"col": ar.String()}, strict=False)
        
        assert json.loads(schema_strict.to_json())["strict"] is True
        assert json.loads(schema_lenient.to_json())["strict"] is False
    
    def test_to_json_unique_as_list(self):
        """Test that unique constraint is serialized as list."""
        schema = ar.Schema(
            {"id": ar.Integer(), "code": ar.String()},
            unique={"id", "code"}
        )
        payload = json.loads(schema.to_json())
        assert isinstance(payload["unique"], list)
        assert set(payload["unique"]) == {"id", "code"}
    
    def test_to_json_unique_null_when_none(self):
        """Test that unique is null when None."""
        schema = ar.Schema({"col": ar.String()}, unique=None)
        payload = json.loads(schema.to_json())
        assert payload["unique"] is None
    
    def test_to_json_with_rules_emits_warning(self):
        """Test that UserWarning is emitted when rules are present."""
        schema = ar.Schema({"a": ar.Integer(), "b": ar.Integer()})
        schema.rules.append(lambda df: df["a"] < df["b"])
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            schema.to_json()
            
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            assert len(user_warnings) > 0
            assert any("cross-field rules" in str(uw.message) for uw in user_warnings)
    
    def test_to_json_with_rules_sets_omitted_flag(self):
        """Test that rules_omitted flag is true when rules present."""
        schema = ar.Schema({"a": ar.Integer()})
        schema.rules.append(lambda df: True)
        
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            payload = json.loads(schema.to_json())
        
        assert payload.get("rules_omitted") is True
    
    def test_to_json_without_rules_no_omitted_flag(self):
        """Test that rules_omitted flag is absent when no rules."""
        schema = ar.Schema({"col": ar.String()})
        payload = json.loads(schema.to_json())
        assert "rules_omitted" not in payload
    
    def test_to_json_roundtrip_without_rules(self):
        """Test schema can be restored from JSON (without rules)."""
        original = ar.Schema(
            {"name": ar.String(), "age": ar.Integer()},
            strict=True,
            unique={"name"}
        )
        json_str = original.to_json()
        restored = ar.Schema.from_json(json_str)
        
        assert restored.strict == original.strict
        assert set(restored.fields.keys()) == set(original.fields.keys())


class TestSchemaValidate:
    """Tests for Schema.validate() method."""
    
    def test_validate_returns_validation_result(self):
        """Test that validate returns a ValidationResult object."""
        schema = ar.Schema({"col": ar.String()})
        df = pd.DataFrame({"col": ["value"]})
        frame = ar.ArFrame(df)
        
        result = schema.validate(frame)
        assert hasattr(result, "is_valid")
        assert hasattr(result, "issues")
    
    def test_validate_valid_data(self):
        """Test validation succeeds with valid data."""
        schema = ar.Schema({"name": ar.String(), "age": ar.Integer()})
        df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
        frame = ar.ArFrame(df)
        
        result = schema.validate(frame)
        assert result.is_valid is True
    
    def test_validate_missing_column(self):
        """Test validation fails when required column is missing."""
        schema = ar.Schema({"name": ar.String(), "age": ar.Integer()})
        df = pd.DataFrame({"name": ["Alice"]})
        frame = ar.ArFrame(df)
        
        result = schema.validate(frame)
        assert result.is_valid is False
    
    def test_validate_type_mismatch(self):
        """Test validation detects type mismatches."""
        schema = ar.Schema({"age": ar.Integer()})
        df = pd.DataFrame({"age": ["not_an_integer"]})
        frame = ar.ArFrame(df)
        
        result = schema.validate(frame)
        assert result.is_valid is False
    
    def test_validate_max_errors_limits_issues(self):
        """Test max_errors parameter limits reported issues."""
        schema = ar.Schema({"col1": ar.Integer(), "col2": ar.Integer()})
        df = pd.DataFrame({"col1": ["bad"], "col2": ["bad"]})
        frame = ar.ArFrame(df)
        
        result_limited = schema.validate(frame, max_errors=1)
        result_unlimited = schema.validate(frame, max_errors=None)
        
        assert len(result_limited.issues) <= 1
        assert len(result_unlimited.issues) >= len(result_limited.issues)
