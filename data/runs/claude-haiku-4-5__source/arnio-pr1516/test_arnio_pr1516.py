import pytest
import json
import warnings
import arnio as ar
import pandas as pd


class TestSchemaToJson:
    """Tests for Schema.to_json() serialization behavior"""

    def test_to_json_returns_string(self):
        """to_json() should return a JSON string"""
        schema = ar.schema.Schema(fields={})
        result = schema.to_json()
        assert isinstance(result, str)

    def test_to_json_valid_json_format(self):
        """Returned string should be valid JSON"""
        schema = ar.schema.Schema(fields={"age": {"type": "int32"}})
        result = schema.to_json()
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_to_json_contains_fields_key(self):
        """JSON payload should include fields mapping"""
        schema = ar.schema.Schema(fields={"name": {"type": "string"}})
        result = schema.to_json()
        parsed = json.loads(result)
        assert "fields" in parsed
        assert isinstance(parsed["fields"], dict)

    def test_to_json_contains_strict_flag(self):
        """JSON payload should include strict mode flag"""
        schema_strict = ar.schema.Schema(fields={}, strict=True)
        schema_permissive = ar.schema.Schema(fields={}, strict=False)
        
        assert json.loads(schema_strict.to_json())["strict"] is True
        assert json.loads(schema_permissive.to_json())["strict"] is False

    def test_to_json_contains_unique_constraint(self):
        """JSON payload should include unique constraints"""
        schema = ar.schema.Schema(fields={}, unique=["id"])
        parsed = json.loads(schema.to_json())
        assert "unique" in parsed
        assert parsed["unique"] == ["id"]

    def test_to_json_unique_null_when_none(self):
        """JSON unique should be null when not specified"""
        schema = ar.schema.Schema(fields={})
        parsed = json.loads(schema.to_json())
        assert parsed["unique"] is None

    def test_to_json_fields_sorted(self):
        """Fields should appear sorted by key in JSON"""
        schema = ar.schema.Schema(
            fields={"zebra": {}, "apple": {}, "mango": {}}
        )
        parsed = json.loads(schema.to_json())
        field_keys = list(parsed["fields"].keys())
        assert field_keys == sorted(field_keys)

    def test_to_json_no_warning_without_rules(self):
        """Should not emit warning when schema has no cross-field rules"""
        schema = ar.schema.Schema(fields={"col": {}})
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            schema.to_json()
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            assert len(user_warnings) == 0

    def test_to_json_warns_with_cross_field_rules(self):
        """Should emit UserWarning when cross-field rules are present"""
        schema = ar.schema.Schema(fields={})
        schema.rules = [lambda x: True]
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            schema.to_json()
            assert len(w) >= 1
            assert issubclass(w[0].category, UserWarning)
            assert "cross-field rules" in str(w[0].message).lower()

    def test_to_json_rules_omitted_flag_present_with_rules(self):
        """JSON should include rules_omitted:true when rules are present"""
        schema = ar.schema.Schema(fields={})
        schema.rules = [lambda x: True]
        
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = schema.to_json()
        
        parsed = json.loads(result)
        assert parsed.get("rules_omitted") is True

    def test_to_json_rules_omitted_flag_absent_without_rules(self):
        """JSON should not include rules_omitted flag when rules are absent"""
        schema = ar.schema.Schema(fields={})
        result = schema.to_json()
        parsed = json.loads(result)
        assert "rules_omitted" not in parsed or not parsed.get("rules_omitted")

    def test_to_json_stable_across_calls(self):
        """to_json() should produce stable output across multiple calls"""
        schema = ar.schema.Schema(fields={"x": {}, "y": {}}, strict=True)
        result1 = schema.to_json()
        result2 = schema.to_json()
        assert result1 == result2


class TestSchemaValidate:
    """Tests for Schema.validate() method"""

    def test_validate_accepts_frame(self):
        """validate() should accept an ArFrame as first argument"""
        schema = ar.schema.Schema(fields={})
        frame = ar.from_dict({"col": [1, 2]})
        result = schema.validate(frame)
        assert result is not None

    def test_validate_returns_validation_result(self):
        """validate() should return a ValidationResult object"""
        schema = ar.schema.Schema(fields={})
        frame = ar.from_dict({"col": [1]})
        result = schema.validate(frame)
        # ValidationResult should have validation interface
        assert hasattr(result, "is_valid") or hasattr(result, "errors")

    def test_validate_accepts_max_errors_parameter(self):
        """validate() should accept optional max_errors parameter"""
        schema = ar.schema.Schema(fields={})
        frame = ar.from_dict({"col": [1]})
        # Should not raise with max_errors parameter
        result = schema.validate(frame, max_errors=10)
        assert result is not None

    def test_validate_max_errors_none(self):
        """validate() should work with max_errors=None (no limit)"""
        schema = ar.schema.Schema(fields={})
        frame = ar.from_dict({"col": [1, 2, 3]})
        result = schema.validate(frame, max_errors=None)
        assert result is not None

    def test_validate_empty_frame(self):
        """validate() should handle empty frames"""
        schema = ar.schema.Schema(fields={"col": {}})
        frame = ar.from_dict({"col": []})
        result = schema.validate(frame)
        assert result is not None
