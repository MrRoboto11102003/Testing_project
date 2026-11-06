import pytest
import json
import warnings
import pandas as pd
import numpy as np
import arnio as ar


class TestBootstrapFromReport:
    """Tests for Schema.bootstrap_from_report()"""
    
    def test_bootstrap_creates_schema_from_report(self):
        """bootstrap_from_report creates schema with inferred fields."""
        frame = ar.from_pandas(pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"]
        }))
        report = ar.profile(frame)
        schema = ar.Schema.bootstrap_from_report(report)
        
        assert "id" in schema.fields
        assert "name" in schema.fields
    
    def test_bootstrap_detects_nullable_from_nulls(self):
        """bootstrap_from_report sets nullable=True when null_count > 0."""
        frame = ar.from_pandas(pd.DataFrame({
            "value": [1.0, None, 3.0]
        }))
        report = ar.profile(frame)
        schema = ar.Schema.bootstrap_from_report(report)
        
        assert schema.fields["value"].nullable is True
    
    def test_bootstrap_no_nulls_not_nullable(self):
        """bootstrap_from_report sets nullable=False when null_count == 0."""
        frame = ar.from_pandas(pd.DataFrame({
            "value": [1, 2, 3]
        }))
        report = ar.profile(frame)
        schema = ar.Schema.bootstrap_from_report(report)
        
        assert schema.fields["value"].nullable is False
    
    def test_bootstrap_type_error_not_report(self):
        """bootstrap_from_report raises TypeError for non-DataQualityReport input."""
        with pytest.raises(TypeError):
            ar.Schema.bootstrap_from_report("not a report")
    
    def test_bootstrap_value_error_empty_report(self):
        """bootstrap_from_report raises ValueError for empty report."""
        frame = ar.from_pandas(pd.DataFrame())
        report = ar.profile(frame)
        
        with pytest.raises(ValueError, match="no columns"):
            ar.Schema.bootstrap_from_report(report)


class TestToJsonFromJsonRoundTrip:
    """Tests for Schema.to_json() and Schema.from_json()"""
    
    def test_to_json_produces_valid_json(self):
        """to_json() returns valid JSON string."""
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "name": ar.String()
        })
        json_str = schema.to_json()
        
        assert isinstance(json_str, str)
        payload = json.loads(json_str)
        assert "fields" in payload
        assert "strict" in payload
        assert "unique" in payload
    
    def test_from_json_deserializes_basic_schema(self):
        """from_json() deserializes a valid schema JSON."""
        json_str = '''{
            "fields": {"id": {"dtype": "int64", "nullable": false}},
            "strict": false,
            "unique": null
        }'''
        schema = ar.Schema.from_json(json_str)
        
        assert "id" in schema.fields
        assert schema.fields["id"].nullable is False
    
    def test_from_json_invalid_json_raises(self):
        """from_json() raises ValueError for malformed JSON."""
        with pytest.raises(ValueError, match="Invalid schema JSON"):
            ar.Schema.from_json("{invalid json")
    
    def test_from_json_non_dict_raises_type_error(self):
        """from_json() raises TypeError if JSON is not an object."""
        with pytest.raises(TypeError, match="must decode to an object"):
            ar.Schema.from_json('[]')
    
    def test_from_json_unknown_keys_raises(self):
        """from_json() raises ValueError for unknown keys."""
        json_str = '''{
            "fields": {},
            "strict": false,
            "unknown_key": true
        }'''
        with pytest.raises(ValueError, match="unknown key"):
            ar.Schema.from_json(json_str)
    
    def test_round_trip_preserves_schema(self):
        """Schema survives to_json() and from_json() round trip."""
        original = ar.Schema({
            "id": ar.Int64(nullable=False, unique=True),
            "email": ar.String(),
            "score": ar.Float64()
        }, strict=True, unique=["id"])
        
        json_str = original.to_json()
        restored = ar.Schema.from_json(json_str)
        
        assert original.strict == restored.strict
        assert original.unique == restored.unique
        assert set(original.fields.keys()) == set(restored.fields.keys())
    
    def test_to_json_warns_when_rules_present(self):
        """to_json() emits UserWarning if custom rules exist."""
        ar.register_validator("custom_check", lambda v: v > 0)
        schema = ar.Schema(
            {"value": ar.Custom("custom_check")},
            rules=[("value", lambda x: x > 0)]
        )
        
        with pytest.warns(UserWarning, match="rules cannot be serialized"):
            json_str = schema.to_json()
        
        payload = json.loads(json_str)
        assert payload.get("rules_omitted") is True
    
    def test_from_json_silently_accepts_rules_omitted(self):
        """from_json() silently ignores rules_omitted marker."""
        json_str = '''{
            "fields": {"id": {"dtype": "int64"}},
            "strict": false,
            "unique": null,
            "rules_omitted": true
        }'''
        schema = ar.Schema.from_json(json_str)
        
        assert "id" in schema.fields


class TestValidate:
    """Tests for Schema.validate()"""
    
    def test_validate_passes_for_valid_data(self):
        """validate() returns passed=True for valid data."""
        frame = ar.from_pandas(pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"]
        }))
        schema = ar.Schema({
            "id": ar.Int64(),
            "name": ar.String()
        })
        
        result = schema.validate(frame)
        assert result.passed is True
    
    def test_validate_fails_for_invalid_data(self):
        """validate() returns passed=False for invalid data."""
        frame = ar.from_pandas(pd.DataFrame({
            "email": ["not-email", "also-bad"]
        }))
        schema = ar.Schema({
            "email": ar.Email(nullable=False)
        })
        
        result = schema.validate(frame)
        assert result.passed is False
        assert len(result.issues) > 0
    
    def test_validate_respects_max_errors(self):
        """validate() stops after max_errors violations."""
        frame = ar.from_pandas(pd.DataFrame({
            "value": [-1, -2, -3, -4, -5]
        }))
        schema = ar.Schema({
            "value": ar.Int64(min=0)
        })
        
        result = schema.validate(frame, max_errors=2)
        assert len(result.issues) <= 2
    
    def test_validate_bootstrap_consistency(self):
        """Schema bootstrapped from data validates that data successfully."""
        frame = ar.from_pandas(pd.DataFrame({
            "id": [1, 2, 3],
            "score": [95.5, None, 87.3]
        }))
        report = ar.profile(frame)
        schema = ar.Schema.bootstrap_from_report(report)
        
        result = schema.validate(frame)
        assert result.passed is True
