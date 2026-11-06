import pytest
import warnings
import json
import pandas as pd
import numpy as np
import arnio as ar


class TestSchemaToJson:
    """Tests for Schema.to_json() serialization"""
    
    def test_to_json_produces_valid_json(self):
        """Basic schema serializes to valid JSON string"""
        schema = ar.Schema({
            "id": ar.Int64(),
            "name": ar.String(),
        })
        json_str = schema.to_json()
        data = json.loads(json_str)
        
        assert isinstance(data, dict)
        assert "fields" in data
        
    def test_to_json_includes_all_fields(self):
        """All fields are present in serialized JSON"""
        schema = ar.Schema({
            "id": ar.Int64(),
            "email": ar.Email(),
            "active": ar.Bool(),
        })
        data = json.loads(schema.to_json())
        
        assert "id" in data["fields"]
        assert "email" in data["fields"]
        assert "active" in data["fields"]
        
    def test_to_json_with_cross_field_rules_emits_warning(self):
        """Cross-field rules trigger UserWarning"""
        def cross_field_rule(df):
            return []
        
        schema = ar.Schema(
            {"col": ar.String()},
            rules=[cross_field_rule]
        )
        
        with pytest.warns(UserWarning, match="cross-field rules cannot be serialized"):
            schema.to_json()
    
    def test_to_json_sets_rules_omitted_flag_when_rules_present(self):
        """rules_omitted flag is true when rules exist"""
        def dummy_rule(df):
            return []
        
        schema_with_rules = ar.Schema(
            {"col": ar.String()},
            rules=[dummy_rule]
        )
        schema_no_rules = ar.Schema({"col": ar.String()})
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data_with = json.loads(schema_with_rules.to_json())
            data_without = json.loads(schema_no_rules.to_json())
        
        assert data_with.get("rules_omitted") is True
        assert "rules_omitted" not in data_without or data_without.get("rules_omitted") is not True
        
    def test_to_json_serializes_strict_flag(self):
        """strict flag is correctly serialized"""
        schema_strict = ar.Schema({"id": ar.Int64()}, strict=True)
        schema_lenient = ar.Schema({"id": ar.Int64()}, strict=False)
        
        strict_data = json.loads(schema_strict.to_json())
        lenient_data = json.loads(schema_lenient.to_json())
        
        assert strict_data["strict"] is True
        assert lenient_data["strict"] is False
        
    def test_to_json_serializes_unique_constraint(self):
        """unique constraint is serialized as list or null"""
        schema_with_unique = ar.Schema(
            {"id": ar.Int64(), "email": ar.Email()},
            unique=["id"]
        )
        schema_no_unique = ar.Schema({"id": ar.Int64()}, unique=None)
        
        data_with = json.loads(schema_with_unique.to_json())
        data_without = json.loads(schema_no_unique.to_json())
        
        assert isinstance(data_with["unique"], list)
        assert "id" in data_with["unique"]
        assert data_without["unique"] is None
        
    def test_to_json_fields_alphabetically_sorted(self):
        """Field names are sorted alphabetically in output"""
        schema = ar.Schema({
            "zebra": ar.String(),
            "apple": ar.String(),
            "mango": ar.String(),
        })
        data = json.loads(schema.to_json())
        field_keys = list(data["fields"].keys())
        
        assert field_keys == ["apple", "mango", "zebra"]
        
    def test_to_json_roundtrip_via_from_json(self):
        """Schema without rules can roundtrip through JSON"""
        original = ar.Schema({
            "id": ar.Int64(nullable=False),
            "name": ar.String(nullable=True),
        }, strict=True, unique=["id"])
        
        json_str = original.to_json()
        restored = ar.Schema.from_json(json_str)
        json_str_2 = restored.to_json()
        
        # Roundtripped JSON should be identical
        assert json.loads(json_str) == json.loads(json_str_2)


class TestSchemaValidate:
    """Tests for Schema.validate() validation method"""
    
    def test_validate_passes_for_valid_data(self):
        """Valid data passes validation"""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["alice", "bob", "charlie"],
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "id": ar.Int64(),
            "name": ar.String(),
        })
        
        result = schema.validate(frame)
        assert result.passed is True
        assert len(result.issues) == 0
        
    def test_validate_fails_for_dtype_mismatch(self):
        """Invalid data types fail validation"""
        df = pd.DataFrame({
            "age": ["not_a_number", "also_text"],
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({"age": ar.Int64()})
        
        result = schema.validate(frame)
        assert result.passed is False
        assert len(result.issues) > 0
        
    def test_validate_fails_null_when_nullable_false(self):
        """Null values fail when nullable=False"""
        df = pd.DataFrame({
            "required_field": [1, None, 3],
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "required_field": ar.Int64(nullable=False),
        })
        
        result = schema.validate(frame)
        assert result.passed is False
        assert any(issue for issue in result.issues)
        
    def test_validate_respects_min_max_constraints(self):
        """min and max constraints are enforced"""
        df = pd.DataFrame({
            "score": [5, 50, 150],  # 5 below min=20, 150 above max=100
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "score": ar.Int64(min=20, max=100),
        })
        
        result = schema.validate(frame)
        assert result.passed is False
        assert len(result.issues) >= 2
        
    def test_validate_max_errors_parameter(self):
        """max_errors parameter limits returned issues"""
        df = pd.DataFrame({
            "id": ["x", "y", "z", "w", "v"],
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({"id": ar.Int64()})
        
        result_unlimited = schema.validate(frame, max_errors=None)
        result_limited = schema.validate(frame, max_errors=2)
        
        assert len(result_unlimited.issues) > 2
        assert len(result_limited.issues) <= 2
        
    def test_validate_empty_frame(self):
        """Empty frame passes validation (no violations)"""
        df = pd.DataFrame({"id": pd.Series([], dtype="int64")})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"id": ar.Int64(nullable=False)})
        
        result = schema.validate(frame)
        assert result.passed is True
        
    def test_validate_returns_validation_result(self):
        """validate() returns ValidationResult with expected attributes"""
        df = pd.DataFrame({"id": [1, 2, 3]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"id": ar.Int64()})
        
        result = schema.validate(frame)
        
        assert hasattr(result, "passed")
        assert hasattr(result, "issues")
        assert isinstance(result.passed, bool)
        assert isinstance(result.issues, list)
