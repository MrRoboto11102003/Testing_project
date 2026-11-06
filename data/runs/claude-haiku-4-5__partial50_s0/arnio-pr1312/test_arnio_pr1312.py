import pytest
import pandas as pd
import arnio as ar
import json


class TestSchemaBootstrapFromReport:
    """Test Schema.bootstrap_from_report() classmethod"""
    
    def test_bootstrap_from_report_basic(self):
        """Test basic bootstrap from a DataQualityReport"""
        frame = ar.from_pandas(pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
        }))
        report = ar.profile(frame)
        schema = ar.Schema.bootstrap_from_report(report)
        
        assert isinstance(schema, ar.Schema)
    
    def test_bootstrap_from_report_invalid_input_raises_typeerror(self):
        """Test bootstrap raises TypeError for non-DataQualityReport input"""
        with pytest.raises(TypeError):
            ar.Schema.bootstrap_from_report("not a report")
        
        with pytest.raises(TypeError):
            ar.Schema.bootstrap_from_report({})
    
    def test_bootstrap_from_report_infers_column_types(self):
        """Test bootstrap correctly infers types from report"""
        frame = ar.from_pandas(pd.DataFrame({
            "int_col": [1, 2, 3],
            "float_col": [1.1, 2.2, 3.3],
            "str_col": ["a", "b", "c"],
        }))
        report = ar.profile(frame)
        schema = ar.Schema.bootstrap_from_report(report)
        
        result = schema.validate(frame)
        assert isinstance(result, ar.ValidationResult)


class TestSchemaJsonRoundTrip:
    """Test Schema.to_json() and Schema.from_json()"""
    
    def test_to_json_produces_valid_json_string(self):
        """Test to_json returns valid JSON that can be parsed"""
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "name": ar.String(),
        })
        json_str = schema.to_json()
        
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
    
    def test_from_json_deserializes_schema(self):
        """Test from_json returns a Schema instance"""
        original_schema = ar.Schema({
            "id": ar.Int64(nullable=False),
        })
        json_str = original_schema.to_json()
        
        restored = ar.Schema.from_json(json_str)
        assert isinstance(restored, ar.Schema)
    
    def test_roundtrip_with_simple_types(self):
        """Test schema survives to_json and from_json roundtrip"""
        original = ar.Schema({
            "id": ar.Int64(nullable=False),
            "email": ar.Email(nullable=False),
            "active": ar.Bool(),
        })
        
        json_str = original.to_json()
        restored = ar.Schema.from_json(json_str)
        
        frame = ar.from_pandas(pd.DataFrame({
            "id": [1, 2, 3],
            "email": ["a@example.com", "b@example.com", "c@example.com"],
            "active": [True, False, True],
        }))
        
        result_original = original.validate(frame)
        result_restored = restored.validate(frame)
        
        assert result_original.passed == result_restored.passed
    
    def test_roundtrip_preserves_constraints(self):
        """Test numeric and string constraints survive roundtrip"""
        original = ar.Schema({
            "age": ar.Int64(min=0, max=150),
            "username": ar.String(min_length=3, max_length=20),
        })
        
        json_str = original.to_json()
        restored = ar.Schema.from_json(json_str)
        
        # Validate that constraints are preserved by testing violation
        frame_invalid = ar.from_pandas(pd.DataFrame({
            "age": [200],
            "username": ["ab"],
        }))
        
        result_original = original.validate(frame_invalid)
        result_restored = restored.validate(frame_invalid)
        
        assert result_original.passed == result_restored.passed
    
    def test_from_json_invalid_json_raises_error(self):
        """Test from_json raises exception for malformed JSON"""
        with pytest.raises(Exception):
            ar.Schema.from_json("invalid json {")


class TestSchemaValidate:
    """Test Schema.validate() instance method"""
    
    def test_validate_passing_frame_returns_passed(self):
        """Test validate returns passed=True for valid data"""
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
    
    def test_validate_dtype_violation(self):
        """Test validate detects dtype mismatches"""
        schema = ar.Schema({
            "id": ar.Int64(),
        })
        frame = ar.from_pandas(pd.DataFrame({
            "id": ["string", "values", "here"],
        }))
        
        result = schema.validate(frame)
        assert result.passed is False
        assert result.issue_count > 0
    
    def test_validate_nullable_false_violation(self):
        """Test validate enforces nullable=False constraint"""
        schema = ar.Schema({
            "required_col": ar.String(nullable=False),
        })
        frame = ar.from_pandas(pd.DataFrame({
            "required_col": ["value", None, "value"],
        }))
        
        result = schema.validate(frame)
        assert result.passed is False
    
    def test_validate_numeric_min_max_violation(self):
        """Test validate enforces min/max constraints"""
        schema = ar.Schema({
            "age": ar.Int64(min=0, max=120),
        })
        frame = ar.from_pandas(pd.DataFrame({
            "age": [25, -5, 200],
        }))
        
        result = schema.validate(frame)
        assert result.passed is False
    
    def test_validate_string_length_constraints(self):
        """Test validate enforces min_length and max_length"""
        schema = ar.Schema({
            "code": ar.String(min_length=3, max_length=5),
        })
        frame = ar.from_pandas(pd.DataFrame({
            "code": ["ABC", "ABCDE", "AB"],
        }))
        
        result = schema.validate(frame)
        assert result.passed is False
    
    def test_validate_returns_validation_result_object(self):
        """Test validate returns proper ValidationResult with required attributes"""
        schema = ar.Schema({"val": ar.Int64()})
        frame = ar.from_pandas(pd.DataFrame({"val": [1, 2, 3]}))
        
        result = schema.validate(frame)
        
        assert isinstance(result, ar.ValidationResult)
        assert hasattr(result, "passed")
        assert hasattr(result, "issue_count")
        assert hasattr(result, "issues")
    
    def test_validate_with_max_errors_parameter(self):
        """Test validate respects max_errors limit"""
        schema = ar.Schema({
            "id": ar.Int64(min=0),
        })
        frame = ar.from_pandas(pd.DataFrame({
            "id": [-1, -2, -3, -4, -5],
        }))
        
        result_unlimited = schema.validate(frame)
        result_limited = schema.validate(frame, max_errors=2)
        
        # Limited result should not exceed max_errors
        assert result_limited.issue_count <= 2
        # Unlimited should have more issues than limited
        assert result_unlimited.issue_count >= result_limited.issue_count
