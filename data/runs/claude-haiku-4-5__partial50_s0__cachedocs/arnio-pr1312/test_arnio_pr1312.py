import pytest
import pandas as pd
import arnio as ar
import json


class TestSchemaBootstrapFromReport:
    """Tests for Schema.bootstrap_from_report()"""
    
    def test_bootstrap_from_valid_report(self):
        """Create schema from a valid DataQualityReport"""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"]
        })
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        schema = ar.Schema.bootstrap_from_report(report)
        
        assert isinstance(schema, ar.Schema)
        assert len(schema.fields) > 0
    
    def test_bootstrap_invalid_type_raises_typeerror(self):
        """Raises TypeError if input is not a DataQualityReport"""
        with pytest.raises(TypeError):
            ar.Schema.bootstrap_from_report("not a report")
        
        with pytest.raises(TypeError):
            ar.Schema.bootstrap_from_report({})
    
    def test_bootstrap_empty_report_raises_valueerror(self):
        """Raises ValueError if report has no columns"""
        df = pd.DataFrame()
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        with pytest.raises(ValueError):
            ar.Schema.bootstrap_from_report(report)


class TestSchemaJsonRoundtrip:
    """Tests for Schema.to_json() and Schema.from_json()"""
    
    def test_to_json_returns_string(self):
        """to_json() returns a valid JSON string"""
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "name": ar.String(nullable=True),
        })
        
        result = schema.to_json()
        
        assert isinstance(result, str)
        assert len(result) > 0
        json.loads(result)  # Verify it's valid JSON
    
    def test_from_json_deserializes_schema(self):
        """from_json() creates a Schema from JSON string"""
        original = ar.Schema({
            "id": ar.Int64(nullable=False),
        })
        json_str = original.to_json()
        
        restored = ar.Schema.from_json(json_str)
        
        assert isinstance(restored, ar.Schema)
    
    def test_roundtrip_simple_schema(self):
        """Schema survives to_json/from_json roundtrip"""
        original = ar.Schema({
            "id": ar.String(nullable=False),
            "created_at": ar.DateTime(format="%Y-%m-%d"),
        })
        
        json_str = original.to_json()
        restored = ar.Schema.from_json(json_str)
        
        df = pd.DataFrame({
            "id": ["a", "b"],
            "created_at": ["2025-01-01", "2025-01-02"]
        })
        frame = ar.from_pandas(df)
        
        result1 = original.validate(frame)
        result2 = restored.validate(frame)
        assert result1.passed == result2.passed
    
    def test_roundtrip_preserves_constraints(self):
        """Roundtrip preserves schema constraints"""
        original = ar.Schema(
            {
                "id": ar.Int64(nullable=False, unique=True),
                "email": ar.Email(nullable=False),
                "username": ar.String(min_length=3, max_length=20),
            },
            strict=True,
            unique=["id"],
        )
        
        json_str = original.to_json()
        restored = ar.Schema.from_json(json_str)
        
        assert isinstance(restored, ar.Schema)


class TestSchemaValidate:
    """Tests for Schema.validate()"""
    
    def test_validate_passing_frame(self):
        """validate() returns passed result for valid data"""
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "name": ar.String(nullable=False),
        })
        
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"]
        })
        frame = ar.from_pandas(df)
        
        result = schema.validate(frame)
        
        assert result.passed is True
        assert result.issue_count == 0
    
    def test_validate_failing_dtype(self):
        """validate() detects dtype violations"""
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
        })
        
        df = pd.DataFrame({
            "id": ["not_numeric"]
        })
        frame = ar.from_pandas(df)
        
        result = schema.validate(frame)
        
        assert result.passed is False
        assert result.issue_count > 0
    
    def test_validate_failing_nullable_constraint(self):
        """validate() detects null value violations when nullable=False"""
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
        })
        
        df = pd.DataFrame({
            "id": [1, None, 3]
        })
        frame = ar.from_pandas(df)
        
        result = schema.validate(frame)
        
        assert result.passed is False
        assert result.issue_count > 0
    
    def test_validate_max_errors_parameter(self):
        """validate() respects max_errors limit"""
        schema = ar.Schema({
            "score": ar.Int64(min=0, max=100),
        })
        
        df = pd.DataFrame({
            "score": [-1, -2, -3, -4, -5]
        })
        frame = ar.from_pandas(df)
        
        result = schema.validate(frame, max_errors=2)
        
        assert not result.passed
        assert result.issue_count <= 2
    
    def test_validate_returns_validation_result(self):
        """validate() returns object with required attributes"""
        schema = ar.Schema({
            "id": ar.Int64(),
        })
        df = pd.DataFrame({"id": [1, 2, 3]})
        frame = ar.from_pandas(df)
        
        result = schema.validate(frame)
        
        assert hasattr(result, 'passed')
        assert hasattr(result, 'issue_count')
        assert hasattr(result, 'issues')
    
    def test_validate_multiple_violations(self):
        """validate() detects violations across multiple columns"""
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "age": ar.Int64(min=0, max=150),
        })
        
        df = pd.DataFrame({
            "id": [1, None, 3],
            "age": [25, 30, 999]
        })
        frame = ar.from_pandas(df)
        
        result = schema.validate(frame)
        
        assert not result.passed
        assert result.issue_count > 0
    
    def test_validate_nullable_true_allows_nulls(self):
        """validate() allows null values when nullable=True"""
        schema = ar.Schema({
            "required_id": ar.Int64(nullable=False),
            "optional_note": ar.String(nullable=True),
        })
        
        df = pd.DataFrame({
            "required_id": [1, 2, 3],
            "optional_note": ["note", None, "another"]
        })
        frame = ar.from_pandas(df)
        
        result = schema.validate(frame)
        
        assert result.passed is True
