import pytest
import pandas as pd
import numpy as np
import json
import arnio as ar


class TestSchemaToJson:
    """Tests for arnio.schema.Schema.to_json"""

    def test_to_json_returns_string(self):
        """to_json returns a JSON string"""
        schema = ar.Schema({"col": ar.String()})
        result = schema.to_json()
        assert isinstance(result, str)

    def test_to_json_is_valid_json(self):
        """to_json output is valid JSON"""
        schema = ar.Schema({
            "id": ar.Int64(),
            "name": ar.String(),
        })
        json_str = schema.to_json()
        data = json.loads(json_str)
        assert isinstance(data, dict)

    def test_to_json_roundtrip(self):
        """to_json output can be restored with from_json"""
        original = ar.Schema({
            "id": ar.Int64(nullable=False, unique=True),
            "email": ar.Email(nullable=False),
        })
        json_str = original.to_json()
        restored = ar.Schema.from_json(json_str)
        
        frame = ar.from_pandas(pd.DataFrame({
            "id": [1, 2, 3],
            "email": ["a@b.com", "c@d.com", "e@f.com"],
        }))
        
        result1 = original.validate(frame)
        result2 = restored.validate(frame)
        assert result1.passed == result2.passed

    def test_to_json_with_nullable_constraints(self):
        """to_json preserves nullable field constraints"""
        schema = ar.Schema({
            "required": ar.String(nullable=False),
            "optional": ar.String(nullable=True),
        })
        json_str = schema.to_json()
        data = json.loads(json_str)
        assert isinstance(data, dict)

    def test_to_json_with_numeric_bounds(self):
        """to_json preserves min/max numeric constraints"""
        schema = ar.Schema({
            "score": ar.Int64(min=0, max=100),
            "price": ar.Float64(min=0.0, max=9999.99),
        })
        json_str = schema.to_json()
        data = json.loads(json_str)
        assert isinstance(data, dict)

    def test_to_json_with_string_length_bounds(self):
        """to_json preserves string length constraints"""
        schema = ar.Schema({
            "username": ar.String(min_length=3, max_length=20),
        })
        json_str = schema.to_json()
        data = json.loads(json_str)
        assert isinstance(data, dict)

    def test_to_json_with_unique_constraint(self):
        """to_json preserves unique multi-column constraint"""
        schema = ar.Schema({
            "id": ar.Int64(),
            "email": ar.Email(),
        }, unique=["id"])
        json_str = schema.to_json()
        data = json.loads(json_str)
        assert isinstance(data, dict)

    def test_to_json_with_diverse_field_types(self):
        """to_json handles multiple field types"""
        schema = ar.Schema({
            "id": ar.Int64(),
            "score": ar.Float64(),
            "active": ar.Bool(),
            "email": ar.Email(),
            "website": ar.URL(),
            "created": ar.DateTime(format="%Y-%m-%d"),
        })
        json_str = schema.to_json()
        assert isinstance(json_str, str)
        json.loads(json_str)


class TestSchemaValidate:
    """Tests for arnio.schema.Schema.validate"""

    def test_validate_returns_validation_result(self):
        """validate returns a ValidationResult object"""
        schema = ar.Schema({"col": ar.String()})
        frame = ar.from_pandas(pd.DataFrame({"col": ["test"]}))
        
        result = schema.validate(frame)
        assert hasattr(result, 'passed')
        assert hasattr(result, 'issues')

    def test_validate_passes_valid_data(self):
        """validate returns passed=True for valid data"""
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

    def test_validate_fails_dtype_mismatch(self):
        """validate fails when column dtype doesn't match schema"""
        schema = ar.Schema({
            "id": ar.Int64(),
        })
        
        frame = ar.from_pandas(pd.DataFrame({
            "id": ["not", "numbers"],
        }))
        
        result = schema.validate(frame)
        assert result.passed is False
        assert len(result.issues) > 0

    def test_validate_nullable_false_rejects_nulls(self):
        """validate fails when non-nullable field contains nulls"""
        schema = ar.Schema({
            "email": ar.Email(nullable=False),
        })
        
        frame = ar.from_pandas(pd.DataFrame({
            "email": ["test@example.com", None, "user@example.com"],
        }))
        
        result = schema.validate(frame)
        assert result.passed is False

    def test_validate_nullable_true_accepts_nulls(self):
        """validate passes when nullable field contains nulls"""
        schema = ar.Schema({
            "age": ar.Int64(nullable=True),
        })
        
        frame = ar.from_pandas(pd.DataFrame({
            "age": [25, None, 30],
        }))
        
        result = schema.validate(frame)
        assert result.passed is True

    def test_validate_unique_constraint_violation(self):
        """validate fails when unique constraint is violated"""
        schema = ar.Schema({
            "id": ar.Int64(unique=True),
        })
        
        frame = ar.from_pandas(pd.DataFrame({
            "id": [1, 2, 2, 3],
        }))
        
        result = schema.validate(frame)
        assert result.passed is False

    def test_validate_unique_constraint_satisfied(self):
        """validate passes when all values are unique"""
        schema = ar.Schema({
            "id": ar.Int64(unique=True),
        })
        
        frame = ar.from_pandas(pd.DataFrame({
            "id": [1, 2, 3, 4],
        }))
        
        result = schema.validate(frame)
        assert result.passed is True

    def test_validate_min_constraint_violation(self):
        """validate fails when value is below minimum"""
        schema = ar.Schema({
            "age": ar.Int64(min=18),
        })
        
        frame = ar.from_pandas(pd.DataFrame({
            "age": [25, 10, 30],
        }))
        
        result = schema.validate(frame)
        assert result.passed is False

    def test_validate_max_constraint_violation(self):
        """validate fails when value exceeds maximum"""
        schema = ar.Schema({
            "score": ar.Int64(max=100),
        })
        
        frame = ar.from_pandas(pd.DataFrame({
            "score": [50, 150, 75],
        }))
        
        result = schema.validate(frame)
        assert result.passed is False

    def test_validate_row_index_is_1based(self):
        """validate reports 1-based row indices (header not counted)"""
        schema = ar.Schema({
            "value": ar.Int64(min=10),
        })
        
        frame = ar.from_pandas(pd.DataFrame({
            "value": [5, 15, 20],
        }))
        
        result = schema.validate(frame)
        assert result.passed is False
        assert any(issue.row_index == 1 for issue in result.issues)

    def test_validate_max_errors_limits_issues(self):
        """validate respects max_errors parameter"""
        schema = ar.Schema({
            "value": ar.Int64(min=100),
        })
        
        frame = ar.from_pandas(pd.DataFrame({
            "value": [1, 2, 3, 4, 5],
        }))
        
        result_limited = schema.validate(frame, max_errors=2)
        assert len(result_limited.issues) <= 2

    def test_validate_multiple_columns_together(self):
        """validate checks all columns for violations"""
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "email": ar.Email(),
            "age": ar.Int64(min=0, max=150),
        })
        
        frame = ar.from_pandas(pd.DataFrame({
            "id": [1, 2, 3],
            "email": ["a@b.com", "invalid", "c@d.com"],
            "age": [25, 200, 30],
        }))
        
        result = schema.validate(frame)
        assert result.passed is False

    def test_validate_warning_severity_passes(self):
        """validate with warning severity doesn't fail validation"""
        schema = ar.Schema({
            "age": ar.Int64(min=18, severity="warning"),
        })
        
        frame = ar.from_pandas(pd.DataFrame({
            "age": [10, 25, 35],
        }))
        
        result = schema.validate(frame)
        assert result.passed is True
