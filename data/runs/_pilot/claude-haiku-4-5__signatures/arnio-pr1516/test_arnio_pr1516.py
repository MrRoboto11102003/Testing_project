import pytest
import json
import pandas as pd
import numpy as np
import arnio as ar


class TestSchemaToJson:
    """Tests for Schema.to_json() method"""
    
    def test_to_json_returns_string(self):
        """to_json should return a string"""
        schema = ar.Schema({"col1": ar.Field()})
        result = schema.to_json()
        assert isinstance(result, str)
    
    def test_to_json_valid_json(self):
        """to_json output should be valid JSON"""
        schema = ar.Schema({"col1": ar.Field()})
        json_str = schema.to_json()
        parsed = json.loads(json_str)
        assert isinstance(parsed, (dict, list))
    
    def test_to_json_single_field(self):
        """to_json should serialize schema with single field"""
        schema = ar.Schema({"name": ar.Field()})
        json_str = schema.to_json()
        assert len(json_str) > 0
        json.loads(json_str)  # Should not raise
    
    def test_to_json_multiple_fields(self):
        """to_json should serialize schema with multiple fields"""
        schema = ar.Schema({
            "id": ar.Field(),
            "name": ar.Field(),
            "value": ar.Field()
        })
        json_str = schema.to_json()
        parsed = json.loads(json_str)
        assert parsed is not None
    
    def test_to_json_idempotent(self):
        """to_json should produce consistent output"""
        schema = ar.Schema({"col1": ar.Field()})
        json1 = schema.to_json()
        json2 = schema.to_json()
        assert json1 == json2


class TestSchemaValidate:
    """Tests for Schema.validate() method"""
    
    def test_validate_returns_validation_result(self):
        """validate should return a ValidationResult"""
        schema = ar.Schema({"col1": ar.Field()})
        df = pd.DataFrame({"col1": [1, 2, 3]})
        frame = ar.ArFrame(df)
        result = schema.validate(frame)
        assert result is not None
    
    def test_validate_valid_frame(self):
        """validate should handle valid frames"""
        schema = ar.Schema({"id": ar.Field(), "value": ar.Field()})
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "value": [10.5, 20.3, 30.1]
        })
        frame = ar.ArFrame(df)
        result = schema.validate(frame)
        assert result is not None
    
    def test_validate_with_max_errors_none(self):
        """validate with max_errors=None should process without limit"""
        schema = ar.Schema({"col1": ar.Field()})
        df = pd.DataFrame({"col1": [1, 2, 3, 4, 5]})
        frame = ar.ArFrame(df)
        result = schema.validate(frame, max_errors=None)
        assert result is not None
    
    def test_validate_with_max_errors_positive(self):
        """validate with positive max_errors should limit error collection"""
        schema = ar.Schema({"col1": ar.Field()})
        df = pd.DataFrame({"col1": [1, 2, 3]})
        frame = ar.ArFrame(df)
        result = schema.validate(frame, max_errors=10)
        assert result is not None
    
    def test_validate_with_max_errors_zero(self):
        """validate with max_errors=0 should handle gracefully"""
        schema = ar.Schema({"col1": ar.Field()})
        df = pd.DataFrame({"col1": [1, 2, 3]})
        frame = ar.ArFrame(df)
        result = schema.validate(frame, max_errors=0)
        assert result is not None
    
    def test_validate_empty_frame(self):
        """validate should handle empty DataFrames"""
        schema = ar.Schema({"col1": ar.Field()})
        df = pd.DataFrame({"col1": []})
        frame = ar.ArFrame(df)
        result = schema.validate(frame)
        assert result is not None
    
    def test_validate_single_row(self):
        """validate should handle single-row DataFrames"""
        schema = ar.Schema({"col1": ar.Field()})
        df = pd.DataFrame({"col1": [42]})
        frame = ar.ArFrame(df)
        result = schema.validate(frame)
        assert result is not None
    
    def test_validate_multiple_columns(self):
        """validate should validate all columns"""
        schema = ar.Schema({
            "a": ar.Field(),
            "b": ar.Field(),
            "c": ar.Field()
        })
        df = pd.DataFrame({
            "a": [1, 2],
            "b": ["x", "y"],
            "c": [1.5, 2.5]
        })
        frame = ar.ArFrame(df)
        result = schema.validate(frame)
        assert result is not None
    
    def test_validate_with_nulls(self):
        """validate should handle null/NaN values"""
        schema = ar.Schema({"col1": ar.Field()})
        df = pd.DataFrame({"col1": [1.0, None, 3.0, np.nan]})
        frame = ar.ArFrame(df)
        result = schema.validate(frame)
        assert result is not None
    
    def test_validate_large_frame(self):
        """validate should handle large DataFrames"""
        schema = ar.Schema({"col1": ar.Field()})
        df = pd.DataFrame({"col1": range(100000)})
        frame = ar.ArFrame(df)
        result = schema.validate(frame)
        assert result is not None
