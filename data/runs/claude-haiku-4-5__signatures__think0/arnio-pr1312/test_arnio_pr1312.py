import json
import pytest
import pandas as pd
import numpy as np
from io import StringIO

import arnio as ar


class TestSchemaBootstrapFromReport:
    """Test Schema.bootstrap_from_report class method."""
    
    def test_bootstrap_from_report_basic(self):
        """Test bootstrapping schema from a basic report."""
        report = {
            "columns": {
                "age": {"dtype": "int64"},
                "name": {"dtype": "object"},
            }
        }
        schema = ar.Schema.bootstrap_from_report(report)
        assert schema is not None
        assert isinstance(schema, ar.Schema)
    
    def test_bootstrap_from_report_with_validation_rules(self):
        """Test bootstrapping schema with validation rules in report."""
        report = {
            "columns": {
                "age": {"dtype": "int64", "min": 0, "max": 150},
                "email": {"dtype": "object", "pattern": r".*@.*"},
            }
        }
        schema = ar.Schema.bootstrap_from_report(report)
        assert schema is not None
    
    def test_bootstrap_from_report_empty_report(self):
        """Test bootstrapping from minimal/empty report structure."""
        report = {"columns": {}}
        schema = ar.Schema.bootstrap_from_report(report)
        assert schema is not None


class TestSchemaFromJson:
    """Test Schema.from_json class method."""
    
    def test_from_json_basic_schema(self):
        """Test parsing basic JSON schema string."""
        json_str = '{"columns": {"col1": {"dtype": "int64"}}}'
        schema = ar.Schema.from_json(json_str)
        assert schema is not None
        assert isinstance(schema, ar.Schema)
    
    def test_from_json_roundtrip(self):
        """Test that from_json can parse to_json output."""
        original_json = '{"columns": {"age": {"dtype": "int64"}, "name": {"dtype": "object"}}}'
        schema = ar.Schema.from_json(original_json)
        json_output = schema.to_json()
        
        # Parse both to dicts and compare structure
        original_dict = json.loads(original_json)
        output_dict = json.loads(json_output)
        assert original_dict == output_dict
    
    def test_from_json_complex_schema(self):
        """Test parsing complex schema with multiple rules."""
        json_str = json.dumps({
            "columns": {
                "id": {"dtype": "int64", "nullable": False},
                "score": {"dtype": "float64", "min": 0.0, "max": 100.0},
                "category": {"dtype": "object"},
            }
        })
        schema = ar.Schema.from_json(json_str)
        assert schema is not None
    
    def test_from_json_invalid_json(self):
        """Test that invalid JSON raises appropriate error."""
        with pytest.raises(Exception):
            ar.Schema.from_json("invalid json {")
    
    def test_from_json_empty_columns(self):
        """Test parsing JSON with empty columns."""
        json_str = '{"columns": {}}'
        schema = ar.Schema.from_json(json_str)
        assert schema is not None


class TestSchemaToJson:
    """Test Schema.to_json method."""
    
    def test_to_json_returns_string(self):
        """Test that to_json returns a string."""
        schema = ar.Schema.bootstrap_from_report({"columns": {"col1": {"dtype": "int64"}}})
        json_str = schema.to_json()
        assert isinstance(json_str, str)
    
    def test_to_json_valid_json(self):
        """Test that to_json output is valid JSON."""
        schema = ar.Schema.bootstrap_from_report(
            {"columns": {"col1": {"dtype": "int64"}, "col2": {"dtype": "object"}}}
        )
        json_str = schema.to_json()
        # Should not raise
        parsed = json.loads(json_str)
        assert "columns" in parsed
    
    def test_to_json_preserves_structure(self):
        """Test that to_json preserves schema structure."""
        report = {
            "columns": {
                "age": {"dtype": "int64"},
                "name": {"dtype": "object"},
            }
        }
        schema = ar.Schema.bootstrap_from_report(report)
        json_str = schema.to_json()
        parsed = json.loads(json_str)
        
        assert "age" in parsed["columns"]
        assert "name" in parsed["columns"]
        assert parsed["columns"]["age"]["dtype"] == "int64"
        assert parsed["columns"]["name"]["dtype"] == "object"


class TestSchemaValidate:
    """Test Schema.validate method."""
    
    def test_validate_valid_dataframe(self):
        """Test validation of a valid DataFrame."""
        df = pd.DataFrame({
            "age": [25, 30, 35],
            "name": ["Alice", "Bob", "Charlie"]
        })
        schema = ar.Schema.bootstrap_from_report({
            "columns": {
                "age": {"dtype": "int64"},
                "name": {"dtype": "object"},
            }
        })
        result = schema.validate(df)
        assert isinstance(result, ar.ValidationResult)
    
    def test_validate_with_max_errors(self):
        """Test validation with max_errors parameter."""
        df = pd.DataFrame({
            "age": [25, 30, 35],
            "name": ["Alice", "Bob", "Charlie"]
        })
        schema = ar.Schema.bootstrap_from_report({
            "columns": {
                "age": {"dtype": "int64"},
                "name": {"dtype": "object"},
            }
        })
        # Should not raise
        result = schema.validate(df, max_errors=5)
        assert result is not None
    
    def test_validate_max_errors_zero(self):
        """Test validation with max_errors=0."""
        df = pd.DataFrame({
            "age": [25, 30, 35],
            "name": ["Alice", "Bob", "Charlie"]
        })
        schema = ar.Schema.bootstrap_from_report({
            "columns": {
                "age": {"dtype": "int64"},
                "name": {"dtype": "object"},
            }
        })
        result = schema.validate(df, max_errors=0)
        assert result is not None
    
    def test_validate_empty_dataframe(self):
        """Test validation of an empty DataFrame."""
        df = pd.DataFrame({
            "age": pd.Series([], dtype="int64"),
            "name": pd.Series([], dtype="object")
        })
        schema = ar.Schema.bootstrap_from_report({
            "columns": {
                "age": {"dtype": "int64"},
                "name": {"dtype": "object"},
            }
        })
        result = schema.validate(df)
        assert result is not None
    
    def test_validate_with_nullable_column(self):
        """Test validation respects nullable column rules."""
        df = pd.DataFrame({
            "age": [25, np.nan, 35],
            "name": ["Alice", None, "Charlie"]
        })
        schema = ar.Schema.bootstrap_from_report({
            "columns": {
                "age": {"dtype": "float64", "nullable": True},
                "name": {"dtype": "object", "nullable": True},
            }
        })
        result = schema.validate(df)
        assert result is not None
    
    def test_validate_returns_validation_result(self):
        """Test that validate returns a ValidationResult object."""
        df = pd.DataFrame({"col1": [1, 2, 3]})
        schema = ar.Schema.bootstrap_from_report({
            "columns": {"col1": {"dtype": "int64"}}
        })
        result = schema.validate(df)
        assert hasattr(result, "errors") or hasattr(result, "is_valid")
