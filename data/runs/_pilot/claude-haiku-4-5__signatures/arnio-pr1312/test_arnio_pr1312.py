import pytest
import pandas as pd
import numpy as np
import json
from arnio.schema import Schema, ValidationResult
from arnio import ArFrame


class TestSchemaBootstrapFromReport:
    """Tests for Schema.bootstrap_from_report classmethod."""
    
    def test_bootstrap_from_dict_report(self):
        """Test bootstrapping schema from dictionary report."""
        report = {"columns": {"col1": {"type": "int64"}}}
        schema = Schema.bootstrap_from_report(report)
        assert isinstance(schema, Schema)
    
    def test_bootstrap_from_dataframe_report(self):
        """Test bootstrapping schema from DataFrame."""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        schema = Schema.bootstrap_from_report(df)
        assert isinstance(schema, Schema)
    
    def test_bootstrap_from_empty_report(self):
        """Test bootstrapping schema from empty report."""
        schema = Schema.bootstrap_from_report({})
        assert isinstance(schema, Schema)


class TestSchemaFromJson:
    """Tests for Schema.from_json classmethod."""
    
    def test_from_json_simple_schema(self):
        """Test loading simple schema from JSON string."""
        json_str = '{"columns": {"col1": {"type": "int64"}}}'
        schema = Schema.from_json(json_str)
        assert isinstance(schema, Schema)
    
    def test_from_json_empty_schema(self):
        """Test loading empty schema from JSON."""
        json_str = '{}'
        schema = Schema.from_json(json_str)
        assert isinstance(schema, Schema)
    
    def test_from_json_multiple_columns(self):
        """Test loading schema with multiple column definitions."""
        json_str = '{"columns": {"col1": {"type": "int64"}, "col2": {"type": "object"}}}'
        schema = Schema.from_json(json_str)
        assert isinstance(schema, Schema)
    
    def test_from_json_invalid_raises_error(self):
        """Test that invalid JSON raises JSONDecodeError."""
        with pytest.raises(json.JSONDecodeError):
            Schema.from_json("{invalid json")


class TestSchemaToJson:
    """Tests for Schema.to_json method."""
    
    def test_to_json_returns_string(self):
        """Test that to_json returns string type."""
        schema = Schema.from_json('{"columns": {}}')
        result = schema.to_json()
        assert isinstance(result, str)
    
    def test_to_json_valid_json_output(self):
        """Test that to_json output is valid JSON."""
        schema = Schema.from_json('{"columns": {"col1": {"type": "int64"}}}')
        result = schema.to_json()
        parsed = json.loads(result)
        assert isinstance(parsed, dict)
    
    def test_to_json_round_trip(self):
        """Test schema survives JSON serialization round-trip."""
        original = '{"columns": {"col1": {"type": "int64"}, "col2": {"type": "object"}}}'
        schema1 = Schema.from_json(original)
        serialized = schema1.to_json()
        schema2 = Schema.from_json(serialized)
        assert schema1.to_json() == schema2.to_json()


class TestSchemaValidate:
    """Tests for Schema.validate method."""
    
    def test_validate_returns_validation_result(self):
        """Test that validate returns ValidationResult instance."""
        schema = Schema.from_json('{"columns": {"col1": {"type": "int64"}}}')
        df = pd.DataFrame({"col1": [1, 2, 3]})
        frame = ArFrame(df)
        result = schema.validate(frame)
        assert isinstance(result, ValidationResult)
    
    def test_validate_with_max_errors_limit(self):
        """Test validate respects max_errors parameter."""
        schema = Schema.from_json('{"columns": {"col1": {"type": "int64"}}}')
        df = pd.DataFrame({"col1": [1, 2, 3]})
        frame = ArFrame(df)
        result = schema.validate(frame, max_errors=10)
        assert isinstance(result, ValidationResult)
    
    def test_validate_with_max_errors_none(self):
        """Test validate with max_errors=None (no limit)."""
        schema = Schema.from_json('{"columns": {"col1": {"type": "int64"}}}')
        df = pd.DataFrame({"col1": [1, 2, 3]})
        frame = ArFrame(df)
        result = schema.validate(frame, max_errors=None)
        assert isinstance(result, ValidationResult)
    
    def test_validate_empty_dataframe(self):
        """Test validate on empty dataframe."""
        schema = Schema.from_json('{"columns": {"col1": {"type": "int64"}}}')
        df = pd.DataFrame({"col1": pd.Series([], dtype='int64')})
        frame = ArFrame(df)
        result = schema.validate(frame)
        assert isinstance(result, ValidationResult)
    
    def test_validate_with_nan_values(self):
        """Test validate on dataframe with NaN values."""
        schema = Schema.from_json('{"columns": {"col1": {"type": "float64"}}}')
        df = pd.DataFrame({"col1": [1.0, np.nan, 3.0]})
        frame = ArFrame(df)
        result = schema.validate(frame)
        assert isinstance(result, ValidationResult)
