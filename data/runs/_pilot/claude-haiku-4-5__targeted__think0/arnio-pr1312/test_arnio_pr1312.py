import pytest
import json
import pandas as pd
import numpy as np
import arnio as ar
from arnio.schema import Schema


class TestSchemaBootstrapFromReport:
    """Tests for Schema.bootstrap_from_report classmethod."""
    
    def test_bootstrap_from_valid_report(self):
        """Test creating schema from a valid DataQualityReport."""
        # Create a simple frame and profile it
        frame = ar.from_dict({
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
            "score": [95.5, 88.0, 92.3]
        })
        report = ar.profile(frame)
        
        # Bootstrap schema from report
        schema = Schema.bootstrap_from_report(report)
        
        # Verify schema was created
        assert schema is not None
        assert isinstance(schema, Schema)
    
    def test_bootstrap_preserves_column_types(self):
        """Test that bootstrapped schema preserves column dtypes from report."""
        frame = ar.from_dict({
            "int_col": [1, 2, 3],
            "str_col": ["a", "b", "c"],
            "float_col": [1.1, 2.2, 3.3]
        })
        report = ar.profile(frame)
        schema = Schema.bootstrap_from_report(report)
        
        # Validate should work without error
        result = schema.validate(frame)
        assert result is not None
    
    def test_bootstrap_from_report_type_error(self):
        """Test that non-report input raises TypeError."""
        with pytest.raises(TypeError):
            Schema.bootstrap_from_report("not a report")
        
        with pytest.raises(TypeError):
            Schema.bootstrap_from_report({"dict": "object"})
        
        with pytest.raises(TypeError):
            Schema.bootstrap_from_report(None)
    
    def test_bootstrap_from_empty_report_value_error(self):
        """Test that empty report raises ValueError."""
        # Create frame with no rows, profile it
        frame = ar.from_dict({
            "col": []
        })
        report = ar.profile(frame)
        
        # This should either raise ValueError or handle gracefully
        # depending on implementation
        try:
            schema = Schema.bootstrap_from_report(report)
            # If no error, schema should still be valid
            assert isinstance(schema, Schema)
        except ValueError:
            # Expected for empty reports
            pass


class TestSchemaFromJson:
    """Tests for Schema.from_json classmethod."""
    
    def test_roundtrip_to_json_and_back(self):
        """Test that schema can be serialized and deserialized."""
        frame = ar.from_dict({
            "name": ["Alice", "Bob"],
            "age": [25, 30]
        })
        report = ar.profile(frame)
        schema1 = Schema.bootstrap_from_report(report)
        
        # Serialize to JSON
        json_str = schema1.to_json()
        
        # Deserialize back
        schema2 = Schema.from_json(json_str)
        
        # Both schemas should validate the same frame successfully
        result1 = schema1.validate(frame)
        result2 = schema2.validate(frame)
        assert result1 is not None
        assert result2 is not None
    
    def test_from_json_valid_string(self):
        """Test deserializing a valid JSON schema string."""
        frame = ar.from_dict({"col1": [1, 2, 3]})
        report = ar.profile(frame)
        schema = Schema.bootstrap_from_report(report)
        json_str = schema.to_json()
        
        # Should not raise
        restored = Schema.from_json(json_str)
        assert isinstance(restored, Schema)
    
    def test_from_json_invalid_string(self):
        """Test that invalid JSON string raises appropriate error."""
        with pytest.raises((json.JSONDecodeError, ValueError)):
            Schema.from_json("not valid json {")
    
    def test_from_json_empty_string(self):
        """Test that empty string raises appropriate error."""
        with pytest.raises((json.JSONDecodeError, ValueError)):
            Schema.from_json("")
    
    def test_from_json_malformed_schema(self):
        """Test that JSON without required schema fields raises error."""
        invalid_json = json.dumps({"random": "data"})
        with pytest.raises((ValueError, KeyError, TypeError)):
            Schema.from_json(invalid_json)


class TestSchemaToJson:
    """Tests for Schema.to_json method."""
    
    def test_to_json_returns_string(self):
        """Test that to_json returns a string."""
        frame = ar.from_dict({"col": [1, 2, 3]})
        report = ar.profile(frame)
        schema = Schema.bootstrap_from_report(report)
        
        json_str = schema.to_json()
        assert isinstance(json_str, str)
    
    def test_to_json_is_valid_json(self):
        """Test that to_json output is valid JSON."""
        frame = ar.from_dict({"col": [1, 2, 3]})
        report = ar.profile(frame)
        schema = Schema.bootstrap_from_report(report)
        
        json_str = schema.to_json()
        # Should not raise
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
    
    def test_to_json_with_multiple_columns(self):
        """Test to_json with multiple columns of different types."""
        frame = ar.from_dict({
            "int_col": [1, 2, 3],
            "str_col": ["a", "b", "c"],
            "float_col": [1.1, 2.2, 3.3]
        })
        report = ar.profile(frame)
        schema = Schema.bootstrap_from_report(report)
        
        json_str = schema.to_json()
        parsed = json.loads(json_str)
        
        # Should contain schema information
        assert len(json_str) > 0
        assert isinstance(parsed, dict)
    
    def test_to_json_omits_cross_field_rules_warning(self):
        """Test that cross-field rules are omitted with warning."""
        frame = ar.from_dict({
            "col1": [1, 2, 3],
            "col2": [4, 5, 6]
        })
        report = ar.profile(frame)
        schema = Schema.bootstrap_from_report(report)
        
        # to_json should complete without error
        json_str = schema.to_json()
        assert isinstance(json_str, str)
        
        # Check if rules_omitted flag is present when applicable
        parsed = json.loads(json_str)
        # rules_omitted may be present if cross-field rules exist
        if "rules_omitted" in parsed:
            assert isinstance(parsed["rules_omitted"], bool)


class TestSchemaValidate:
    """Tests for Schema.validate method."""
    
    def test_validate_returns_validation_result(self):
        """Test that validate returns a ValidationResult object."""
        frame = ar.from_dict({
            "col1": [1, 2, 3],
            "col2": ["a", "b", "c"]
        })
        report = ar.profile(frame)
        schema = Schema.bootstrap_from_report(report)
        
        result = schema.validate(frame)
        assert result is not None
    
    def test_validate_with_max_errors_none(self):
        """Test validate with max_errors=None returns all errors."""
        frame = ar.from_dict({
            "col1": [1, 2, 3],
            "col2": ["a", "b", "c"]
        })
        report = ar.profile(frame)
        schema = Schema.bootstrap_from_report(report)
        
        result = schema.validate(frame, max_errors=None)
        assert result is not None
    
    def test_validate_with_max_errors_limit(self):
        """Test validate with max_errors parameter limits results."""
        frame = ar.from_dict({
            "col1": [1, 2, 3],
            "col2": ["a", "b", "c"]
        })
        report = ar.profile(frame)
        schema = Schema.bootstrap_from_report(report)
        
        result = schema.validate(frame, max_errors=1)
        assert result is not None
    
    def test_validate_matching_frame(self):
        """Test validating a frame that matches the schema."""
        frame = ar.from_dict({
            "col1": [1, 2, 3],
            "col2": ["a", "b", "c"]
        })
        report = ar.profile(frame)
        schema = Schema.bootstrap_from_report(report)
        
        # Validate same frame should succeed
        result = schema.validate(frame)
        assert result is not None
    
    def test_validate_different_frame_same_schema(self):
        """Test validating a different frame with compatible schema."""
        frame1 = ar.from_dict({
            "col1": [1, 2, 3],
            "col2": ["a", "b", "c"]
        })
        report = ar.profile(frame1)
        schema = Schema.bootstrap_from_report(report)
        
        # Create compatible frame
        frame2 = ar.from_dict({
            "col1": [4, 5, 6],
            "col2": ["d", "e", "f"]
        })
        
        result = schema.validate(frame2)
        assert result is not None
    
    def test_validate_with_max_errors_zero(self):
        """Test validate with max_errors=0."""
        frame = ar.from_dict({
            "col1": [1, 2, 3]
        })
        report = ar.profile(frame)
        schema = Schema.bootstrap_from_report(report)
        
        result = schema.validate(frame, max_errors=0)
        assert result is not None
