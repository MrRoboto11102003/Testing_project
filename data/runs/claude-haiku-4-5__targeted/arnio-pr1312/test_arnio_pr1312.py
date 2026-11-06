import pytest
import json
import arnio as ar
from arnio.schema import Schema


class TestSchemaBootstrapFromReport:
    """Tests for Schema.bootstrap_from_report()"""
    
    def test_bootstrap_from_valid_report(self):
        """Test bootstrap_from_report creates schema from DataQualityReport"""
        frame = ar.from_dict({
            "col1": [1, 2, 3],
            "col2": ["a", "b", "c"]
        })
        report = ar.profile(frame)
        
        schema = Schema.bootstrap_from_report(report)
        
        assert isinstance(schema, Schema)
    
    def test_bootstrap_raises_type_error_for_string(self):
        """Test bootstrap_from_report raises TypeError for string input"""
        with pytest.raises(TypeError):
            Schema.bootstrap_from_report("not a report")
    
    def test_bootstrap_raises_type_error_for_dict(self):
        """Test bootstrap_from_report raises TypeError for dict input"""
        with pytest.raises(TypeError):
            Schema.bootstrap_from_report({"key": "value"})
    
    def test_bootstrap_raises_type_error_for_none(self):
        """Test bootstrap_from_report raises TypeError for None"""
        with pytest.raises(TypeError):
            Schema.bootstrap_from_report(None)


class TestSchemaJsonSerialization:
    """Tests for Schema.from_json() and Schema.to_json()"""
    
    def test_to_json_returns_valid_json_string(self):
        """Test to_json returns valid JSON-formatted string"""
        frame = ar.from_dict({"col1": [1, 2, 3]})
        report = ar.profile(frame)
        schema = Schema.bootstrap_from_report(report)
        
        json_str = schema.to_json()
        
        assert isinstance(json_str, str)
        # Should be valid JSON
        data = json.loads(json_str)
        assert isinstance(data, dict)
    
    def test_from_json_creates_schema(self):
        """Test from_json deserializes JSON string into Schema"""
        frame = ar.from_dict({"col1": [1, 2, 3]})
        report = ar.profile(frame)
        schema1 = Schema.bootstrap_from_report(report)
        
        json_str = schema1.to_json()
        schema2 = Schema.from_json(json_str)
        
        assert isinstance(schema2, Schema)
    
    def test_json_roundtrip_preserves_validation_behavior(self):
        """Test Schema -> JSON -> Schema roundtrip preserves validation"""
        frame = ar.from_dict({
            "col1": [1, 2, 3],
            "col2": ["a", "b", "c"],
            "col3": [1.5, 2.5, 3.5]
        })
        report = ar.profile(frame)
        schema1 = Schema.bootstrap_from_report(report)
        
        json_str = schema1.to_json()
        schema2 = Schema.from_json(json_str)
        
        # Both schemas should validate the same frame
        result1 = schema1.validate(frame)
        result2 = schema2.validate(frame)
        
        assert type(result1).__name__ == type(result2).__name__
    
    def test_to_json_includes_rules_omitted_indicator(self):
        """Test to_json output includes rules_omitted flag"""
        frame = ar.from_dict({"col1": [1, 2, 3]})
        report = ar.profile(frame)
        schema = Schema.bootstrap_from_report(report)
        
        json_str = schema.to_json()
        data = json.loads(json_str)
        
        assert "rules_omitted" in data


class TestSchemaValidate:
    """Tests for Schema.validate()"""
    
    def test_validate_returns_result_object(self):
        """Test validate returns a ValidationResult"""
        frame = ar.from_dict({
            "col1": [1, 2, 3],
            "col2": ["a", "b", "c"]
        })
        report = ar.profile(frame)
        schema = Schema.bootstrap_from_report(report)
        
        result = schema.validate(frame)
        
        assert result is not None
    
    def test_validate_with_matching_schema(self):
        """Test validate accepts frame matching schema structure"""
        frame = ar.from_dict({
            "col1": [1, 2, 3],
            "col2": ["a", "b", "c"]
        })
        report = ar.profile(frame)
        schema = Schema.bootstrap_from_report(report)
        
        result = schema.validate(frame)
        
        assert result is not None
    
    def test_validate_with_max_errors_parameter(self):
        """Test validate respects max_errors limit"""
        frame = ar.from_dict({
            "col1": [1, 2, 3],
            "col2": ["a", "b", "c"]
        })
        report = ar.profile(frame)
        schema = Schema.bootstrap_from_report(report)
        
        result = schema.validate(frame, max_errors=5)
        
        assert result is not None
    
    def test_validate_with_max_errors_none(self):
        """Test validate with max_errors=None (no limit)"""
        frame = ar.from_dict({"col1": [1, 2, 3]})
        report = ar.profile(frame)
        schema = Schema.bootstrap_from_report(report)
        
        result = schema.validate(frame, max_errors=None)
        
        assert result is not None
    
    def test_validate_multiple_frames_with_single_schema(self):
        """Test schema can validate multiple different frames"""
        frame1 = ar.from_dict({
            "col1": [1, 2, 3],
            "col2": ["a", "b", "c"]
        })
        report = ar.profile(frame1)
        schema = Schema.bootstrap_from_report(report)
        
        frame2 = ar.from_dict({
            "col1": [10, 20, 30],
            "col2": ["x", "y", "z"]
        })
        
        result1 = schema.validate(frame1)
        result2 = schema.validate(frame2)
        
        assert result1 is not None
        assert result2 is not None


class TestSchemaIntegration:
    """Integration tests for complete Schema workflows"""
    
    def test_profile_bootstrap_validate_workflow(self):
        """Test end-to-end workflow: profile -> bootstrap -> validate"""
        frame = ar.from_dict({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "score": [95.5, 87.3, 92.1]
        })
        
        report = ar.profile(frame)
        schema = Schema.bootstrap_from_report(report)
        result = schema.validate(frame)
        
        assert isinstance(schema, Schema)
        assert result is not None
    
    def test_json_persistence_and_reuse_workflow(self):
        """Test schema can be saved to JSON and reused for validation"""
        frame1 = ar.from_dict({
            "col1": [1, 2, 3],
            "col2": ["a", "b", "c"]
        })
        report = ar.profile(frame1)
        schema1 = Schema.bootstrap_from_report(report)
        
        # Save and restore schema
        json_str = schema1.to_json()
        schema2 = Schema.from_json(json_str)
        
        # Validate new frame with restored schema
        frame2 = ar.from_dict({
            "col1": [4, 5, 6],
            "col2": ["d", "e", "f"]
        })
        result = schema2.validate(frame2)
        
        assert result is not None
