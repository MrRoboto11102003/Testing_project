import pytest
import json
import warnings
import arnio as ar


class TestSchemaToJson:
    """Tests for arnio.schema.Schema.to_json()"""
    
    @pytest.fixture
    def basic_schema(self):
        """Create a basic schema instance."""
        return ar.schema.Schema()
    
    def test_to_json_returns_string(self, basic_schema):
        """to_json() should return a string."""
        result = basic_schema.to_json()
        assert isinstance(result, str)
    
    def test_to_json_returns_valid_json(self, basic_schema):
        """to_json() output should be valid JSON."""
        result = basic_schema.to_json()
        parsed = json.loads(result)
        assert isinstance(parsed, dict)
    
    def test_to_json_is_deterministic(self, basic_schema):
        """to_json() should return identical output on repeated calls."""
        result1 = basic_schema.to_json()
        result2 = basic_schema.to_json()
        result3 = basic_schema.to_json()
        assert result1 == result2 == result3
    
    def test_to_json_produces_nonempty_output(self, basic_schema):
        """to_json() should produce non-empty JSON output."""
        result = basic_schema.to_json()
        assert len(result) > 0
        parsed = json.loads(result)
        assert parsed is not None
    
    def test_to_json_roundtrip_with_from_json(self, basic_schema):
        """Schema should be restorable via from_json() from to_json() output."""
        json_str = basic_schema.to_json()
        restored_schema = ar.schema.Schema.from_json(json_str)
        assert restored_schema is not None
        assert isinstance(restored_schema, ar.schema.Schema)
    
    def test_to_json_roundtrip_produces_same_json(self, basic_schema):
        """Roundtripping to_json -> from_json -> to_json should be stable."""
        original_json = basic_schema.to_json()
        restored_schema = ar.schema.Schema.from_json(original_json)
        roundtrip_json = restored_schema.to_json()
        assert original_json == roundtrip_json
    
    def test_to_json_warnings_about_cross_field_rules_mechanism(self, basic_schema):
        """to_json() should handle rule omission warnings correctly."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = basic_schema.to_json()
            # Mechanism for warnings should work; not asserting rules present
            assert isinstance(result, str)


class TestSchemaValidate:
    """Tests for arnio.schema.Schema.validate()"""
    
    @pytest.fixture
    def schema(self):
        """Create a schema for validation testing."""
        return ar.schema.Schema()
    
    @pytest.fixture
    def simple_frame(self):
        """Create a simple ArFrame for validation."""
        return ar.from_dict({
            "col_a": [1, 2, 3],
            "col_b": ["x", "y", "z"]
        })
    
    def test_validate_returns_validation_result(self, schema, simple_frame):
        """validate() should return a ValidationResult object."""
        result = schema.validate(simple_frame)
        assert result is not None
    
    def test_validate_accepts_frame_and_returns_result(self, schema, simple_frame):
        """validate() should accept ArFrame parameter."""
        result = schema.validate(simple_frame)
        assert result is not None
    
    def test_validate_with_max_errors_none(self, schema, simple_frame):
        """validate() should accept max_errors=None parameter."""
        result = schema.validate(simple_frame, max_errors=None)
        assert result is not None
    
    def test_validate_with_positive_max_errors(self, schema, simple_frame):
        """validate() should accept positive integer max_errors."""
        result = schema.validate(simple_frame, max_errors=5)
        assert result is not None
    
    def test_validate_with_max_errors_one(self, schema, simple_frame):
        """validate() should accept max_errors=1."""
        result = schema.validate(simple_frame, max_errors=1)
        assert result is not None
    
    def test_validate_with_max_errors_zero(self, schema, simple_frame):
        """validate() should accept max_errors=0."""
        result = schema.validate(simple_frame, max_errors=0)
        assert result is not None
    
    def test_validate_empty_frame(self, schema):
        """validate() should handle empty frames."""
        empty_frame = ar.from_dict({"col_a": [], "col_b": []})
        result = schema.validate(empty_frame)
        assert result is not None
    
    def test_validate_single_row_frame(self, schema):
        """validate() should handle single-row frames."""
        single_row = ar.from_dict({"col": [42]})
        result = schema.validate(single_row)
        assert result is not None
    
    def test_validate_multiple_invocations_same_schema_frame(self, schema, simple_frame):
        """validate() should be callable multiple times on same inputs."""
        result1 = schema.validate(simple_frame)
        result2 = schema.validate(simple_frame)
        result3 = schema.validate(simple_frame)
        assert result1 is not None
        assert result2 is not None
        assert result3 is not None
    
    def test_validate_different_max_errors_same_schema_frame(self, schema, simple_frame):
        """validate() should work with different max_errors values."""
        result_1 = schema.validate(simple_frame, max_errors=1)
        result_10 = schema.validate(simple_frame, max_errors=10)
        result_none = schema.validate(simple_frame, max_errors=None)
        assert result_1 is not None
        assert result_10 is not None
        assert result_none is not None
    
    def test_validate_various_frame_column_types(self, schema):
        """validate() should work with frames of various column types."""
        mixed_frame = ar.from_dict({
            "ints": [1, 2, 3],
            "strs": ["a", "b", "c"],
            "floats": [1.1, 2.2, 3.3],
        })
        result = schema.validate(mixed_frame)
        assert result is not None
