import pytest
import json
import warnings
import arnio as ar
from arnio.schema import Schema, Field, ValidationResult


@pytest.fixture
def simple_frame():
    """Create a simple ArFrame with no nulls."""
    return ar.from_dict({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "score": [85.5, 92.3, 78.9]
    })


@pytest.fixture
def frame_with_nulls():
    """Create an ArFrame with null values."""
    return ar.from_dict({
        "id": [1, None, 3],
        "name": ["Alice", "Bob", None],
        "score": [85.5, None, 78.9]
    })


@pytest.fixture
def simple_schema():
    """Create a simple schema for testing."""
    return Schema(fields={
        "id": Field(dtype="int64", nullable=False),
        "name": Field(dtype="string", nullable=True),
        "score": Field(dtype="float64", nullable=False)
    })


class TestBootstrapFromReport:
    """Tests for Schema.bootstrap_from_report()"""

    def test_bootstrap_creates_schema_from_report(self, simple_frame):
        """bootstrap_from_report should create a valid Schema from a DataQualityReport."""
        report = ar.profile(simple_frame)
        schema = Schema.bootstrap_from_report(report)
        
        assert isinstance(schema, Schema)
        assert len(schema.fields) > 0
        assert "id" in schema.fields
        assert "name" in schema.fields
        assert "score" in schema.fields

    def test_bootstrap_detects_nullable_true_with_nulls(self, frame_with_nulls):
        """bootstrap_from_report should mark columns as nullable if they contain nulls."""
        report = ar.profile(frame_with_nulls)
        schema = Schema.bootstrap_from_report(report)
        
        # All columns have at least one null value
        for field in schema.fields.values():
            assert field.nullable is True

    def test_bootstrap_detects_nullable_false_without_nulls(self, simple_frame):
        """bootstrap_from_report should mark columns as non-nullable if they have no nulls."""
        report = ar.profile(simple_frame)
        schema = Schema.bootstrap_from_report(report)
        
        # simple_frame has no null values
        for field in schema.fields.values():
            assert field.nullable is False

    def test_bootstrap_raises_type_error_on_non_report(self):
        """bootstrap_from_report should raise TypeError when given non-DataQualityReport input."""
        with pytest.raises(TypeError, match="Expected DataQualityReport"):
            Schema.bootstrap_from_report("not a report")
        
        with pytest.raises(TypeError, match="Expected DataQualityReport"):
            Schema.bootstrap_from_report({})
        
        with pytest.raises(TypeError, match="Expected DataQualityReport"):
            Schema.bootstrap_from_report(None)


class TestToJson:
    """Tests for Schema.to_json()"""

    def test_to_json_produces_valid_json_string(self, simple_schema):
        """to_json should produce valid JSON that can be parsed."""
        result = simple_schema.to_json()
        assert isinstance(result, str)
        payload = json.loads(result)
        assert isinstance(payload, dict)

    def test_to_json_includes_all_fields_and_metadata(self, simple_schema):
        """to_json should include fields, strict, and unique in output."""
        result = simple_schema.to_json()
        payload = json.loads(result)
        
        assert "fields" in payload
        assert "strict" in payload
        assert "unique" in payload
        assert "id" in payload["fields"]
        assert "name" in payload["fields"]
        assert "score" in payload["fields"]

    def test_to_json_with_rules_emits_user_warning(self):
        """to_json should emit UserWarning when schema contains rules."""
        schema = Schema(fields={"col": Field()}, rules=[lambda df: []])
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            schema.to_json()
            
            assert len(w) >= 1
            assert issubclass(w[0].category, UserWarning)
            assert "rules" in str(w[0].message).lower()

    def test_to_json_sets_rules_omitted_flag_when_rules_present(self):
        """to_json should set rules_omitted: true when rules are present."""
        schema = Schema(fields={"col": Field()}, rules=[lambda df: []])
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = schema.to_json()
        
        payload = json.loads(result)
        assert payload.get("rules_omitted") is True


class TestFromJson:
    """Tests for Schema.from_json()"""

    def test_from_json_deserializes_valid_schema_json(self):
        """from_json should deserialize valid schema JSON into Schema object."""
        json_str = '{"fields": {"col": {"dtype": "int64"}}, "strict": false, "unique": null}'
        schema = Schema.from_json(json_str)
        
        assert isinstance(schema, Schema)
        assert "col" in schema.fields
        assert schema.fields["col"].dtype == "int64"

    def test_from_json_raises_value_error_on_invalid_json(self):
        """from_json should raise ValueError for malformed JSON."""
        with pytest.raises(ValueError, match="Invalid schema JSON"):
            Schema.from_json("{not valid json")

    def test_from_json_raises_type_error_if_not_object(self):
        """from_json should raise TypeError if JSON root is not an object."""
        with pytest.raises(TypeError, match="must decode to an object"):
            Schema.from_json('["array", "not", "object"]')

    def test_from_json_raises_type_error_on_invalid_strict_type(self):
        """from_json should raise TypeError if strict is not boolean."""
        with pytest.raises(TypeError, match="'strict' must be a boolean"):
            Schema.from_json('{"fields": {}, "strict": "not_bool"}')

    def test_from_json_round_trip_preserves_schema(self, simple_schema):
        """from_json should successfully deserialize output from to_json."""
        json_str = simple_schema.to_json()
        restored = Schema.from_json(json_str)
        
        assert restored.strict == simple_schema.strict
        assert restored.unique == simple_schema.unique
        assert set(restored.fields.keys()) == set(simple_schema.fields.keys())
        assert restored.fields["id"].dtype == simple_schema.fields["id"].dtype
        assert restored.fields["id"].nullable == simple_schema.fields["id"].nullable


class TestValidate:
    """Tests for Schema.validate()"""

    def test_validate_returns_validation_result_instance(self, simple_frame, simple_schema):
        """validate should return a ValidationResult object."""
        result = simple_schema.validate(simple_frame)
        assert isinstance(result, ValidationResult)

    def test_validate_result_has_required_attributes(self, simple_frame, simple_schema):
        """validate result should have passed, row_count, and issue_count attributes."""
        result = simple_schema.validate(simple_frame)
        
        assert hasattr(result, "passed")
        assert hasattr(result, "row_count")
        assert hasattr(result, "issue_count")
        assert isinstance(result.passed, bool)

    def test_validate_accepts_max_errors_parameter(self, simple_frame, simple_schema):
        """validate should accept max_errors parameter and return result."""
        result = simple_schema.validate(simple_frame, max_errors=10)
        
        assert isinstance(result, ValidationResult)
        assert result.row_count == 3
