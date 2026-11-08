import pytest
import json
from decimal import Decimal
import numpy as np
import pandas as pd

import arnio as ar


@pytest.fixture
def sample_report():
    """Create a mock DataQualityReport for testing."""
    report = ar.DataQualityReport()
    report.columns = {
        "id": ar.ColumnProfile(dtype="int64", null_count=0),
        "name": ar.ColumnProfile(dtype="string", null_count=2),
        "score": ar.ColumnProfile(dtype="float64", null_count=0),
    }
    return report


@pytest.fixture
def valid_json_schema():
    """Create a valid JSON schema string."""
    return json.dumps({
        "fields": {
            "id": {"dtype": "int64"},
            "name": {"dtype": "string", "nullable": True},
        },
        "strict": False,
        "unique": None,
    })


@pytest.fixture
def invalid_json_schema():
    """Create an invalid JSON schema string."""
    return '{"fields": {"id": {"dtype": "int64"}}, "strict": 123}'


class TestSchema:
    """Test Suite for arnio.schema.Schema functions."""

    def test_bootstrap_from_report_success(self, sample_report):
        """Test successful schema creation from a valid report."""
        schema = ar.Schema.bootstrap_from_report(sample_report)
        assert isinstance(schema, ar.Schema)
        assert "id" in schema.fields
        assert "name" in schema.fields
        assert schema.fields["id"].dtype == "int64"
        assert schema.fields["name"].nullable is True

    def test_bootstrap_from_report_type_error(self):
        """Test TypeError when input is not a DataQualityReport."""
        with pytest.raises(TypeError, match="Expected DataQualityReport"):
            ar.Schema.bootstrap_from_report("not_a_report")

    def test_bootstrap_from_report_empty_report(self):
        """Test ValueError when report has no columns."""
        empty_report = ar.DataQualityReport()
        with pytest.raises(ValueError, match="empty report"):
            ar.Schema.bootstrap_from_report(empty_report)

    def test_bootstrap_from_report_missing_dtype(self):
        """Test ValueError when column profile is missing dtype."""
        incomplete_report = ar.DataQualityReport()
        incomplete_report.columns = {
            "id": ar.ColumnProfile(null_count=0),  # Missing dtype
        }
        with pytest.raises(ValueError, match="missing 'dtype' key"):
            ar.Schema.bootstrap_from_report(incomplete_report)

    def test_bootstrap_from_report_null_count_inference(self):
        """Test that nullable is inferred from null_count."""
        report = ar.DataQualityReport()
        report.columns = {
            "id": ar.ColumnProfile(dtype="int64", null_count=5),
        }
        schema = ar.Schema.bootstrap_from_report(report)
        assert schema.fields["id"].nullable is True

    def test_from_json_valid(self, valid_json_schema):
        """Test successful deserialization of valid JSON."""
        schema = ar.Schema.from_json(valid_json_schema)
        assert isinstance(schema, ar.Schema)
        assert "fields" in schema.fields
        assert schema.fields["id"].dtype == "int64"
        assert schema.strict is False

    def test_from_json_invalid_json(self, invalid_json_schema):
        """Test ValueError when JSON cannot be parsed."""
        with pytest.raises(ValueError, match="Invalid schema JSON"):
            ar.Schema.from_json(invalid_json_schema)

    def test_from_json_not_dict(self):
        """Test TypeError when JSON decodes to non-dict."""
        invalid_payload = "123"  # Decodes to int, not dict
        with pytest.raises(TypeError, match="object with 'fields'"):
            ar.Schema.from_json(invalid_payload)

    def test_from_json_unknown_keys(self):
        """Test ValueError when JSON contains unknown keys."""
        invalid_payload = {"extra_field": "value"}
        with pytest.raises(ValueError, match="unknown key"):
            ar.Schema.from_json(json.dumps(invalid_payload))

    def test_from_json_fields_not_dict(self):
        """Test TypeError when fields is not a dict."""
        invalid_payload = {"fields": ["list", "not", "dict"]}
        with pytest.raises(TypeError, match="object mapping names"):
            ar.Schema.from_json(json.dumps(invalid_payload))

    def test_from_json_strict_not_bool(self):
        """Test TypeError when strict is not a boolean."""
        invalid_payload = {"fields": {}, "strict": True}
        with pytest.raises(TypeError, match="must be a boolean"):
            ar.Schema.from_json(json.dumps(invalid_payload))

    def test_from_json_unique_not_list(self):
        """Test TypeError when unique is not a list or None."""
        invalid_payload = {"fields": {}, "unique": "string"}
        with pytest.raises(TypeError, match="must be a list"):
            ar.Schema.from_json(json.dumps(invalid_payload))

    def test_from_json_round_trip(self):
        """Test that from_json after to_json preserves structure."""
        original_schema = ar.Schema({
            "id": ar.Field(dtype="int64", nullable=False),
            "name": ar.Field(dtype="string", nullable=True),
        })
        json_str = original_schema.to_json()
        restored_schema = ar.Schema.from_json(json_str)

        assert isinstance(restored_schema, ar.Schema)
        assert "id" in restored_schema.fields
        assert "name" in restored_schema.fields
        assert restored_schema.fields["id"].dtype == "int64"
        assert restored_schema.fields["name"].nullable is True

    def test_from_json_missing_required_keys(self):
        """Test that required keys are validated."""
        # Missing 'fields' key
        with pytest.raises(TypeError, match="object with 'fields'"):
            ar.Schema.from_json(json.dumps({"strict": False}))

    def test_to_json_includes_rules_omitted_flag(self):
        """Test that rules_omitted flag is included in JSON output."""
        schema = ar.Schema({
            "id": ar.Field(dtype="int64", nullable=False),
        })
        schema.rules = [("rule1", {"key": "value"})]  # Add some rules
        json_str = schema.to_json()
        data = json.loads(json_str)
        assert data.get("rules_omitted") is True

    def test_to_json_excludes_rules(self):
        """Test that rules are silently omitted from JSON."""
        schema = ar.Schema({
            "id": ar.Field(dtype="int64", nullable=False),
        })
        schema.rules = [("rule1", {"key": "value"})]
        json_str = schema.to_json()
        data = json.loads(json_str)
        assert "rules" not in data

    def test_to_json_warning_on_rules(self):
        """Test that warning is emitted when rules are omitted."""
        import warnings

        schema = ar.Schema({
            "id": ar.Field(dtype="int64", nullable=False),
        })
        schema.rules = [("rule1", {"key": "value"})]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            json_str = schema.to_json()
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert "rules cannot be serialized" in str(w[0].message)

    def test_validate_success(self):
        """Test successful validation of a frame against a schema."""
        frame = ar.from_pandas(pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
        }))
        schema = ar.Schema({
            "id": ar.Field(dtype="int64", nullable=False),
            "name": ar.Field(dtype="string", nullable=True),
        })
        result = schema.validate(frame)
        assert isinstance(result, ar.ValidationResult)
        assert result.passed is True

    def test_validate_invalid_dtype(self):
        """Test validation failure for invalid dtype."""
        frame = ar.from_pandas(pd.DataFrame({
            "id": ["1", "2", "3"],  # string instead of int64
            "name": ["Alice", "Bob", "Charlie"],
        }))
        schema = ar.Schema({
            "id": ar.Field(dtype="int64", nullable=False),
            "name": ar.Field(dtype="string", nullable=True),
        })
        result = schema.validate(frame)
        assert isinstance(result, ar.ValidationResult)
        assert not result.passed
        assert len(result.issues) > 0

    def test_validate_max_errors(self):
        """Test that validation stops after max_errors limit."""
        frame = ar.from_pandas(pd.DataFrame({
            "id": ["1", "2", "3"],
            "name": ["Alice", "Bob", "Charlie"],
        }))
        schema = ar.Schema({
            "id": ar.Field(dtype="int64", nullable=False),
            "name": ar.Field(dtype="string", nullable=True),
        })
        result = schema.validate(frame, max_errors=2)
        assert len(result.issues) <= 2

    def test_validate_with_custom_schema(self):
        """Test validation using a schema created from JSON."""
        json_str = json.dumps({
            "fields": {
                "id": {"dtype": "int64", "nullable": False},
                "name": {"dtype": "string", "nullable": True},
            },
            "strict": True,
        })
        schema = ar.Schema.from_json(json_str)

        frame = ar.from_pandas(pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
        }))
        result = schema.validate(frame)
        assert isinstance(result, ar.ValidationResult)
