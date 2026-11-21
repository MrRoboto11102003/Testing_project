import pytest
import json
import warnings
import pandas as pd
import numpy as np
import datetime
import arnio as ar
from arnio.schema import Field, Schema


class TestBootstrapFromReport:
    """Tests for Schema.bootstrap_from_report classmethod."""

    def test_bootstrap_basic_dtypes(self):
        """Bootstrap schema infers dtypes correctly from profiled frame."""
        frame = ar.from_pandas(pd.DataFrame({
            "id": [1, 2, 3],
            "value": [1.5, 2.5, 3.5],
            "flag": [True, False, True],
            "name": ["a", "b", "c"]
        }))
        report = ar.profile(frame)
        schema = ar.Schema.bootstrap_from_report(report)

        assert schema.fields["id"].dtype == "int64"
        assert schema.fields["value"].dtype == "float64"
        assert schema.fields["flag"].dtype == "bool"
        assert schema.fields["name"].dtype == "string"

    def test_bootstrap_nullable_from_null_count(self):
        """nullable is True when null_count > 0, False otherwise."""
        frame = ar.from_pandas(pd.DataFrame({
            "nulls": [1, None, 3],
            "no_nulls": [1, 2, 3]
        }))
        report = ar.profile(frame)
        schema = ar.Schema.bootstrap_from_report(report)

        assert schema.fields["nulls"].nullable is True
        assert schema.fields["no_nulls"].nullable is False

    def test_bootstrap_non_report_raises_typeerror(self):
        """TypeError when input is not a DataQualityReport."""
        with pytest.raises(TypeError, match="Expected DataQualityReport, got dict"):
            ar.Schema.bootstrap_from_report({})

        with pytest.raises(TypeError, match="Expected DataQualityReport, got NoneType"):
            ar.Schema.bootstrap_from_report(None)

    def test_bootstrap_empty_report_raises_valueerror(self):
        """ValueError when report has no columns."""
        from arnio.quality import DataQualityReport
        report = DataQualityReport(
            row_count=0, column_count=0, memory_usage=0,
            duplicate_rows=0, duplicate_ratio=0.0, columns={}
        )
        with pytest.raises(ValueError, match="Cannot bootstrap schema from an empty report"):
            ar.Schema.bootstrap_from_report(report)

    def test_bootstrap_missing_dtype_raises_valueerror(self):
        """ValueError when column profile has no dtype."""
        from arnio.quality import DataQualityReport, ColumnProfile
        prof = ColumnProfile(
            name="bad", dtype=None, semantic_type="x", row_count=1,
            null_count=0, null_ratio=0.0, unique_count=1, unique_ratio=1.0
        )
        report = DataQualityReport(
            row_count=1, column_count=1, memory_usage=0,
            duplicate_rows=0, duplicate_ratio=0.0, columns={"bad": prof}
        )
        with pytest.raises(ValueError, match="Column profile for 'bad' is missing 'dtype' key"):
            ar.Schema.bootstrap_from_report(report)

    def test_bootstrap_preserves_column_order(self):
        """Field order matches report column order."""
        frame = ar.from_pandas(pd.DataFrame({
            "z": [1], "a": [2], "m": [3]
        }))
        report = ar.profile(frame)
        schema = ar.Schema.bootstrap_from_report(report)

        assert list(schema.fields.keys()) == list(report.columns.keys())


class TestFromJson:
    """Tests for Schema.from_json classmethod."""

    def test_from_json_basic(self):
        """Deserialize basic schema from JSON."""
        json_str = '{"fields": {"a": {"dtype": "int64"}}, "strict": false, "unique": null}'
        schema = ar.Schema.from_json(json_str)

        assert schema.fields["a"].dtype == "int64"
        assert schema.strict is False
        assert schema.unique is None

    def test_from_json_invalid_json_raises_valueerror(self):
        """ValueError for invalid JSON syntax."""
        with pytest.raises(ValueError, match="Invalid schema JSON"):
            ar.Schema.from_json("not json")

    def test_from_json_non_object_raises_typeerror(self):
        """TypeError when JSON does not deserialize to an object."""
        with pytest.raises(TypeError, match="Schema JSON must decode to an object"):
            ar.Schema.from_json("[]")

    def test_from_json_missing_fields_raises_typeerror(self):
        """TypeError when fields key missing or not an object."""
        with pytest.raises(TypeError, match="Schema JSON 'fields' must be an object"):
            ar.Schema.from_json('{"strict": false}')

    def test_from_json_unknown_key_raises_valueerror(self):
        """ValueError for unknown top-level keys."""
        with pytest.raises(ValueError, match="Schema JSON contains unknown key"):
            ar.Schema.from_json('{"fields": {}, "bad_key": true}')

    def test_from_json_field_conversions(self):
        """Field list values are converted to sets and tuples."""
        json_str = '''
        {
            "fields": {
                "a": {"allowed": ["z", "a"], "required_if": ["b", 42]}
            },
            "strict": false, "unique": null
        }
        '''
        schema = ar.Schema.from_json(json_str)

        assert isinstance(schema.fields["a"].allowed, set)
        assert schema.fields["a"].allowed == {"z", "a"}
        assert isinstance(schema.fields["a"].required_if, tuple)
        assert schema.fields["a"].required_if == ("b", 42)

    def test_from_json_round_trip(self):
        """Round-trip: schema -> to_json -> from_json preserves structure."""
        original = ar.Schema({
            "id": ar.Int64(nullable=False, min=1, max=100),
            "name": ar.String(nullable=False, min_length=1, max_length=50)
        }, strict=True, unique=["id"])

        json_str = original.to_json()
        restored = ar.Schema.from_json(json_str)

        assert restored.fields["id"].dtype == original.fields["id"].dtype
        assert restored.fields["id"].min == original.fields["id"].min
        assert restored.fields["name"].max_length == original.fields["name"].max_length
        assert restored.strict == original.strict
        assert list(restored.unique) == list(original.unique)


class TestToJson:
    """Tests for Schema.to_json method."""

    def test_to_json_basic_serialization(self):
        """Serialize schema to valid JSON."""
        schema = ar.Schema({"a": ar.Int64()})
        json_str = schema.to_json()

        payload = json.loads(json_str)
        assert "fields" in payload
        assert "a" in payload["fields"]
        assert payload["fields"]["a"]["dtype"] == "int64"

    def test_to_json_deterministic_key_order(self):
        """JSON keys are sorted for determinism."""
        schema = ar.Schema({"z": ar.Int64(), "a": ar.Float64()})
        json_str = schema.to_json()

        payload = json.loads(json_str)
        field_keys = list(payload["fields"]["a"].keys())
        assert field_keys == sorted(field_keys)

    def test_to_json_warns_when_rules_present(self):
        """UserWarning emitted when rules are present."""
        def dummy_rule(df):
            return []

        schema = ar.Schema({"a": ar.Int64()}, rules=[dummy_rule])

        with pytest.warns(UserWarning, match="cross-field rules cannot be serialized"):
            json_str = schema.to_json()

        payload = json.loads(json_str)
        assert payload.get("rules_omitted") is True

    def test_to_json_no_warning_without_rules(self):
        """No warning when rules absent."""
        schema = ar.Schema({"a": ar.Int64()})

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            json_str = schema.to_json()

        payload = json.loads(json_str)
        assert "rules_omitted" not in payload


class TestValidate:
    """Tests for Schema.validate method."""

    def test_validate_passing_frame(self):
        """Validation passes for compliant frame."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
        schema = ar.Schema({"a": ar.Int64()})
        result = schema.validate(frame)

        assert result.passed is True
        assert result.issue_count == 0

    def test_validate_dtype_mismatch(self):
        """Validation detects dtype mismatches."""
        frame = ar.from_pandas(pd.DataFrame({"a": ["1", "2", "3"]}))
        schema = ar.Schema({"a": ar.Int64()})
        result = schema.validate(frame)

        assert result.passed is False
        assert any(issue.rule == "dtype" for issue in result.issues)

    def test_validate_nullable_violation(self):
        """Validation detects null values when nullable=False."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, None, 3]}))
        schema = ar.Schema({"a": ar.Int64(nullable=False)})
        result = schema.validate(frame)

        assert result.passed is False
        assert any(issue.rule == "nullable" for issue in result.issues)

    def test_validate_row_index_is_one_based(self):
        """row_index values are 1-based, not 0-based."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 100, 3]}))
        schema = ar.Schema({"a": ar.Int64(max=50)})
        result = schema.validate(frame)

        # Second row (pandas index 1) should be row_index 2
        assert any(issue.row_index == 2 for issue in result.issues)

    def test_validate_non_arframe_raises_typeerror(self):
        """TypeError when frame is not an ArFrame."""
        df = pd.DataFrame({"a": [1, 2]})
        schema = ar.Schema({"a": ar.Int64()})

        with pytest.raises(TypeError, match="ArFrame"):
            schema.validate(df)

    def test_validate_max_errors_limit(self):
        """Validation stops and returns at max_errors limit."""
        frame = ar.from_pandas(pd.DataFrame({"a": [100, 200, 300]}))
        schema = ar.Schema({"a": ar.Int64(max=50)})
        result = schema.validate(frame, max_errors=1)

        assert result.issue_count == 1

    def test_validate_max_errors_invalid_raises_valueerror(self):
        """ValueError for max_errors <= 0."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))
        schema = ar.Schema({"a": ar.Int64()})

        with pytest.raises(ValueError, match="max_errors must be >= 1"):
            schema.validate(frame, max_errors=0)

    def test_validate_allowed_values(self):
        """Validation checks allowed value constraints."""
        frame = ar.from_pandas(pd.DataFrame({"status": ["A", "B", "C"]}))
        schema = ar.Schema({"status": ar.String(allowed={"A", "B"})})
        result = schema.validate(frame)

        assert result.passed is False
        # Row with "C" should be reported
        assert len([i for i in result.issues if i.row_index == 3]) > 0
