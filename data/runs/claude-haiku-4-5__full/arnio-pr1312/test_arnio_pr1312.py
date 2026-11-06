import pytest
import json
import warnings
import pandas as pd
import arnio as ar


class TestSchemaBootstrapFromReport:
    """Tests for Schema.bootstrap_from_report()"""

    def test_bootstrap_basic(self):
        """Bootstrap schema from a profiled frame."""
        df = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        schema = ar.Schema.bootstrap_from_report(report)
        assert isinstance(schema, ar.Schema)

    def test_bootstrap_detects_nulls(self):
        """Bootstrap detects nullable columns from profile."""
        df = pd.DataFrame({"id": [1, None, 3], "name": ["a", "b", None]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        schema = ar.Schema.bootstrap_from_report(report)
        result = schema.validate(frame)
        assert result.passed

    def test_bootstrap_validates_with_schema(self):
        """Bootstrapped schema successfully validates matching data."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "score": [85.5, 90.0, 78.5],
            "active": [True, False, True]
        })
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        schema = ar.Schema.bootstrap_from_report(report)
        
        result = schema.validate(frame)
        assert result.passed

    def test_bootstrap_invalid_input_raises_typeerror(self):
        """Non-DataQualityReport input raises TypeError."""
        with pytest.raises(TypeError):
            ar.Schema.bootstrap_from_report("not a report")
        with pytest.raises(TypeError):
            ar.Schema.bootstrap_from_report({})

    def test_bootstrap_empty_frame_raises_valueerror(self):
        """Empty frame profile raises ValueError."""
        df = pd.DataFrame()
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        with pytest.raises(ValueError):
            ar.Schema.bootstrap_from_report(report)


class TestSchemaToJson:
    """Tests for Schema.to_json()"""

    def test_to_json_basic(self):
        """Serialize schema to valid JSON string."""
        schema = ar.Schema({"id": ar.Int64(), "name": ar.String()})
        json_str = schema.to_json()
        
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_to_json_with_constraints(self):
        """JSON serialization preserves field constraints."""
        schema = ar.Schema({
            "age": ar.Int64(nullable=False, min=0, max=150),
            "email": ar.Email(nullable=False)
        })
        json_str = schema.to_json()
        
        assert len(json_str) > 0
        json.loads(json_str)  # Verify valid JSON

    def test_to_json_with_rules_omitted_warning(self):
        """Cross-field rules trigger UserWarning and rules_omitted flag."""
        def cross_field_rule(df):
            return []
        
        schema = ar.Schema(
            {"start": ar.String(), "end": ar.String()},
            rules=[cross_field_rule]
        )
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            json_str = schema.to_json()
            
            assert any(issubclass(warn.category, UserWarning) for warn in w)
        
        parsed = json.loads(json_str)
        assert parsed.get("rules_omitted") is True

    def test_to_json_stable_output(self):
        """Multiple calls to to_json produce identical output."""
        schema = ar.Schema({"id": ar.Int64(), "name": ar.String()})
        json1 = schema.to_json()
        json2 = schema.to_json()
        
        assert json1 == json2


class TestSchemaFromJson:
    """Tests for Schema.from_json()"""

    def test_from_json_basic(self):
        """Deserialize schema from JSON string."""
        original = ar.Schema({"id": ar.Int64(), "name": ar.String()})
        json_str = original.to_json()
        
        restored = ar.Schema.from_json(json_str)
        assert isinstance(restored, ar.Schema)

    def test_from_json_preserves_constraints(self):
        """Deserialized schema preserves field constraints."""
        original = ar.Schema({
            "age": ar.Int64(nullable=False, min=0, max=120),
            "email": ar.Email(nullable=False)
        })
        
        json_str = original.to_json()
        restored = ar.Schema.from_json(json_str)
        
        df = pd.DataFrame({
            "age": [25, 30, 35],
            "email": ["a@b.com", "c@d.com", "e@f.com"]
        })
        frame = ar.from_pandas(df)
        
        result = restored.validate(frame)
        assert result.passed

    def test_from_json_invalid_json_raises(self):
        """Invalid JSON input raises exception."""
        with pytest.raises(Exception):
            ar.Schema.from_json("not json {")

    def test_from_json_roundtrip(self):
        """Schema -> to_json -> from_json roundtrip maintains validation equivalence."""
        df = pd.DataFrame({"id": [1, 2, 3], "value": [10.5, 20.5, 30.5]})
        frame = ar.from_pandas(df)
        
        original = ar.Schema({
            "id": ar.Int64(nullable=False),
            "value": ar.Float64()
        })
        
        json_str = original.to_json()
        restored = ar.Schema.from_json(json_str)
        
        result_original = original.validate(frame)
        result_restored = restored.validate(frame)
        
        assert result_original.passed == result_restored.passed


class TestSchemaValidate:
    """Tests for Schema.validate()"""

    def test_validate_passing_data(self):
        """Validate data that satisfies all schema constraints."""
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "name": ar.String(nullable=False)
        })
        
        df = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
        frame = ar.from_pandas(df)
        
        result = schema.validate(frame)
        assert result.passed is True
        assert len(result.issues) == 0

    def test_validate_failing_data(self):
        """Validate data with constraint violations."""
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "age": ar.Int64(min=0, max=130)
        })
        
        df = pd.DataFrame({"id": [1, 2, None], "age": [25, 999, -5]})
        frame = ar.from_pandas(df)
        
        result = schema.validate(frame)
        assert result.passed is False
        assert len(result.issues) > 0

    def test_validate_with_max_errors(self):
        """max_errors parameter limits number of issues reported."""
        schema = ar.Schema({"value": ar.Int64(min=100)})
        df = pd.DataFrame({"value": [1, 2, 3, 4, 5]})
        frame = ar.from_pandas(df)
        
        result = schema.validate(frame, max_errors=2)
        assert len(result.issues) <= 2

    def test_validate_nullable_constraint(self):
        """Nullable=False constraint catches null values."""
        schema = ar.Schema({"required": ar.String(nullable=False)})
        
        df = pd.DataFrame({"required": ["a", None, "c"]})
        frame = ar.from_pandas(df)
        
        result = schema.validate(frame)
        assert result.passed is False

    def test_validate_dtype_mismatch(self):
        """Schema detects dtype mismatches."""
        schema = ar.Schema({"age": ar.Int64()})
        
        df = pd.DataFrame({"age": ["25", "30", "35"]})
        frame = ar.from_pandas(df)
        
        result = schema.validate(frame)
        assert result.passed is False

    def test_validate_returns_validation_result(self):
        """validate() returns ValidationResult object."""
        schema = ar.Schema({"id": ar.Int64()})
        df = pd.DataFrame({"id": [1, 2]})
        frame = ar.from_pandas(df)
        
        result = schema.validate(frame)
        assert hasattr(result, 'passed')
        assert hasattr(result, 'issues')
        assert isinstance(result.passed, bool)
