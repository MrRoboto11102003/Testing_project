import pytest
import json
import warnings
import pandas as pd
import arnio as ar


class TestBootstrapFromReport:
    """Tests for Schema.bootstrap_from_report()"""

    def test_bootstrap_creates_schema_from_valid_report(self):
        """Should create a Schema from a valid DataQualityReport."""
        frame = ar.from_pandas(pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
        }))
        report = ar.profile(frame)
        
        schema = ar.Schema.bootstrap_from_report(report)
        
        assert "id" in schema.fields
        assert "name" in schema.fields

    def test_bootstrap_sets_nullable_based_on_nulls(self):
        """Should set nullable=True when null_count > 0, False otherwise."""
        frame = ar.from_pandas(pd.DataFrame({
            "no_nulls": [1, 2, 3],
            "has_nulls": [1.0, None, 3.0],
        }))
        report = ar.profile(frame)
        
        schema = ar.Schema.bootstrap_from_report(report)
        
        assert schema.fields["no_nulls"].nullable is False
        assert schema.fields["has_nulls"].nullable is True

    def test_bootstrap_infers_correct_dtypes(self):
        """Should infer column dtypes from profile."""
        frame = ar.from_pandas(pd.DataFrame({
            "int_col": [1, 2, 3],
            "float_col": [1.5, 2.5, 3.5],
            "str_col": ["a", "b", "c"],
            "bool_col": [True, False, True],
        }))
        report = ar.profile(frame)
        
        schema = ar.Schema.bootstrap_from_report(report)
        
        assert schema.fields["int_col"].dtype == "int64"
        assert schema.fields["float_col"].dtype == "float64"
        assert schema.fields["str_col"].dtype == "string"
        assert schema.fields["bool_col"].dtype == "bool"

    def test_bootstrap_raises_typeerror_on_invalid_input(self):
        """Should raise TypeError if input is not a DataQualityReport."""
        with pytest.raises(TypeError, match="Expected DataQualityReport"):
            ar.Schema.bootstrap_from_report("not a report")

    def test_bootstrap_raises_valueerror_on_empty_report(self):
        """Should raise ValueError if report has no columns."""
        class MockEmptyReport:
            columns = {}
        
        with pytest.raises(ValueError, match="empty report"):
            ar.Schema.bootstrap_from_report(MockEmptyReport())


class TestToJsonFromJson:
    """Tests for Schema.to_json() and Schema.from_json()"""

    def test_to_json_produces_valid_json_structure(self):
        """Should produce JSON with fields, strict, and unique keys."""
        schema = ar.Schema({
            "id": ar.Int64(),
            "name": ar.String(),
        })
        
        json_str = schema.to_json()
        
        payload = json.loads(json_str)
        assert "fields" in payload
        assert "strict" in payload
        assert isinstance(payload["fields"], dict)
        assert isinstance(payload["strict"], bool)

    def test_from_json_restores_schema_with_nullable(self):
        """Should deserialize schema preserving nullable property."""
        original = ar.Schema({
            "id": ar.Int64(nullable=False),
            "name": ar.String(nullable=True),
        })
        
        json_str = original.to_json()
        restored = ar.Schema.from_json(json_str)
        
        assert restored.fields["id"].nullable is False
        assert restored.fields["name"].nullable is True

    def test_json_roundtrip_preserves_strict_and_unique(self):
        """Should preserve strict and unique constraints through roundtrip."""
        original = ar.Schema(
            {"col1": ar.String(), "col2": ar.String()},
            strict=True,
            unique=["col1"],
        )
        
        json_str = original.to_json()
        restored = ar.Schema.from_json(json_str)
        
        assert restored.strict == True
        assert restored.unique == original.unique

    def test_to_json_omits_rules_and_warns(self):
        """Should omit cross-field rules and emit UserWarning."""
        def custom_rule(df):
            return []
        
        schema = ar.Schema(
            {"col": ar.String()},
            rules=[custom_rule],
        )
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            json_str = schema.to_json()
            
            assert any(issubclass(warning.category, UserWarning) for warning in w)
        
        payload = json.loads(json_str)
        assert payload.get("rules_omitted") is True

    def test_from_json_raises_valueerror_on_invalid_json(self):
        """Should raise ValueError for malformed JSON."""
        with pytest.raises(ValueError, match="Invalid schema JSON"):
            ar.Schema.from_json("{not valid json")

    def test_from_json_raises_typeerror_on_non_dict_json(self):
        """Should raise TypeError if JSON doesn't decode to a dict."""
        with pytest.raises(TypeError, match="must decode to an object"):
            ar.Schema.from_json('["array", "not", "dict"]')

    def test_from_json_raises_on_unknown_keys(self):
        """Should raise ValueError on unknown schema keys."""
        bad_json = json.dumps({
            "fields": {"col": {"dtype": "int64"}},
            "strict": True,
            "unknown_key": "value",
        })
        
        with pytest.raises(ValueError, match="unknown key"):
            ar.Schema.from_json(bad_json)

    def test_from_json_accepts_rules_omitted_marker(self):
        """Should silently accept rules_omitted=true marker."""
        json_str = json.dumps({
            "fields": {"col": {"dtype": "int64"}},
            "strict": False,
            "rules_omitted": True,
        })
        
        # Should not raise
        schema = ar.Schema.from_json(json_str)
        assert "col" in schema.fields


class TestValidate:
    """Tests for Schema.validate()"""

    def test_validate_valid_frame_returns_passed_true(self):
        """Should return ValidationResult with passed=True for valid data."""
        frame = ar.from_pandas(pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
        }))
        schema = ar.Schema({
            "id": ar.Int64(),
            "name": ar.String(),
        })
        
        result = schema.validate(frame)
        
        assert result.passed is True
        assert len(result.issues) == 0

    def test_validate_dtype_mismatch_fails(self):
        """Should fail validation when column dtype doesn't match schema."""
        frame = ar.from_pandas(pd.DataFrame({
            "age": ["not_int", "also_not", "nope"],
        }))
        schema = ar.Schema({
            "age": ar.Int64(),
        })
        
        result = schema.validate(frame)
        
        assert result.passed is False
        assert len(result.issues) > 0

    def test_validate_respects_max_errors(self):
        """Should stop validation at max_errors limit."""
        frame = ar.from_pandas(pd.DataFrame({
            "val": ["a", "b", "c", "d"],
        }))
        schema = ar.Schema({
            "val": ar.Int64(),
        })
        
        result = schema.validate(frame, max_errors=2)
        
        assert len(result.issues) <= 2

    def test_validate_nullable_false_constraint(self):
        """Should fail when nullable=False but column contains nulls."""
        frame = ar.from_pandas(pd.DataFrame({
            "id": [1.0, None, 3.0],
        }))
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
        })
        
        result = schema.validate(frame)
        
        assert result.passed is False

    def test_validate_missing_column_fails(self):
        """Should fail when required column is missing from frame."""
        frame = ar.from_pandas(pd.DataFrame({
            "name": ["Alice"],
        }))
        schema = ar.Schema({
            "id": ar.Int64(),
            "name": ar.String(),
        })
        
        result = schema.validate(frame)
        
        assert result.passed is False
