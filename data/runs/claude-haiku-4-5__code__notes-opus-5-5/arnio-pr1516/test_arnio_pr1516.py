import pytest
import json
import warnings
import pandas as pd
import numpy as np
import arnio as ar


class TestSchemaToJson:
    """Tests for Schema.to_json() method."""

    def test_empty_schema_to_json(self):
        """Empty schema produces minimal JSON."""
        schema = ar.Schema({})
        result = schema.to_json()
        payload = json.loads(result)
        assert payload == {"fields": {}, "strict": False, "unique": None}

    def test_single_field_complete_structure(self):
        """Single Int64 field has all 18 required keys."""
        schema = ar.Schema({"a": ar.Int64()})
        result = schema.to_json()
        payload = json.loads(result)
        field_dict = payload["fields"]["a"]
        
        expected_keys = {
            "allowed", "case_sensitive", "date_max", "date_min",
            "datetime_max", "datetime_min", "dtype", "format",
            "max", "max_length", "min", "min_length", "nullable",
            "pattern", "required_if", "semantic", "severity", "unique"
        }
        assert set(field_dict.keys()) == expected_keys
        assert field_dict["dtype"] == "int64"

    def test_fields_sorted_alphabetically(self):
        """Multiple fields are sorted alphabetically in output."""
        schema = ar.Schema({"z": ar.Int64(), "a": ar.String(), "m": ar.Float64()})
        result = schema.to_json()
        payload = json.loads(result)
        assert list(payload["fields"].keys()) == ["a", "m", "z"]

    def test_rules_trigger_warning_and_flag(self):
        """Schema with rules emits warning and sets rules_omitted flag."""
        schema = ar.Schema(
            {"a": ar.Int64()},
            rules=[lambda df: []]
        )
        with pytest.warns(UserWarning, match="cross-field rules cannot be serialized"):
            result = schema.to_json()
        
        payload = json.loads(result)
        assert payload.get("rules_omitted") is True

    def test_no_rules_no_warning(self):
        """Schema without rules does not emit warning."""
        schema = ar.Schema({"a": ar.Int64()})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = schema.to_json()
        
        payload = json.loads(result)
        assert "rules_omitted" not in payload

    def test_unique_tuple_serialization(self):
        """Unique constraint serializes as list, preserving order."""
        schema = ar.Schema(
            {"a": ar.Int64(), "b": ar.Int64()},
            unique=("b", "a")
        )
        result = schema.to_json()
        payload = json.loads(result)
        assert payload["unique"] == ["b", "a"]

    def test_allowed_set_becomes_sorted_list(self):
        """Allowed values set is converted to sorted list."""
        schema = ar.Schema({"a": ar.Int64(allowed={5, 1, 3})})
        result = schema.to_json()
        payload = json.loads(result)
        assert payload["fields"]["a"]["allowed"] == [1, 3, 5]

    def test_non_serializable_raises_type_error(self):
        """Non-JSON-serializable field value raises TypeError."""
        import datetime
        schema = ar.Schema({
            "d": ar.String(allowed={datetime.date(2024, 1, 1)})
        })
        with pytest.raises(TypeError, match="is not JSON serializable"):
            schema.to_json()


class TestSchemaValidate:
    """Tests for Schema.validate() method."""

    def test_valid_frame_passes(self):
        """Valid frame produces no errors."""
        schema = ar.Schema({"a": ar.Int64()})
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
        result = schema.validate(frame)
        
        assert result.passed is True
        assert result.issue_count == 0

    def test_dtype_mismatch_reported(self):
        """Dtype mismatch creates error issue."""
        schema = ar.Schema({"a": ar.Int64()})
        frame = ar.from_pandas(pd.DataFrame({"a": ["x", "y", "z"]}))
        result = schema.validate(frame)
        
        assert not result.passed
        dtype_issues = [i for i in result.issues if i.rule == "dtype"]
        assert len(dtype_issues) > 0

    def test_nullable_violation_reported(self):
        """Null in non-nullable column creates issue."""
        schema = ar.Schema({"a": ar.Int64(nullable=False)})
        frame = ar.from_pandas(pd.DataFrame({"a": [1, None, 3]}))
        result = schema.validate(frame)
        
        nullable_issues = [i for i in result.issues if i.rule == "nullable"]
        assert len(nullable_issues) > 0
        assert not result.passed

    def test_min_max_bounds_checked(self):
        """Min and max bounds are enforced."""
        schema = ar.Schema({"a": ar.Int64(min=2, max=5)})
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3, 5, 10]}))
        result = schema.validate(frame)
        
        min_issues = [i for i in result.issues if i.rule == "min"]
        max_issues = [i for i in result.issues if i.rule == "max"]
        assert len(min_issues) > 0
        assert len(max_issues) > 0

    def test_max_errors_limits_issues(self):
        """max_errors parameter stops validation after limit."""
        schema = ar.Schema({"a": ar.Int64(min=10)})
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3, 4, 5]}))
        result = schema.validate(frame, max_errors=2)
        
        assert result.issue_count == 2

    def test_strict_mode_catches_extra_columns(self):
        """Strict mode reports unexpected columns."""
        schema = ar.Schema({"a": ar.Int64()}, strict=True)
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))
        result = schema.validate(frame)
        
        unexpected = [i for i in result.issues if i.rule == "unexpected_column"]
        assert len(unexpected) == 1
        assert not result.passed

    def test_non_arframe_raises_type_error(self):
        """Passing pandas DataFrame instead of ArFrame raises TypeError."""
        schema = ar.Schema({"a": ar.Int64()})
        with pytest.raises(TypeError, match="ArFrame"):
            schema.validate(pd.DataFrame({"a": [1, 2, 3]}))
