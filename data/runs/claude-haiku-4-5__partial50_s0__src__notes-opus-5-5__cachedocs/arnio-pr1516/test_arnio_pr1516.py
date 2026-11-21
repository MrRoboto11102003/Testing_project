import pytest
import json
import warnings
import pandas as pd
import numpy as np
import arnio as ar


class TestSchemaToJson:
    """Tests for Schema.to_json()"""

    def test_empty_schema(self):
        """Empty schema serializes with minimal structure"""
        schema = ar.Schema({})
        result = schema.to_json()
        payload = json.loads(result)
        assert payload == {"fields": {}, "strict": False, "unique": None}

    def test_simple_int64_field_has_all_18_keys(self):
        """Field serializes with all 18 keys in sorted order"""
        schema = ar.Schema({"a": ar.Int64()})
        result = schema.to_json()
        payload = json.loads(result)
        field_a = payload["fields"]["a"]
        expected_keys = [
            "allowed", "case_sensitive", "date_max", "date_min",
            "datetime_max", "datetime_min", "dtype", "format",
            "max", "max_length", "min", "min_length", "nullable",
            "pattern", "required_if", "semantic", "severity", "unique"
        ]
        assert list(field_a.keys()) == expected_keys
        assert field_a["dtype"] == "int64"

    def test_field_with_constraints(self):
        """Min/max/length constraints serialize correctly"""
        schema = ar.Schema({
            "age": ar.Int64(min=0, max=150),
            "name": ar.String(min_length=1, max_length=100),
        })
        result = schema.to_json()
        payload = json.loads(result)
        assert payload["fields"]["age"]["min"] == 0
        assert payload["fields"]["age"]["max"] == 150
        assert payload["fields"]["name"]["min_length"] == 1
        assert payload["fields"]["name"]["max_length"] == 100

    def test_allowed_set_converts_to_sorted_list(self):
        """Set of allowed values becomes sorted list"""
        schema = ar.Schema({
            "status": ar.String(allowed={"pending", "active", "inactive"}),
        })
        result = schema.to_json()
        payload = json.loads(result)
        assert payload["fields"]["status"]["allowed"] == ["active", "inactive", "pending"]

    def test_required_if_tuple_converts_to_list(self):
        """required_if tuple becomes list"""
        schema = ar.Schema({
            "country": ar.String(required_if=("user_type", "international")),
        })
        result = schema.to_json()
        payload = json.loads(result)
        assert payload["fields"]["country"]["required_if"] == ["user_type", "international"]

    def test_schema_level_unique_as_list(self):
        """Schema-level unique tuple becomes list"""
        schema = ar.Schema(
            {"id": ar.Int64(), "email": ar.String()},
            unique=("id", "email")
        )
        result = schema.to_json()
        payload = json.loads(result)
        assert payload["unique"] == ["id", "email"]

    def test_cross_field_rules_emit_warning_and_flag(self):
        """Cross-field rules are omitted with warning and rules_omitted flag"""
        def my_rule(df):
            return []
        
        schema = ar.Schema({"a": ar.Int64()}, rules=[my_rule])
        with pytest.warns(UserWarning, match=r"cross-field rules cannot be serialized"):
            result = schema.to_json()
        
        payload = json.loads(result)
        assert payload.get("rules_omitted") is True


class TestSchemaValidate:
    """Tests for Schema.validate()"""

    def test_validate_simple_pass(self):
        """Valid data passes all checks"""
        schema = ar.Schema({"age": ar.Int64()})
        frame = ar.from_pandas(pd.DataFrame({"age": [25, 30, 35]}))
        result = schema.validate(frame)
        assert result.passed
        assert result.issue_count == 0
        assert result.row_count == 3

    def test_validate_dtype_mismatch(self):
        """Dtype mismatch is detected"""
        schema = ar.Schema({"age": ar.Int64()})
        frame = ar.from_pandas(pd.DataFrame({"age": ["25", "30", "35"]}))
        result = schema.validate(frame)
        assert not result.passed
        assert any(issue.rule == "dtype" for issue in result.issues)

    def test_validate_nullable_false_catches_nulls(self):
        """nullable=False rejects rows with null values"""
        schema = ar.Schema({"name": ar.String(nullable=False)})
        frame = ar.from_pandas(pd.DataFrame({"name": ["Alice", None, "Bob"]}))
        result = schema.validate(frame)
        assert not result.passed
        assert any(issue.rule == "nullable" for issue in result.issues)
        assert 2 in result.bad_rows  # row_index is 1-based

    def test_validate_min_max_constraints(self):
        """min and max bounds are enforced"""
        schema = ar.Schema({"score": ar.Int64(min=0, max=100)})
        frame = ar.from_pandas(pd.DataFrame({"score": [50, -5, 150]}))
        result = schema.validate(frame)
        assert not result.passed
        min_issues = [i for i in result.issues if i.rule == "min"]
        max_issues = [i for i in result.issues if i.rule == "max"]
        assert len(min_issues) == 1 and min_issues[0].row_index == 2
        assert len(max_issues) == 1 and max_issues[0].row_index == 3

    def test_validate_max_errors_stops_early(self):
        """max_errors parameter limits issue count"""
        schema = ar.Schema({"age": ar.Int64(min=0)})
        frame = ar.from_pandas(pd.DataFrame({"age": [-1, -2, -3, -4, -5]}))
        result = schema.validate(frame, max_errors=2)
        assert result.issue_count == 2

    def test_validate_composite_unique_constraint(self):
        """Composite unique constraint detects duplicate row pairs"""
        schema = ar.Schema(
            {"user_id": ar.Int64(), "course_id": ar.Int64()},
            unique=["user_id", "course_id"]
        )
        frame = ar.from_pandas(pd.DataFrame({
            "user_id": [1, 1, 2, 2],
            "course_id": [10, 10, 10, 20],
        }))
        result = schema.validate(frame)
        assert not result.passed
        assert 1 in result.bad_rows and 2 in result.bad_rows

    def test_validate_not_arframe_raises_typeerror(self):
        """Non-ArFrame input raises TypeError with helpful message"""
        schema = ar.Schema({"a": ar.Int64()})
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(TypeError, match="single ArFrame"):
            schema.validate(df)

    def test_validate_row_index_is_one_based(self):
        """row_index values start at 1, not 0"""
        schema = ar.Schema({"value": ar.Int64(min=0)})
        frame = ar.from_pandas(pd.DataFrame({"value": [-1]}))
        result = schema.validate(frame)
        assert result.issues[0].row_index == 1
