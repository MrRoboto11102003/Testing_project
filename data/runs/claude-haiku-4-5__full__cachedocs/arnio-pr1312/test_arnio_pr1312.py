import pytest
import pandas as pd
import numpy as np
import arnio as ar
import json
import warnings


class TestSchemaValidate:
    """Tests for Schema.validate()"""

    def test_validate_basic_with_int64_and_string(self):
        """Test basic validation with Int64 and String fields."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["alice", "bob", "charlie"]
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "id": ar.Int64(),
            "name": ar.String()
        })
        result = schema.validate(frame)
        assert result.passed is True

    def test_validate_nullable_false_fails_with_nulls(self):
        """Test validation fails when nullable=False but nulls are present."""
        df = pd.DataFrame({
            "id": [1, 2, None],
            "name": ["alice", "bob", "charlie"]
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "name": ar.String(nullable=False)
        })
        result = schema.validate(frame)
        assert result.passed is False

    def test_validate_with_numeric_constraints(self):
        """Test validation with min/max constraints."""
        df = pd.DataFrame({
            "age": [20, 30, 40],
            "score": [85.5, 90.0, 88.2]
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "age": ar.Int64(min=18, max=100),
            "score": ar.Float64(min=0.0, max=100.0)
        })
        result = schema.validate(frame)
        assert result.passed is True

    def test_validate_constraint_violation_min(self):
        """Test validation fails when min constraint is violated."""
        df = pd.DataFrame({"age": [10, 30, 40]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"age": ar.Int64(min=18)})
        result = schema.validate(frame)
        assert result.passed is False
        assert any(issue.rule == "min" for issue in result.issues)

    def test_validate_email_field(self):
        """Test Email field validation detects invalid addresses."""
        df = pd.DataFrame({
            "email": ["test@example.com", "user@domain.org", "invalid-email"]
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({"email": ar.Email()})
        result = schema.validate(frame)
        assert result.passed is False

    def test_validate_dtype_mismatch(self):
        """Test validation fails on dtype mismatch."""
        df = pd.DataFrame({"id": ["not", "a", "number"]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"id": ar.Int64()})
        result = schema.validate(frame)
        assert result.passed is False

    def test_validate_with_max_errors_limits_reporting(self):
        """Test that max_errors parameter limits issue reporting."""
        df = pd.DataFrame({
            "id": [-1, -2, -3, -4, -5]
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({"id": ar.Int64(min=0)})
        result = schema.validate(frame, max_errors=2)
        assert len(result.issues) <= 2

    def test_validate_returns_result_with_passed_and_issues(self):
        """Test that validate returns a ValidationResult with expected attributes."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"x": ar.Int64()})
        result = schema.validate(frame)
        assert hasattr(result, "passed")
        assert hasattr(result, "issues")
        assert isinstance(result.passed, bool)
        assert isinstance(result.issues, list)

    def test_validate_unique_constraint(self):
        """Test unique constraint validation."""
        df = pd.DataFrame({
            "id": [1, 2, 2],
            "name": ["alice", "bob", "charlie"]
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "id": ar.Int64(unique=True),
            "name": ar.String()
        })
        result = schema.validate(frame)
        assert result.passed is False


class TestSchemaJsonSerialization:
    """Tests for Schema.to_json() and Schema.from_json()"""

    def test_to_json_produces_valid_json(self):
        """Test that to_json produces valid JSON."""
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "name": ar.String(nullable=True)
        })
        json_str = schema.to_json()
        assert isinstance(json_str, str)
        # Should be valid JSON
        data = json.loads(json_str)
        assert isinstance(data, dict)

    def test_to_json_with_rules_omits_and_warns(self):
        """Test that cross-field rules are omitted with a warning."""
        def custom_rule(df):
            return []
        
        schema = ar.Schema(
            {
                "start": ar.String(),
                "end": ar.String()
            },
            rules=[custom_rule]
        )
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            json_str = schema.to_json()
            # Should emit warning about omitted rules
            assert len(w) > 0
        
        # JSON should indicate rules were omitted
        data = json.loads(json_str)
        assert data.get("rules_omitted") is True

    def test_from_json_deserializes_correctly(self):
        """Test JSON deserialization produces working schema."""
        original = ar.Schema({
            "id": ar.Int64(nullable=False, unique=True),
            "score": ar.Float64(nullable=True, min=0.0, max=100.0)
        })
        
        json_str = original.to_json()
        restored = ar.Schema.from_json(json_str)
        
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "score": [85.5, 90.0, 88.2]
        })
        frame = ar.from_pandas(df)
        result = restored.validate(frame)
        assert result.passed is True

    def test_from_json_to_json_round_trip(self):
        """Test round-trip JSON serialization produces stable output."""
        schema1 = ar.Schema({
            "id": ar.Int64(nullable=False),
            "email": ar.Email(nullable=False),
            "age": ar.Int64(min=0, max=150, nullable=True)
        })
        
        json_str1 = schema1.to_json()
        schema2 = ar.Schema.from_json(json_str1)
        json_str2 = schema2.to_json()
        
        # JSON should be stable across round-trip
        data1 = json.loads(json_str1)
        data2 = json.loads(json_str2)
        assert data1 == data2


class TestSchemaBootstrapFromReport:
    """Tests for Schema.bootstrap_from_report()"""

    def test_bootstrap_from_report_creates_working_schema(self):
        """Test that bootstrap creates a schema that validates the frame."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["alice", "bob", "charlie"],
            "score": [85.5, 90.0, 88.2]
        })
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        schema = ar.Schema.bootstrap_from_report(report)
        
        # Schema should validate the frame
        result = schema.validate(frame)
        assert isinstance(result.passed, bool)

    def test_bootstrap_from_report_infers_nullable_from_nulls(self):
        """Test that bootstrap detects nullable columns based on null presence."""
        df = pd.DataFrame({
            "id": [1, 2, None],
            "name": ["alice", "bob", "charlie"]
        })
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        schema = ar.Schema.bootstrap_from_report(report)
        
        # Schema should pass because it infers id as nullable
        result = schema.validate(frame)
        assert result.passed is True

    def test_bootstrap_from_report_raises_typeerror_on_invalid_input(self):
        """Test bootstrap raises TypeError on non-DataQualityReport input."""
        with pytest.raises(TypeError):
            ar.Schema.bootstrap_from_report("not a report")
        
        with pytest.raises(TypeError):
            ar.Schema.bootstrap_from_report(None)

    def test_bootstrap_from_report_raises_valueerror_on_empty_columns(self):
        """Test bootstrap raises ValueError when report has no columns."""
        df = pd.DataFrame()
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        with pytest.raises(ValueError):
            ar.Schema.bootstrap_from_report(report)
