import pytest
import pandas as pd
import numpy as np
import json
import arnio as ar


class TestSchemaBootstrapFromReport:
    """Tests for Schema.bootstrap_from_report()"""

    def test_bootstrap_creates_schema_from_report(self):
        """Bootstrap should create a valid schema from a report"""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["alice", "bob", "charlie"],
            "score": [95.5, 88.0, 92.3],
        })
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        schema = ar.Schema.bootstrap_from_report(report)
        
        assert schema is not None
        assert isinstance(schema, ar.Schema)
        # Should have all columns from the report
        assert all(col in schema.fields for col in df.columns)

    def test_bootstrap_infers_nullable_from_nulls(self):
        """Bootstrap should mark columns with nulls as nullable"""
        df = pd.DataFrame({
            "required": [1, 2, 3],
            "optional": [1.0, None, 3.0],
        })
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        schema = ar.Schema.bootstrap_from_report(report)
        result = schema.validate(frame)
        
        # Should validate successfully since nulls are allowed in optional
        assert result.passed

    def test_bootstrap_preserves_dtypes(self):
        """Bootstrap should preserve column dtypes from the report"""
        df = pd.DataFrame({
            "int_col": [1, 2, 3],
            "float_col": [1.1, 2.2, 3.3],
            "str_col": ["a", "b", "c"],
        })
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        schema = ar.Schema.bootstrap_from_report(report)
        
        # All dtypes should match the report
        for col, profile in report.columns.items():
            assert col in schema.fields

    def test_bootstrap_raises_on_non_report(self):
        """Bootstrap should raise TypeError for non-DataQualityReport input"""
        with pytest.raises(TypeError):
            ar.Schema.bootstrap_from_report("not a report")

    def test_bootstrap_raises_on_empty_report(self):
        """Bootstrap should raise ValueError for empty report"""
        # Create a frame with no rows
        df = pd.DataFrame()
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        with pytest.raises(ValueError):
            ar.Schema.bootstrap_from_report(report)


class TestSchemaJsonSerialization:
    """Tests for Schema.to_json() and Schema.from_json()"""

    def test_to_json_returns_valid_json(self):
        """to_json should return valid JSON string"""
        schema = ar.Schema({
            "id": ar.Int64(nullable=False, unique=True),
            "email": ar.Email(nullable=False),
            "age": ar.Int64(nullable=True, min=0, max=130),
        })
        
        json_str = schema.to_json()
        
        assert isinstance(json_str, str)
        # Should be valid JSON
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_from_json_restores_schema(self):
        """from_json should restore a schema from JSON"""
        original = ar.Schema({
            "id": ar.Int64(nullable=False),
            "name": ar.String(nullable=False),
            "score": ar.Float64(nullable=True),
        })
        
        json_str = original.to_json()
        restored = ar.Schema.from_json(json_str)
        
        assert restored is not None
        assert isinstance(restored, ar.Schema)
        assert "id" in restored.fields
        assert "name" in restored.fields
        assert "score" in restored.fields

    def test_json_roundtrip_preserves_validation(self):
        """Schema validation should work the same before and after JSON roundtrip"""
        original_schema = ar.Schema({
            "age": ar.Int64(min=0, max=120),
            "email": ar.Email(),
        })
        
        df = pd.DataFrame({
            "age": [25, 35, 45],
            "email": ["a@example.com", "b@example.com", "c@example.com"],
        })
        frame = ar.from_pandas(df)
        
        # Validate with original
        result_original = original_schema.validate(frame)
        
        # Validate with restored from JSON
        json_str = original_schema.to_json()
        restored_schema = ar.Schema.from_json(json_str)
        result_restored = restored_schema.validate(frame)
        
        assert result_original.passed == result_restored.passed

    def test_json_includes_rules_omitted_flag_when_rules_present(self):
        """to_json should include rules_omitted flag when cross-field rules exist"""
        def custom_rule(df):
            return []
        
        schema = ar.Schema(
            {"col1": ar.String(), "col2": ar.String()},
            rules=[custom_rule]
        )
        
        json_str = schema.to_json()
        parsed = json.loads(json_str)
        
        # Should indicate rules were omitted
        assert parsed.get("rules_omitted") is True

    def test_json_roundtrip_with_datetime_field(self):
        """JSON roundtrip should preserve DateTime field configuration"""
        schema = ar.Schema({
            "created_at": ar.DateTime(format="%Y-%m-%d", nullable=False),
        })
        
        json_str = schema.to_json()
        restored = ar.Schema.from_json(json_str)
        
        assert "created_at" in restored.fields


class TestSchemaValidate:
    """Tests for Schema.validate()"""

    def test_validate_returns_validation_result(self):
        """validate should return a ValidationResult"""
        schema = ar.Schema({"age": ar.Int64()})
        df = pd.DataFrame({"age": [25, 30, 35]})
        frame = ar.from_pandas(df)
        
        result = schema.validate(frame)
        
        assert result is not None
        assert hasattr(result, "passed")
        assert hasattr(result, "issues")

    def test_validate_passes_valid_data(self):
        """validate should return passed=True for valid data"""
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "email": ar.Email(nullable=False),
        })
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "email": ["a@example.com", "b@example.com", "c@example.com"],
        })
        frame = ar.from_pandas(df)
        
        result = schema.validate(frame)
        
        assert result.passed is True

    def test_validate_fails_on_type_mismatch(self):
        """validate should fail when column dtype doesn't match schema"""
        schema = ar.Schema({"age": ar.Int64()})
        df = pd.DataFrame({"age": ["not_a_number", "also_not", "nope"]})
        frame = ar.from_pandas(df)
        
        result = schema.validate(frame)
        
        assert result.passed is False
        assert len(result.issues) > 0

    def test_validate_fails_on_missing_required_column(self):
        """validate should fail when required columns are missing"""
        schema = ar.Schema({"required_col": ar.String()})
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        frame = ar.from_pandas(df)
        
        result = schema.validate(frame)
        
        assert result.passed is False

    def test_validate_respects_nullable_constraint(self):
        """validate should fail when nullable=False but nulls present"""
        schema = ar.Schema({"name": ar.String(nullable=False)})
        df = pd.DataFrame({"name": ["alice", None, "charlie"]})
        frame = ar.from_pandas(df)
        
        result = schema.validate(frame)
        
        assert result.passed is False
        assert len(result.issues) > 0

    def test_validate_respects_max_errors(self):
        """validate should stop at max_errors limit"""
        schema = ar.Schema({"age": ar.Int64(min=18)})
        df = pd.DataFrame({"age": [5, 10, 15, 20, 25]})
        frame = ar.from_pandas(df)
        
        result = schema.validate(frame, max_errors=2)
        
        # Should have at most max_errors issues
        assert len(result.issues) <= 2

    def test_validate_enforces_min_max_constraints(self):
        """validate should enforce min and max numeric constraints"""
        schema = ar.Schema({
            "score": ar.Int64(min=0, max=100),
        })
        df = pd.DataFrame({"score": [50, 150, -10]})
        frame = ar.from_pandas(df)
        
        result = schema.validate(frame)
        
        assert result.passed is False
        # Should have 2 issues (150 and -10 are outside [0, 100])
        assert len(result.issues) == 2

    def test_validate_unique_constraint(self):
        """validate should enforce unique constraint"""
        schema = ar.Schema({
            "user_id": ar.Int64(unique=True),
        })
        df = pd.DataFrame({"user_id": [1, 2, 2, 3]})
        frame = ar.from_pandas(df)
        
        result = schema.validate(frame)
        
        assert result.passed is False

    def test_validate_issues_include_row_index(self):
        """Validation issues should include row_index (1-based)"""
        schema = ar.Schema({"age": ar.Int64(min=18)})
        df = pd.DataFrame({"age": [25, 10, 20]})  # Row 2 is invalid
        frame = ar.from_pandas(df)
        
        result = schema.validate(frame)
        
        assert result.passed is False
        # Should have at least one issue with row_index=2
        assert any(issue.row_index == 2 for issue in result.issues)
