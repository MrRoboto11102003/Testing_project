"""
System (end-to-end) tests for arnio schema validation workflows.

Tests cover:
- Schema.bootstrap_from_report
- Schema.to_json / Schema.from_json
- Schema.validate

Each test drives complete user workflows through real files, loading,
profiling, cleaning, validating, and writing results.
"""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

import arnio as ar


class TestBootstrapFromReport:
    """Test Schema.bootstrap_from_report end-to-end workflows."""

    def test_bootstrap_from_profile_basic(self, tmp_path):
        """Load CSV, profile, bootstrap schema, and validate same frame."""
        # Create test CSV
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            "id,name,score\n"
            "1,Alice,95.5\n"
            "2,Bob,87.3\n"
            "3,Charlie,92.1\n"
        )

        # Load and profile
        frame = ar.read_csv(csv_file)
        report = ar.profile(frame)

        # Bootstrap schema from report
        schema = ar.Schema.bootstrap_from_report(report)

        # Validate same frame
        result = schema.validate(frame)
        assert result.passed is True
        assert len(schema.fields) == 3
        assert "id" in schema.fields
        assert "name" in schema.fields
        assert "score" in schema.fields

    def test_bootstrap_respects_nullability(self, tmp_path):
        """Bootstrap schema correctly identifies nullable columns from report."""
        csv_file = tmp_path / "sparse.csv"
        csv_file.write_text(
            "id,required_name,optional_score\n"
            "1,Alice,95\n"
            "2,Bob,\n"
            "3,,92\n"
        )

        frame = ar.read_csv(csv_file)
        report = ar.profile(frame)
        schema = ar.Schema.bootstrap_from_report(report)

        # Columns with nulls should be marked nullable
        assert schema.fields["optional_score"].nullable is True
        assert schema.fields["required_name"].nullable is True

    def test_bootstrap_raises_on_invalid_report_type(self):
        """bootstrap_from_report raises TypeError for non-report input."""
        with pytest.raises(TypeError, match="Expected DataQualityReport"):
            ar.Schema.bootstrap_from_report({"not": "a report"})

    def test_bootstrap_raises_on_empty_report(self):
        """bootstrap_from_report raises ValueError for empty report."""
        # Create a frame with no data columns
        empty_frame = ar.from_dict({})
        empty_report = ar.profile(empty_frame)

        with pytest.raises(ValueError, match="no columns"):
            ar.Schema.bootstrap_from_report(empty_report)

    def test_bootstrap_to_json_persistence(self, tmp_path):
        """Bootstrap schema, serialize to JSON, reload, and validate."""
        csv_file = tmp_path / "customers.csv"
        csv_file.write_text(
            "id,email,active\n"
            "1,alice@example.com,true\n"
            "2,bob@example.com,false\n"
        )

        # Profile and bootstrap
        frame = ar.read_csv(csv_file)
        report = ar.profile(frame)
        schema = ar.Schema.bootstrap_from_report(report)

        # Serialize
        json_str = schema.to_json()
        schema_file = tmp_path / "schema.json"
        schema_file.write_text(json_str)

        # Reload and validate
        restored_schema = ar.Schema.from_json(schema_file.read_text())
        result = restored_schema.validate(frame)
        assert result.passed is True


class TestSchemaJsonSerialization:
    """Test to_json and from_json round-trip workflows."""

    def test_to_json_produces_valid_json(self):
        """to_json produces deserializable JSON."""
        schema = ar.Schema(
            {
                "id": ar.Int64(nullable=False),
                "name": ar.String(nullable=False),
                "score": ar.Float64(nullable=True),
            }
        )

        json_str = schema.to_json()
        payload = json.loads(json_str)

        assert "fields" in payload
        assert payload["fields"]["id"]["dtype"] == "int64"
        assert payload["fields"]["id"]["nullable"] is False
        assert payload["fields"]["score"]["nullable"] is True

    def test_from_json_basic_deserialization(self):
        """from_json deserializes JSON schema correctly."""
        json_str = json.dumps(
            {
                "fields": {
                    "user_id": {"dtype": "int64", "nullable": False},
                    "email": {"dtype": "email", "nullable": False},
                },
                "strict": True,
                "unique": ["user_id"],
            }
        )

        schema = ar.Schema.from_json(json_str)
        assert len(schema.fields) == 2
        assert schema.fields["user_id"].dtype == "int64"
        assert schema.strict is True
        assert schema.unique == ["user_id"]

    def test_json_round_trip_validates_consistently(self, tmp_path):
        """Schema survives to_json -> from_json round-trip and validates identically."""
        # Create original schema
        original_schema = ar.Schema(
            {
                "user_id": ar.Int64(nullable=False, unique=True),
                "email": ar.Email(nullable=False),
                "created_at": ar.DateTime(format="%Y-%m-%d"),
                "is_active": ar.Bool(nullable=True),
            },
            strict=True,
        )

        # Round-trip through JSON
        json_str = original_schema.to_json()
        restored_schema = ar.Schema.from_json(json_str)

        # Create test frame
        frame = ar.from_dict(
            {
                "user_id": [1, 2, 3],
                "email": ["alice@example.com", "bob@example.com", "charlie@example.com"],
                "created_at": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "is_active": [True, False, True],
            }
        )

        # Both schemas should validate identically
        result_original = original_schema.validate(frame)
        result_restored = restored_schema.validate(frame)

        assert result_original.passed == result_restored.passed
        assert result_original.issue_count == result_restored.issue_count

    def test_from_json_raises_on_invalid_json(self):
        """from_json raises ValueError on malformed JSON."""
        with pytest.raises(ValueError, match="Invalid schema JSON"):
            ar.Schema.from_json("{ invalid json }")

    def test_from_json_raises_on_unknown_keys(self):
        """from_json raises ValueError for unknown schema keys."""
        invalid_json = json.dumps(
            {
                "fields": {},
                "strict": False,
                "unknown_key": "should not exist",
            }
        )

        with pytest.raises(ValueError, match="unknown key"):
            ar.Schema.from_json(invalid_json)

    def test_from_json_raises_on_invalid_fields_structure(self):
        """from_json raises TypeError if fields is not a dict."""
        invalid_json = json.dumps(
            {
                "fields": ["should", "be", "dict"],
                "strict": False,
            }
        )

        with pytest.raises(TypeError, match="fields.*must be an object"):
            ar.Schema.from_json(invalid_json)

    def test_to_json_with_rules_emits_warning(self):
        """to_json with cross-field rules emits UserWarning."""

        def sample_rule(row):
            return True

        schema = ar.Schema(
            {"age": ar.Int64()},
            rules=[sample_rule],
        )

        with pytest.warns(UserWarning, match="rules cannot be serialized"):
            json_str = schema.to_json()

        # JSON should mark rules as omitted
        payload = json.loads(json_str)
        assert payload.get("rules_omitted") is True


class TestSchemaValidateWorkflows:
    """Test Schema.validate end-to-end workflows."""

    def test_validate_passing_data_returns_passed_true(self):
        """Validate frame that meets all schema constraints."""
        schema = ar.Schema(
            {
                "id": ar.Int64(nullable=False),
                "email": ar.Email(nullable=False),
                "age": ar.Int64(min=0, max=150),
            }
        )

        frame = ar.from_dict(
            {
                "id": [1, 2, 3],
                "email": [
                    "alice@example.com",
                    "bob@example.com",
                    "charlie@example.com",
                ],
                "age": [25, 30, 35],
            }
        )

        result = schema.validate(frame)
        assert result.passed is True
        assert result.issue_count == 0

    def test_validate_failing_data_reports_issues(self):
        """Validate frame with constraint violations returns issues."""
        schema = ar.Schema(
            {
                "id": ar.Int64(nullable=False),
                "age": ar.Int64(min=18),
            }
        )

        frame = ar.from_dict(
            {
                "id": [1, 2, None],
                "age": [25, 15, 30],
            }
        )

        result = schema.validate(frame)
        assert result.passed is False
        assert result.issue_count > 0

        # Issues should have row_index and column info
        for issue in result.issues:
            assert hasattr(issue, "row_index")
            assert hasattr(issue, "column")
            assert hasattr(issue, "message")

    def test_validate_with_max_errors_limit(self):
        """Validate respects max_errors limit."""
        schema = ar.Schema(
            {
                "value": ar.Int64(min=100),
            }
        )

        frame = ar.from_dict(
            {
                "value": [1, 2, 3, 4, 5, 6, 7, 8],
            }
        )

        result = schema.validate(frame, max_errors=3)
        assert result.issue_count <= 3

    def test_validate_csv_to_pandas_workflow(self, tmp_path):
        """Full workflow: CSV -> ArFrame -> Profile -> Bootstrap -> Validate -> Pandas."""
        # Create messy CSV
        csv_file = tmp_path / "orders.csv"
        csv_file.write_text(
            "order_id,customer_name,amount,status\n"
            "101,Alice Smith,1500.50,completed\n"
            "102,Bob Jones,2000.75,pending\n"
            "103,Charlie Brown,999.99,completed\n"
        )

        # Load CSV
        frame = ar.read_csv(csv_file)

        # Profile and bootstrap schema
        report = ar.profile(frame)
        schema = ar.Schema.bootstrap_from_report(report)

        # Enhance schema with custom constraints
        schema.fields["amount"] = ar.Float64(min=0, max=100000, nullable=False)
        schema.fields["order_id"] = ar.Int64(nullable=False, unique=True)

        # Validate
        result = schema.validate(frame)
        assert result.passed is True

        # Convert to pandas and verify
        df = ar.to_pandas(frame)
        assert len(df) == 3
        assert list(df.columns) == ["order_id", "customer_name", "amount", "status"]

        # Save output
        output_csv = tmp_path / "validated_orders.csv"
        df.to_csv(output_csv, index=False)
        assert output_csv.exists()

    def test_validate_with_pandas_accessor(self, tmp_path):
        """Validate using pandas DataFrame.arnio accessor."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "email": ["alice@example.com", "bob@example.com", "charlie@example.com"],
                "score": [95.0, 87.0, 92.0],
            }
        )

        schema = ar.Schema(
            {
                "id": ar.Int64(nullable=False),
                "email": ar.Email(nullable=False),
                "score": ar.Float64(min=0, max=100),
            }
        )

        # Validate through accessor
        result = df.arnio.validate(schema)
        assert result.passed is True

    def test_complex_messy_data_pipeline(self, tmp_path):
        """Full pipeline: messy CSV -> clean -> profile -> schema -> validate -> export."""
        # Create messy CSV with whitespace and duplicates
        csv_file = tmp_path / "messy.csv"
        csv_file.write_text(
            "  ID  ,  NAME  , SCORE \n"
            "1, Alice ,95\n"
            "2, Bob ,87\n"
            "2, Bob ,87\n"  # duplicate
            "3,  Charlie  ,92\n"
        )

        # Load
        frame = ar.read_csv(csv_file)

        # Get cleaning suggestions and clean
        suggestions = ar.suggest_cleaning(frame)
        clean_frame = ar.pipeline(frame, suggestions)

        # Profile and bootstrap schema
        report = ar.profile(clean_frame)
        schema = ar.Schema.bootstrap_from_report(report)

        # Validate cleaned data
        result = schema.validate(clean_frame)
        assert result.passed is True

        # Serialize schema
        schema_json = schema.to_json()
        schema_file = tmp_path / "schema.json"
        schema_file.write_text(schema_json)

        # Export cleaned data
        df_clean = ar.to_pandas(clean_frame)
        output_csv = tmp_path / "cleaned.csv"
        df_clean.to_csv(output_csv, index=False)

        # Verify outputs exist and schema is valid JSON
        assert schema_file.exists()
        assert output_csv.exists()
        restored_schema = ar.Schema.from_json(schema_file.read_text())
        assert len(restored_schema.fields) > 0
