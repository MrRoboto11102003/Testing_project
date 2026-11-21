import pytest
import pandas as pd
import arnio as ar
import json
import warnings
from pathlib import Path


class TestSchemaJsonSerialization:
    """End-to-end tests for Schema.to_json() serialization."""

    def test_schema_to_json_roundtrip_no_rules(self, tmp_path):
        """Schema without cross-field rules can serialize and restore from JSON."""
        # Create a schema with multiple field types
        original_schema = ar.Schema({
            "user_id": ar.Int64(nullable=False, unique=True),
            "email": ar.Email(nullable=False),
            "score": ar.Float64(nullable=True, min=0.0, max=100.0),
            "is_active": ar.Bool(nullable=False),
            "username": ar.String(min_length=3, max_length=20),
        })
        
        # Serialize to JSON
        json_str = original_schema.to_json()
        assert isinstance(json_str, str)
        payload = json.loads(json_str)
        
        # Verify structure
        assert "fields" in payload
        assert "user_id" in payload["fields"]
        assert "email" in payload["fields"]
        assert payload["fields"]["user_id"]["nullable"] is False
        
        # Save and restore from file
        schema_file = tmp_path / "schema.json"
        schema_file.write_text(json_str)
        
        restored_json = schema_file.read_text()
        restored_schema = ar.Schema.from_json(restored_json)
        
        # Verify restored schema has same fields
        assert set(restored_schema.fields.keys()) == set(original_schema.fields.keys())
        assert restored_schema.strict == original_schema.strict

    def test_schema_to_json_emits_warning_with_cross_field_rules(self):
        """to_json() emits UserWarning when cross-field rules are present."""
        def cross_field_rule(df):
            return []
        
        schema = ar.Schema(
            {
                "start_date": ar.String(nullable=False),
                "end_date": ar.String(nullable=False),
            },
            rules=[cross_field_rule]
        )
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            json_str = schema.to_json()
            
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert "cross-field rules" in str(w[0].message)
        
        # Verify rules_omitted flag is set
        payload = json.loads(json_str)
        assert payload.get("rules_omitted") is True

    def test_schema_json_preserves_field_constraints(self, tmp_path):
        """Schema constraints (min, max, nullable, etc.) survive JSON roundtrip."""
        schema = ar.Schema({
            "age": ar.Int64(nullable=False, min=0, max=150),
            "email": ar.Email(nullable=False),
            "score": ar.Float64(nullable=True, min=0.0),
        })
        
        json_str = schema.to_json()
        payload = json.loads(json_str)
        
        # Check age constraints
        age_field = payload["fields"]["age"]
        assert age_field["nullable"] is False
        assert age_field.get("min") == 0
        assert age_field.get("max") == 150
        
        # Check email is Email type
        assert payload["fields"]["email"]["dtype"] == "email"
        
        # Check score has min
        score_field = payload["fields"]["score"]
        assert score_field["nullable"] is True
        assert score_field.get("min") == 0.0


class TestSchemaValidateAgainstCsvData:
    """End-to-end tests for validating real CSV files against schemas."""

    def test_validate_clean_csv_passes(self, tmp_path):
        """Validating clean CSV data against matching schema passes."""
        # Create a clean CSV file
        csv_path = tmp_path / "clean.csv"
        csv_path.write_text(
            "id,email,age\n"
            "1,alice@example.com,25\n"
            "2,bob@example.com,30\n"
            "3,charlie@example.com,35\n"
        )
        
        # Load and validate
        frame = ar.read_csv(str(csv_path))
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "email": ar.Email(nullable=False),
            "age": ar.Int64(nullable=False, min=0, max=150),
        })
        
        result = schema.validate(frame)
        
        assert result.passed is True
        assert len(result.issues) == 0

    def test_validate_csv_with_multiple_violations(self, tmp_path):
        """Validation detects and reports multiple data quality issues."""
        csv_path = tmp_path / "messy.csv"
        csv_path.write_text(
            "id,email,age\n"
            "not_an_int,alice@example.com,25\n"
            "2,invalid-email,30\n"
            "3,charlie@example.com,999\n"
        )
        
        frame = ar.read_csv(str(csv_path))
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "email": ar.Email(nullable=False),
            "age": ar.Int64(nullable=False, min=0, max=150),
        })
        
        result = schema.validate(frame)
        
        assert result.passed is False
        assert len(result.issues) >= 3  # Type error, email error, age max error
        
        # Export issues to CSV for analysis
        issues_df = result.to_pandas()
        issues_path = tmp_path / "issues.csv"
        issues_df.to_csv(str(issues_path), index=False)
        
        assert issues_path.exists()
        assert "column" in issues_df.columns
        assert "row_index" in issues_df.columns

    def test_validate_missing_required_column_fails(self, tmp_path):
        """Validation fails when CSV is missing a required schema column."""
        csv_path = tmp_path / "incomplete.csv"
        csv_path.write_text(
            "id,name\n"
            "1,alice\n"
            "2,bob\n"
        )
        
        frame = ar.read_csv(str(csv_path))
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "name": ar.String(nullable=False),
            "email": ar.Email(nullable=False),  # Missing from CSV
        })
        
        result = schema.validate(frame)
        
        assert result.passed is False
        assert any(issue.column == "email" for issue in result.issues)

    def test_validate_csv_with_nulls_in_non_nullable_field(self, tmp_path):
        """Validation fails when non-nullable field contains null values."""
        csv_path = tmp_path / "with_nulls.csv"
        csv_path.write_text(
            "id,email,status\n"
            "1,alice@example.com,active\n"
            "2,,inactive\n"
            "3,charlie@example.com,\n"
        )
        
        frame = ar.read_csv(str(csv_path))
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "email": ar.Email(nullable=False),
            "status": ar.String(nullable=True),  # Nullable, so OK
        })
        
        result = schema.validate(frame)
        
        assert result.passed is False
        # Should have issues for missing email in row 2
        assert any(issue.column == "email" for issue in result.issues)

    def test_validate_with_max_errors_limit(self, tmp_path):
        """Validation respects max_errors parameter to limit issue collection."""
        csv_path = tmp_path / "many_errors.csv"
        # Create CSV with many invalid email rows
        lines = ["id,email"]
        for i in range(100):
            lines.append(f"{i},invalid-email-{i}")
        csv_path.write_text("\n".join(lines))
        
        frame = ar.read_csv(str(csv_path))
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "email": ar.Email(nullable=False),
        })
        
        # Validate with max_errors=5
        result = schema.validate(frame, max_errors=5)
        
        assert len(result.issues) <= 5
        assert result.passed is False

    def test_validate_numeric_constraints(self, tmp_path):
        """Validation enforces min/max constraints on numeric fields."""
        csv_path = tmp_path / "scores.csv"
        csv_path.write_text(
            "user_id,score\n"
            "1,50\n"
            "2,101\n"
            "3,-5\n"
            "4,75\n"
        )
        
        frame = ar.read_csv(str(csv_path))
        schema = ar.Schema({
            "user_id": ar.Int64(nullable=False),
            "score": ar.Float64(nullable=False, min=0.0, max=100.0),
        })
        
        result = schema.validate(frame)
        
        assert result.passed is False
        # Should have violations for rows with score > 100 and score < 0
        score_violations = [i for i in result.issues if i.column == "score"]
        assert len(score_violations) >= 2


class TestSchemaValidateWithCrossFieldRules:
    """End-to-end tests for schemas with cross-field validation rules."""

    def test_cross_field_rule_validation(self, tmp_path):
        """Cross-field rules can validate relationships between columns."""
        def end_date_after_start_date(df):
            """Custom rule: end_date >= start_date."""
            issues = []
            for i, row in df.iterrows():
                # Compare as strings (simple date format)
                if row["end_date"] < row["start_date"]:
                    issues.append(
                        ar.ValidationIssue(
                            column="end_date",
                            rule="date_range",
                            message="end_date must be >= start_date",
                            row_index=int(i) + 1,
                        )
                    )
            return issues
        
        csv_path = tmp_path / "dates.csv"
        csv_path.write_text(
            "event_id,start_date,end_date\n"
            "1,2025-01-01,2025-12-31\n"
            "2,2025-02-01,2025-01-01\n"
            "3,2025-03-01,2025-03-31\n"
        )
        
        frame = ar.read_csv(str(csv_path))
        schema = ar.Schema(
            {
                "event_id": ar.Int64(nullable=False),
                "start_date": ar.String(nullable=False),
                "end_date": ar.String(nullable=False),
            },
            rules=[end_date_after_start_date]
        )
        
        result = schema.validate(frame)
        
        assert result.passed is False
        # Should have one cross-field rule violation
        rule_violations = [i for i in result.issues if i.rule == "date_range"]
        assert len(rule_violations) >= 1
        assert rule_violations[0].row_index == 2  # Second data row


class TestSchemaValidateWithPandasWorkflow:
    """End-to-end tests using pandas workflows with validation."""

    def test_validate_pandas_dataframe_via_arnio(self, tmp_path):
        """Validate pandas DataFrame converted through arnio."""
        df = pd.DataFrame({
            "user_id": [1, 2, 3],
            "email": ["alice@example.com", "bob@example.com", "charlie@example.com"],
            "score": [95.5, 87.0, 92.3],
        })
        
        # Convert to ArFrame
        frame = ar.from_pandas(df)
        
        schema = ar.Schema({
            "user_id": ar.Int64(nullable=False, unique=True),
            "email": ar.Email(nullable=False),
            "score": ar.Float64(nullable=False, min=0.0, max=100.0),
        })
        
        result = schema.validate(frame)
        
        assert result.passed is True
        assert len(result.issues) == 0

    def test_pipeline_cleaning_then_validation(self, tmp_path):
        """End-to-end: load messy CSV, clean with pipeline, validate."""
        csv_path = tmp_path / "messy.csv"
        csv_path.write_text(
            "  user_id  ,  email  ,  status  \n"
            "1,  ALICE@EXAMPLE.COM  , ACTIVE\n"
            "2, bob@EXAMPLE.COM , INACTIVE\n"
            "1,  ALICE@EXAMPLE.COM  , ACTIVE\n"
        )
        
        # Load and clean
        frame = ar.read_csv(str(csv_path))
        clean = ar.pipeline(frame, [
            ("trim_column_names",),
            ("strip_whitespace",),
            ("normalize_case", {"subset": ["email", "status"], "case_type": "lower"}),
            ("drop_duplicates",),
        ])
        
        # Validate cleaned data
        schema = ar.Schema({
            "user_id": ar.Int64(nullable=False),
            "email": ar.Email(nullable=False),
            "status": ar.String(nullable=False),
        })
        
        result = schema.validate(clean)
        
        assert result.passed is True
        # Result should have fewer rows due to deduplication
        result_df = ar.to_pandas(clean)
        assert len(result_df) == 2

    def test_validation_result_markdown_export(self, tmp_path):
        """ValidationResult can be exported to markdown for reporting."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text(
            "id,email,age\n"
            "1,bad-email,25\n"
            "2,good@example.com,999\n"
        )
        
        frame = ar.read_csv(str(csv_path))
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "email": ar.Email(nullable=False),
            "age": ar.Int64(nullable=False, min=0, max=150),
        })
        
        result = schema.validate(frame)
        
        # Export to markdown
        markdown_report = result.to_markdown()
        assert isinstance(markdown_report, str)
        assert len(markdown_report) > 0
        
        # Save report
        report_path = tmp_path / "validation_report.md"
        report_path.write_text(markdown_report)
        assert report_path.exists()

    def test_validation_result_issues_summary(self, tmp_path):
        """ValidationResult provides structured issue information."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text(
            "id,name\n"
            "not_int,alice\n"
            "2,bob\n"
        )
        
        frame = ar.read_csv(str(csv_path))
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "name": ar.String(nullable=False),
        })
        
        result = schema.validate(frame)
        
        # Check issue details
        assert result.passed is False
        assert len(result.issues) > 0
        
        for issue in result.issues:
            assert issue.column is not None
            assert issue.rule is not None
            assert issue.message is not None
            assert issue.row_index is not None


class TestSchemaValidateEdgeCases:
    """End-to-end tests for edge cases and boundary conditions."""

    def test_validate_empty_dataframe(self):
        """Validation handles empty DataFrames gracefully."""
        df = pd.DataFrame({
            "id": pd.Series([], dtype="int64"),
            "email": pd.Series([], dtype="object"),
        })
        
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "email": ar.Email(nullable=False),
        })
        
        result = schema.validate(frame)
        
        # Empty frame should pass (no rows to violate rules)
        assert result.passed is True
        assert len(result.issues) == 0

    def test_validate_single_row_dataframe(self, tmp_path):
        """Validation works correctly with single-row data."""
        csv_path = tmp_path / "single_row.csv"
        csv_path.write_text(
            "id,email\n"
            "1,user@example.com\n"
        )
        
        frame = ar.read_csv(str(csv_path))
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "email": ar.Email(nullable=False),
        })
        
        result = schema.validate(frame)
        
        assert result.passed is True

    def test_schema_strict_mode_rejects_extra_columns(self, tmp_path):
        """Strict schema rejects data with extra columns."""
        csv_path = tmp_path / "extra_cols.csv"
        csv_path.write_text(
            "id,email,debug_col\n"
            "1,user@example.com,extra\n"
        )
        
        frame = ar.read_csv(str(csv_path))
        
        # Strict schema: exact columns required
        schema = ar.Schema(
            {
                "id": ar.Int64(nullable=False),
                "email": ar.Email(nullable=False),
            },
            strict=True
        )
        
        result = schema.validate(frame)
        
        # Strict mode should fail due to extra column
        assert result.passed is False

    def test_validate_all_field_types(self, tmp_path):
        """Validation works with diverse field types."""
        csv_path = tmp_path / "all_types.csv"
        csv_path.write_text(
            "int_col,float_col,string_col,bool_col,email_col\n"
            "42,3.14,hello,true,user@example.com\n"
        )
        
        frame = ar.read_csv(str(csv_path))
        schema = ar.Schema({
            "int_col": ar.Int64(nullable=False),
            "float_col": ar.Float64(nullable=False),
            "string_col": ar.String(nullable=False),
            "bool_col": ar.Bool(nullable=False),
            "email_col": ar.Email(nullable=False),
        })
        
        result = schema.validate(frame)
        
        assert result.passed is True
