import pytest
import csv
import tempfile
import pandas as pd
import numpy as np
import arnio as ar
from pathlib import Path


class TestSchemaValidation:
    """End-to-end tests for arnio.schema.Schema.validate()"""

    @pytest.fixture
    def write_csv(self, tmp_path):
        """Helper to write a CSV file and return its path"""
        def _write(filename, rows, fieldnames):
            path = tmp_path / filename
            with open(path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            return path
        return _write

    def test_valid_data_passes_all_checks(self, write_csv):
        """Valid data should pass validation with no issues"""
        path = write_csv(
            "users.csv",
            [
                {"id": "1", "name": "Alice", "age": "30", "active": "true"},
                {"id": "2", "name": "Bob", "age": "25", "active": "false"},
            ],
            ["id", "name", "age", "active"]
        )

        frame = ar.read_csv(str(path))
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "name": ar.String(nullable=False),
            "age": ar.Int64(min=0, max=150),
            "active": ar.Bool(),
        })

        result = schema.validate(frame)
        assert result.passed is True
        assert result.issue_count == 0
        assert len(result.issues) == 0

    def test_nullable_constraint_violation(self, write_csv):
        """Null value in non-nullable field should fail"""
        path = write_csv(
            "users.csv",
            [
                {"id": "1", "email": "alice@example.com"},
                {"id": "2", "email": ""},  # Empty/null
            ],
            ["id", "email"]
        )

        frame = ar.read_csv(str(path))
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "email": ar.Email(nullable=False),
        })

        result = ar.validate(frame, schema)
        assert result.passed is False
        assert result.issue_count > 0

    def test_dtype_mismatch_detection(self, write_csv):
        """Type mismatch should be detected"""
        path = write_csv(
            "data.csv",
            [
                {"id": "1", "age": "thirty"},  # Non-numeric string
                {"id": "2", "age": "25"},
            ],
            ["id", "age"]
        )

        frame = ar.read_csv(str(path))
        schema = ar.Schema({
            "id": ar.Int64(),
            "age": ar.Int64(),
        })

        result = schema.validate(frame)
        assert result.passed is False

    def test_string_length_constraints(self, write_csv):
        """String min_length and max_length constraints validated"""
        path = write_csv(
            "users.csv",
            [
                {"username": "alice"},      # 5 chars, valid
                {"username": "ab"},         # 2 chars, too short
                {"username": "thisusernameistoolongforthelimit"},  # too long
            ],
            ["username"]
        )

        frame = ar.read_csv(str(path))
        schema = ar.Schema({
            "username": ar.String(min_length=3, max_length=20),
        })

        result = schema.validate(frame)
        assert result.passed is False
        assert result.issue_count == 2

    def test_numeric_min_max_bounds(self, write_csv):
        """Numeric min and max constraints enforced"""
        path = write_csv(
            "products.csv",
            [
                {"id": "1", "price": "50"},
                {"id": "2", "price": "-10"},  # Below min
                {"id": "3", "price": "1000"},  # Above max
            ],
            ["id", "price"]
        )

        frame = ar.read_csv(str(path))
        schema = ar.Schema({
            "id": ar.Int64(),
            "price": ar.Float64(min=0, max=500),
        })

        result = schema.validate(frame)
        assert result.passed is False
        summary = result.summary()
        assert "price" in summary["issues_by_column"]

    def test_unique_constraint_single_column(self, write_csv):
        """Single column unique constraint violation detected"""
        path = write_csv(
            "users.csv",
            [
                {"id": "1", "email": "alice@example.com"},
                {"id": "2", "email": "bob@example.com"},
                {"id": "1", "email": "charlie@example.com"},  # Duplicate id
            ],
            ["id", "email"]
        )

        frame = ar.read_csv(str(path))
        schema = ar.Schema({
            "id": ar.Int64(unique=True),
            "email": ar.String(),
        })

        result = ar.validate(frame, schema)
        assert result.passed is False

    def test_composite_unique_constraint(self, write_csv):
        """Composite uniqueness (multiple columns) violations detected"""
        path = write_csv(
            "enrollments.csv",
            [
                {"student_id": "1", "course_id": "101"},
                {"student_id": "1", "course_id": "102"},
                {"student_id": "1", "course_id": "101"},  # Duplicate composite key
            ],
            ["student_id", "course_id"]
        )

        frame = ar.read_csv(str(path))
        schema = ar.Schema(
            {
                "student_id": ar.Int64(),
                "course_id": ar.Int64(),
            },
            unique=["student_id", "course_id"]
        )

        result = ar.validate(frame, schema)
        assert result.passed is False

    def test_email_field_validation(self, write_csv):
        """Email format validation"""
        path = write_csv(
            "users.csv",
            [
                {"email": "valid@example.com"},
                {"email": "another.valid+tag@example.org"},
                {"email": "not_an_email"},  # Invalid format
            ],
            ["email"]
        )

        frame = ar.read_csv(str(path))
        schema = ar.Schema({
            "email": ar.Email(),
        })

        result = schema.validate(frame)
        assert result.passed is False
        assert result.issue_count >= 1

    def test_regex_pattern_validation(self, write_csv):
        """Regex pattern constraints validated"""
        path = write_csv(
            "users.csv",
            [
                {"user_code": "USR-0001"},
                {"user_code": "USR-0002"},
                {"user_code": "INVALID"},  # Doesn't match pattern
            ],
            ["user_code"]
        )

        frame = ar.read_csv(str(path))
        schema = ar.Schema({
            "user_code": ar.Regex(r"^USR-\d{4}$", nullable=False),
        })

        result = schema.validate(frame)
        assert result.passed is False
        assert any(issue.column == "user_code" for issue in result.issues)

    def test_warning_severity_does_not_fail(self, write_csv):
        """Warning-level severity allows validation to pass despite issues"""
        path = write_csv(
            "users.csv",
            [
                {"age": "15"},  # Below min
                {"age": "25"},
            ],
            ["age"]
        )

        frame = ar.read_csv(str(path))
        schema = ar.Schema({
            "age": ar.Int64(min=18, severity="warning"),
        })

        result = schema.validate(frame)
        assert result.passed is True  # Validation passes
        assert result.issue_count > 0  # But issues are recorded

    def test_max_errors_parameter_limits_issues(self, write_csv):
        """max_errors parameter limits number of reported issues"""
        path = write_csv(
            "data.csv",
            [
                {"value": "not_a_number"},
                {"value": "also_not_a_number"},
                {"value": "another_bad_value"},
            ],
            ["value"]
        )

        frame = ar.read_csv(str(path))
        schema = ar.Schema({"value": ar.Int64()})

        result_limited = ar.validate(frame, schema, max_errors=1)
        result_unlimited = ar.validate(frame, schema)

        assert result_limited.issue_count <= 1
        assert result_unlimited.issue_count >= result_limited.issue_count

    def test_validation_result_to_pandas_dataframe(self, write_csv):
        """ValidationResult can be converted to pandas DataFrame"""
        path = write_csv(
            "data.csv",
            [
                {"id": "1", "age": "invalid"},
                {"id": "2", "age": "not_a_number"},
            ],
            ["id", "age"]
        )

        frame = ar.read_csv(str(path))
        schema = ar.Schema({
            "id": ar.Int64(),
            "age": ar.Int64(),
        })

        result = ar.validate(frame, schema)
        df = result.to_pandas()

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "row_index" in df.columns or "column" in df.columns

    def test_validation_result_to_markdown(self, write_csv):
        """ValidationResult produces markdown output"""
        path = write_csv(
            "data.csv",
            [
                {"email": "invalid"},
                {"email": "also_invalid"},
            ],
            ["email"]
        )

        frame = ar.read_csv(str(path))
        schema = ar.Schema({
            "email": ar.Email(),
        })

        result = ar.validate(frame, schema)
        md = result.to_markdown()

        assert isinstance(md, str)
        assert len(md) > 0

    def test_summary_structure_and_content(self, write_csv):
        """summary() returns correctly structured dictionary"""
        path = write_csv(
            "data.csv",
            [
                {"id": "not_number", "name": "alice"},
                {"id": "2", "name": ""},  # Empty name
            ],
            ["id", "name"]
        )

        frame = ar.read_csv(str(path))
        schema = ar.Schema({
            "id": ar.Int64(),
            "name": ar.String(nullable=False),
        })

        result = schema.validate(frame)
        summary = result.summary()

        assert isinstance(summary, dict)
        assert "issues_by_rule" in summary
        assert "issues_by_column" in summary
        assert "issues_by_column_and_rule" in summary

    def test_row_index_one_based_convention(self, write_csv):
        """Row indices in issues are 1-based (data rows only)"""
        path = write_csv(
            "data.csv",
            [
                {"value": "1"},      # Row 1 (data row, not header)
                {"value": "invalid"},  # Row 2 (should have row_index=2)
            ],
            ["value"]
        )

        frame = ar.read_csv(str(path))
        schema = ar.Schema({"value": ar.Int64()})
        result = schema.validate(frame)

        # Should have issue with row_index=2 (1-based, header not counted)
        assert any(issue.row_index == 2 for issue in result.issues)

    def test_validation_with_mixed_violations(self, write_csv):
        """Multiple validation violations in complex schema"""
        path = write_csv(
            "users.csv",
            [
                {"id": "1", "username": "al", "email": "alice@example.com", "age": "30"},
                {"id": "2", "username": "bob", "email": "invalid_email", "age": "-5"},
                {"id": "1", "username": "charlie", "email": "charlie@example.com", "age": "200"},
            ],
            ["id", "username", "email", "age"]
        )

        frame = ar.read_csv(str(path))
        schema = ar.Schema({
            "id": ar.Int64(unique=True),
            "username": ar.String(min_length=3, max_length=20),
            "email": ar.Email(),
            "age": ar.Int64(min=0, max=150),
        })

        result = ar.validate(frame, schema)
        assert result.passed is False
        assert result.issue_count >= 4  # At least 4 violations

        # Verify we can get summary and markdown without errors
        summary = result.summary()
        md = result.to_markdown()
        assert isinstance(summary, dict)
        assert isinstance(md, str)
