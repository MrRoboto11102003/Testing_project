import pytest
import pandas as pd
import arnio as ar
import numpy as np


class TestSchemaValidate:
    """Tests for arnio.schema.Schema.validate() method."""

    def test_validate_passes_for_valid_data(self):
        """Validation passes when all constraints are satisfied."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "id": ar.Int64(nullable=False, unique=True),
            "name": ar.String(nullable=False),
        })
        result = schema.validate(frame)
        assert result.passed is True
        assert result.issue_count == 0

    def test_validate_fails_on_dtype_mismatch(self):
        """Validation fails when column dtype doesn't match schema."""
        df = pd.DataFrame({"value": ["text", "more_text"]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"value": ar.Int64()})
        result = schema.validate(frame)
        assert result.passed is False
        assert len(result.issues) > 0

    def test_validate_nullable_false_with_nulls(self):
        """Validation fails when nullable=False but nulls are present."""
        df = pd.DataFrame({"required": [1.0, 2.0, None]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"required": ar.Float64(nullable=False)})
        result = schema.validate(frame)
        assert result.passed is False

    def test_validate_nullable_true_allows_nulls(self):
        """Validation passes when nullable=True and nulls are present."""
        df = pd.DataFrame({"optional": [1, 2, None]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"optional": ar.Int64(nullable=True)})
        result = schema.validate(frame)
        assert result.passed is True

    def test_validate_unique_constraint_violation(self):
        """Validation fails when unique constraint is violated."""
        df = pd.DataFrame({"id": [1, 2, 2, 3]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"id": ar.Int64(unique=True)})
        result = schema.validate(frame)
        assert result.passed is False

    def test_validate_unique_constraint_satisfied(self):
        """Validation passes when all values are unique."""
        df = pd.DataFrame({"id": [1, 2, 3, 4]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"id": ar.Int64(unique=True)})
        result = schema.validate(frame)
        assert result.passed is True

    def test_validate_min_bound_violation(self):
        """Validation fails when numeric value is below min."""
        df = pd.DataFrame({"age": [18, 25, 10]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"age": ar.Int64(min=18)})
        result = schema.validate(frame)
        assert result.passed is False

    def test_validate_max_bound_violation(self):
        """Validation fails when numeric value exceeds max."""
        df = pd.DataFrame({"score": [50.0, 75.0, 150.0]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"score": ar.Float64(max=100)})
        result = schema.validate(frame)
        assert result.passed is False

    def test_validate_string_length_constraints(self):
        """Validation enforces min_length and max_length for strings."""
        df = pd.DataFrame({"code": ["AB", "ABC", "X" * 100]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"code": ar.String(min_length=3, max_length=50)})
        result = schema.validate(frame)
        assert result.passed is False

    def test_validate_email_format(self):
        """Validation enforces email format constraint."""
        df = pd.DataFrame({
            "email": ["valid@example.com", "not-an-email"]
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({"email": ar.Email()})
        result = schema.validate(frame)
        assert result.passed is False

    def test_validate_url_format(self):
        """Validation enforces URL format constraint."""
        df = pd.DataFrame({
            "website": ["https://example.com", "not-a-url"]
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({"website": ar.URL()})
        result = schema.validate(frame)
        assert result.passed is False

    def test_validate_composite_unique_constraint(self):
        """Validation enforces multi-column uniqueness via schema unique parameter."""
        df = pd.DataFrame({
            "user_id": [1, 1, 2, 2],
            "course_id": [101, 101, 101, 102],
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema(
            {
                "user_id": ar.Int64(),
                "course_id": ar.Int64(),
            },
            unique=["user_id", "course_id"]
        )
        result = schema.validate(frame)
        assert result.passed is False

    def test_validate_row_index_one_based(self):
        """ValidationIssue.row_index is 1-based, not 0-based (header not counted)."""
        df = pd.DataFrame({"value": ["invalid", "2", "3"]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"value": ar.Int64()})
        result = schema.validate(frame)
        # First data row should have row_index=1
        assert result.issues[0].row_index == 1

    def test_validate_max_errors_parameter_limits_issues(self):
        """max_errors parameter limits number of issues returned."""
        df = pd.DataFrame({"value": ["invalid"] * 10})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"value": ar.Int64()})
        result = schema.validate(frame, max_errors=3)
        assert len(result.issues) <= 3

    def test_validate_warning_severity_does_not_fail(self):
        """Validation passes with only warning-level issues; passed=True."""
        df = pd.DataFrame({"age": [5, 10]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "age": ar.Int64(min=18, severity="warning"),
        })
        result = schema.validate(frame)
        assert result.passed is True
        assert result.issue_count > 0

    def test_validate_result_has_summary_method(self):
        """ValidationResult has summary() method returning dict."""
        df = pd.DataFrame({"id": [1, 1]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"id": ar.Int64(unique=True)})
        result = schema.validate(frame)
        summary = result.summary()
        assert isinstance(summary, dict)
