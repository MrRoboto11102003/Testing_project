import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestSchemaValidate:
    """Tests for arnio.schema.Schema.validate() method"""

    def test_validate_returns_result_object(self):
        """validate() returns a ValidationResult with passed and issues attributes"""
        schema = ar.Schema({"age": ar.Int64()})
        frame = ar.from_pandas(pd.DataFrame({"age": [25, 30]}))
        result = schema.validate(frame)
        assert hasattr(result, "passed")
        assert hasattr(result, "issues")
        assert isinstance(result.passed, bool)
        assert isinstance(result.issues, list)

    def test_validate_passes_on_valid_data(self):
        """validate() returns passed=True when all constraints are satisfied"""
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "age": ar.Int64(min=0, max=150),
            "name": ar.String(nullable=False),
        })
        frame = ar.from_pandas(pd.DataFrame({
            "id": [1, 2, 3],
            "age": [25, 50, 100],
            "name": ["alice", "bob", "charlie"],
        }))
        result = schema.validate(frame)
        assert result.passed is True
        assert len(result.issues) == 0

    def test_validate_fails_on_type_mismatch(self):
        """validate() fails when column dtype doesn't match schema type"""
        schema = ar.Schema({"count": ar.Int64()})
        frame = ar.from_pandas(pd.DataFrame({"count": ["1", "2", "3"]}))
        result = schema.validate(frame)
        assert result.passed is False
        assert len(result.issues) > 0

    def test_validate_fails_on_min_constraint(self):
        """validate() fails when numeric values are below minimum"""
        schema = ar.Schema({"age": ar.Int64(min=18)})
        frame = ar.from_pandas(pd.DataFrame({"age": [10, 30, 5]}))
        result = schema.validate(frame)
        assert result.passed is False
        assert any(issue.rule == "min" for issue in result.issues)

    def test_validate_fails_on_max_constraint(self):
        """validate() fails when numeric values exceed maximum"""
        schema = ar.Schema({"score": ar.Float64(max=100.0)})
        frame = ar.from_pandas(pd.DataFrame({"score": [50.0, 150.0, 75.0]}))
        result = schema.validate(frame)
        assert result.passed is False
        assert any(issue.rule == "max" for issue in result.issues)

    def test_validate_nullable_true_allows_nulls(self):
        """validate() allows null values when nullable=True"""
        schema = ar.Schema({"optional_field": ar.String(nullable=True)})
        frame = ar.from_pandas(pd.DataFrame({"optional_field": ["text", None, "more"]}))
        result = schema.validate(frame)
        assert result.passed is True

    def test_validate_nullable_false_rejects_nulls(self):
        """validate() fails when nullable=False and nulls are present"""
        schema = ar.Schema({"required_field": ar.String(nullable=False)})
        frame = ar.from_pandas(pd.DataFrame({"required_field": ["text", None, "more"]}))
        result = schema.validate(frame)
        assert result.passed is False
        assert any(issue.rule == "required" for issue in result.issues)

    def test_validate_unique_constraint_enforced(self):
        """validate() fails when unique constraint is violated"""
        schema = ar.Schema({"user_id": ar.Int64(unique=True)})
        frame = ar.from_pandas(pd.DataFrame({"user_id": [1, 2, 1, 3]}))
        result = schema.validate(frame)
        assert result.passed is False
        assert any("unique" in issue.rule.lower() for issue in result.issues)

    def test_validate_string_min_length(self):
        """validate() enforces string minimum length constraint"""
        schema = ar.Schema({"username": ar.String(min_length=3)})
        frame = ar.from_pandas(pd.DataFrame({"username": ["ab", "alice", "x"]}))
        result = schema.validate(frame)
        assert result.passed is False
        assert any(issue.rule == "min_length" for issue in result.issues)

    def test_validate_string_max_length(self):
        """validate() enforces string maximum length constraint"""
        schema = ar.Schema({"code": ar.String(max_length=4)})
        frame = ar.from_pandas(pd.DataFrame({"code": ["AB", "TOOLONG", "X"]}))
        result = schema.validate(frame)
        assert result.passed is False
        assert any(issue.rule == "max_length" for issue in result.issues)

    def test_validate_email_format(self):
        """validate() checks email format validity"""
        schema = ar.Schema({"email": ar.Email()})
        frame = ar.from_pandas(pd.DataFrame({
            "email": ["user@example.com", "invalid-email", "test@domain.org"]
        }))
        result = schema.validate(frame)
        assert result.passed is False
        assert any(issue.column == "email" for issue in result.issues)

    def test_validate_regex_pattern(self):
        """validate() validates values against regex patterns"""
        schema = ar.Schema({"code": ar.Regex(r"^[A-Z]{3}-\d{4}$")})
        frame = ar.from_pandas(pd.DataFrame({
            "code": ["ABC-1234", "invalid", "XYZ-9999"]
        }))
        result = schema.validate(frame)
        assert result.passed is False
        assert any(issue.rule == "regex" for issue in result.issues)

    def test_validate_row_index_is_one_based(self):
        """ValidationIssue.row_index is 1-based for data rows only"""
        schema = ar.Schema({"age": ar.Int64(min=18)})
        frame = ar.from_pandas(pd.DataFrame({"age": [10, 25, 30]}))
        result = schema.validate(frame)
        assert result.passed is False
        # First data row has row_index=1 (header not counted)
        assert result.issues[0].row_index == 1

    def test_validate_issue_contains_required_fields(self):
        """ValidationIssue contains column, rule, message, and value"""
        schema = ar.Schema({"count": ar.Int64(min=0)})
        frame = ar.from_pandas(pd.DataFrame({"count": [-5]}))
        result = schema.validate(frame)
        issue = result.issues[0]
        assert issue.column == "count"
        assert issue.rule == "min"
        assert isinstance(issue.message, str) and len(issue.message) > 0
        assert issue.value == -5
        assert issue.row_index == 1

    def test_validate_cross_field_rules(self):
        """validate() applies cross-field validation rules"""
        def date_order_rule(df):
            issues = []
            for i, row in df.iterrows():
                if row["end"] <= row["start"]:
                    issues.append(ar.ValidationIssue(
                        column="end",
                        rule="date_order",
                        message="end must be after start",
                        row_index=int(i) + 1,
                    ))
            return issues

        schema = ar.Schema(
            {"start": ar.Int64(), "end": ar.Int64()},
            rules=[date_order_rule],
        )
        frame = ar.from_pandas(pd.DataFrame({
            "start": [1, 5],
            "end": [10, 3],
        }))
        result = schema.validate(frame)
        assert result.passed is False
        assert any(issue.rule == "date_order" for issue in result.issues)

    def test_validate_max_errors_limits_results(self):
        """validate() respects max_errors parameter to cap reported issues"""
        schema = ar.Schema({"num": ar.Int64(min=100)})
        frame = ar.from_pandas(pd.DataFrame({"num": [1, 2, 3, 4, 5]}))
        result = schema.validate(frame, max_errors=2)
        assert len(result.issues) <= 2

    def test_validate_empty_frame(self):
        """validate() handles empty frames without error"""
        schema = ar.Schema({"id": ar.Int64(), "name": ar.String()})
        frame = ar.from_pandas(pd.DataFrame({"id": [], "name": []}))
        result = schema.validate(frame)
        assert result.passed is True

    def test_validate_missing_column_fails(self):
        """validate() fails when a required schema column is missing from data"""
        schema = ar.Schema({"id": ar.Int64(), "name": ar.String()})
        frame = ar.from_pandas(pd.DataFrame({"id": [1, 2]}))
        result = schema.validate(frame)
        assert result.passed is False
        assert any(issue.rule == "required_column" for issue in result.issues)
