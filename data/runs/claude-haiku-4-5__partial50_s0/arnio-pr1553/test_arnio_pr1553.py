import pytest
import pandas as pd
import arnio as ar


class TestSchemaValidate:
    """Test arnio.schema.Schema.validate() method."""

    def test_validate_passes_valid_data(self):
        """Valid data matching schema should pass."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"]
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "id": ar.Int64(),
            "name": ar.String()
        })
        result = schema.validate(frame)
        assert result.passed is True
        assert len(result.issues) == 0

    def test_validate_dtype_mismatch(self):
        """Dtype mismatches should be detected and reported."""
        df = pd.DataFrame({
            "age": ["not_a_number", "25", "30"]
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({"age": ar.Int64()})
        result = schema.validate(frame)
        assert result.passed is False
        assert len(result.issues) > 0

    def test_validate_nullable_false_with_nulls(self):
        """Null values should fail when nullable=False."""
        df = pd.DataFrame({
            "email": ["alice@example.com", None, "charlie@example.com"]
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "email": ar.Email(nullable=False)
        })
        result = schema.validate(frame)
        assert result.passed is False
        assert any(issue.column == "email" for issue in result.issues)

    def test_validate_nullable_true_allows_nulls(self):
        """Null values should pass when nullable=True."""
        df = pd.DataFrame({
            "email": ["alice@example.com", None, "charlie@example.com"]
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "email": ar.Email(nullable=True)
        })
        result = schema.validate(frame)
        assert result.passed is True

    def test_validate_unique_constraint_violation(self):
        """Duplicate values should fail unique constraint."""
        df = pd.DataFrame({
            "user_id": [1, 2, 2, 3]
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "user_id": ar.Int64(unique=True)
        })
        result = schema.validate(frame)
        assert result.passed is False

    def test_validate_int64_min_constraint(self):
        """Int64 values below min should fail."""
        df = pd.DataFrame({
            "age": [10, 25, -5]
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "age": ar.Int64(min=0)
        })
        result = schema.validate(frame)
        assert result.passed is False

    def test_validate_int64_max_constraint(self):
        """Int64 values above max should fail."""
        df = pd.DataFrame({
            "age": [10, 25, 150]
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "age": ar.Int64(max=120)
        })
        result = schema.validate(frame)
        assert result.passed is False

    def test_validate_string_length_constraints(self):
        """String length violations should fail validation."""
        df = pd.DataFrame({
            "username": ["ab", "valid_user", "ok"]
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "username": ar.String(min_length=3, max_length=15)
        })
        result = schema.validate(frame)
        assert result.passed is False

    def test_validate_email_format(self):
        """Invalid email format should be detected."""
        df = pd.DataFrame({
            "email": ["valid@example.com", "not_an_email", "another@example.com"]
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "email": ar.Email()
        })
        result = schema.validate(frame)
        assert result.passed is False

    def test_validate_row_index_is_one_based(self):
        """Row index should be 1-based for first data row."""
        df = pd.DataFrame({
            "value": [10, -5, 20]
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "value": ar.Int64(min=0)
        })
        result = schema.validate(frame)
        # Second data row (row 2) has -5 violation
        assert any(issue.row_index == 2 for issue in result.issues)

    def test_validate_issue_attributes(self):
        """ValidationIssue objects should have row_index, column, and message."""
        df = pd.DataFrame({"age": [-1]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "age": ar.Int64(min=0)
        })
        result = schema.validate(frame)
        assert len(result.issues) > 0
        issue = result.issues[0]
        assert hasattr(issue, 'row_index') and isinstance(issue.row_index, int)
        assert hasattr(issue, 'column') and isinstance(issue.column, str)
        assert hasattr(issue, 'message') and isinstance(issue.message, str)

    def test_validate_max_errors_parameter(self):
        """max_errors parameter should limit issues reported."""
        df = pd.DataFrame({
            "value": [-1, -2, -3, -4, -5]
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "value": ar.Int64(min=0)
        })
        result = schema.validate(frame, max_errors=2)
        assert len(result.issues) <= 2

    def test_validate_missing_column(self):
        """Validation should fail when required column is missing."""
        df = pd.DataFrame({"id": [1, 2, 3]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "id": ar.Int64(),
            "name": ar.String()
        })
        result = schema.validate(frame)
        assert result.passed is False

    def test_validate_returns_validation_result(self):
        """validate() should return object with expected attributes."""
        df = pd.DataFrame({"id": [1, 2, 3]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"id": ar.Int64()})
        result = schema.validate(frame)
        assert hasattr(result, 'passed')
        assert hasattr(result, 'issues')
        assert hasattr(result, 'summary')
        assert hasattr(result, 'to_pandas')
        assert hasattr(result, 'to_markdown')

    def test_validate_regex_pattern(self):
        """Regex validation should detect non-matching patterns."""
        df = pd.DataFrame({
            "code": ["USR-0001", "USR-0002", "INVALID"]
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "code": ar.Regex(r"^USR-\d{4}$")
        })
        result = schema.validate(frame)
        assert result.passed is False
