"""
Pytest tests for arnio.schema.Schema.validate() method.
Tests the documented behavior including edge cases and constraints.
"""
import pytest
import pandas as pd
import arnio as ar


class TestSchemaValidate:
    """Tests for arnio.schema.Schema.validate() method."""

    def test_valid_data_passes_validation(self):
        """Valid data should pass validation."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["alice", "bob", "charlie"],
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "name": ar.String(nullable=False),
        })
        result = schema.validate(frame)
        assert result.passed
        assert result.issue_count == 0

    def test_invalid_dtype_fails_validation(self):
        """Data with wrong dtype should fail validation."""
        df = pd.DataFrame({
            "age": ["not_a_number", "also_not", "still_not"],
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "age": ar.Int64(nullable=False),
        })
        result = schema.validate(frame)
        assert not result.passed
        assert result.issue_count > 0

    def test_nullable_false_with_null_fails(self):
        """Null values with nullable=False should fail."""
        df = pd.DataFrame({
            "id": [1.0, None, 3.0],
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "id": ar.Float64(nullable=False),
        })
        result = schema.validate(frame)
        assert not result.passed
        assert result.issue_count > 0

    def test_nullable_true_allows_nulls(self):
        """Null values with nullable=True should pass."""
        df = pd.DataFrame({
            "id": [1, None, 3],
            "name": ["alice", "bob", None],
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "id": ar.Int64(nullable=True),
            "name": ar.String(nullable=True),
        })
        result = schema.validate(frame)
        assert result.passed

    def test_min_constraint_numeric(self):
        """Values below min constraint should fail."""
        df = pd.DataFrame({"age": [10, 15, 25]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"age": ar.Int64(min=18)})
        result = schema.validate(frame)
        assert not result.passed
        assert result.issue_count > 0

    def test_max_constraint_numeric(self):
        """Values above max constraint should fail."""
        df = pd.DataFrame({"age": [18, 25, 150]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"age": ar.Int64(max=130)})
        result = schema.validate(frame)
        assert not result.passed
        assert result.issue_count > 0

    def test_min_length_constraint_string(self):
        """Strings shorter than min_length should fail."""
        df = pd.DataFrame({"username": ["a", "valid_user", "another"]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"username": ar.String(min_length=3)})
        result = schema.validate(frame)
        assert not result.passed

    def test_max_length_constraint_string(self):
        """Strings longer than max_length should fail."""
        df = pd.DataFrame({"code": ["ABC", "DEFGH", "XY"]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"code": ar.String(max_length=3)})
        result = schema.validate(frame)
        assert not result.passed

    def test_email_validation_fails_on_invalid(self):
        """Invalid email formats should fail validation."""
        df = pd.DataFrame({
            "email": ["valid@example.com", "invalid-email", "another@test.com"],
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({"email": ar.Email(nullable=False)})
        result = schema.validate(frame)
        assert not result.passed
        assert result.issue_count > 0

    def test_unique_constraint_detects_duplicates(self):
        """Duplicate values with unique=True should fail."""
        df = pd.DataFrame({"id": [1, 2, 1]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"id": ar.Int64(nullable=False, unique=True)})
        result = schema.validate(frame)
        assert not result.passed

    def test_unique_constraint_passes_for_unique_values(self):
        """All unique values should pass unique=True constraint."""
        df = pd.DataFrame({"id": [1, 2, 3]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"id": ar.Int64(nullable=False, unique=True)})
        result = schema.validate(frame)
        assert result.passed

    def test_max_errors_parameter_limits_issues(self):
        """max_errors parameter should limit number of returned issues."""
        df = pd.DataFrame({"age": [-1, -2, -3, -4, -5]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"age": ar.Int64(min=0)})
        
        result_limited = schema.validate(frame, max_errors=2)
        result_unlimited = schema.validate(frame)
        
        assert result_limited.issue_count <= 2
        assert result_unlimited.issue_count > 2

    def test_missing_required_column_fails(self):
        """Schema column not in frame should fail validation."""
        df = pd.DataFrame({"id": [1, 2, 3]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "email": ar.Email(nullable=False),
        })
        result = schema.validate(frame)
        assert not result.passed

    def test_validation_result_provides_issue_details(self):
        """ValidationResult should expose issue metadata."""
        df = pd.DataFrame({"age": [10, 200]})
        frame = ar.from_pandas(df)
        schema = ar.Schema({"age": ar.Int64(min=18, max=130)})
        result = schema.validate(frame)
        
        assert not result.passed
        assert result.issue_count > 0
        assert len(result.issues) == result.issue_count
        
        # Each issue should have required fields
        for issue in result.issues:
            assert hasattr(issue, "column")
            assert hasattr(issue, "rule")
            assert hasattr(issue, "message")
            assert isinstance(issue.column, str)
            assert isinstance(issue.rule, str)
            assert isinstance(issue.message, str)

    def test_empty_frame_passes_validation(self):
        """Empty frame with no data should pass validation."""
        df = pd.DataFrame({
            "id": pd.Series([], dtype="int64"),
            "name": pd.Series([], dtype="object"),
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "name": ar.String(nullable=False),
        })
        result = schema.validate(frame)
        assert result.passed

    def test_multiple_field_validation_reports_all_issues(self):
        """Validation should report issues across multiple fields."""
        df = pd.DataFrame({
            "id": ["invalid", "also_invalid"],
            "age": [10, 200],
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "age": ar.Int64(min=18, max=130),
        })
        result = schema.validate(frame)
        assert not result.passed
        # Should have issues in both columns
        columns_with_issues = {issue.column for issue in result.issues}
        assert len(columns_with_issues) > 1
