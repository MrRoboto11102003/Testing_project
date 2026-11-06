import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestSchemaValidate:
    """Tests for arnio.schema.Schema.validate() method."""

    def test_validate_valid_data_passes(self):
        """Valid data matching schema should pass validation."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35]
        })
        frame = ar.from_pandas(df)
        
        schema = ar.Schema({
            "id": ar.Int64(),
            "name": ar.String(),
            "age": ar.Int64()
        })
        
        result = schema.validate(frame)
        assert result.passed is True
        assert result.issue_count == 0
        assert len(result.issues) == 0

    def test_validate_nullable_false_catches_null(self):
        """Non-nullable field should reject null values."""
        df = pd.DataFrame({
            "id": [1, 2, None],
            "name": ["Alice", "Bob", "Charlie"]
        })
        frame = ar.from_pandas(df)
        
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "name": ar.String()
        })
        
        result = schema.validate(frame)
        assert result.passed is False
        assert result.issue_count > 0
        assert any(issue.column == "id" for issue in result.issues)

    def test_validate_nullable_true_allows_null(self):
        """Nullable field should allow null values."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "phone": ["123-456", None, "789-012"]
        })
        frame = ar.from_pandas(df)
        
        schema = ar.Schema({
            "id": ar.Int64(),
            "phone": ar.String(nullable=True)
        })
        
        result = schema.validate(frame)
        assert result.passed is True

    def test_validate_unique_catches_duplicates(self):
        """Unique constraint should detect duplicate values."""
        df = pd.DataFrame({
            "id": [1, 2, 2],
            "name": ["Alice", "Bob", "Charlie"]
        })
        frame = ar.from_pandas(df)
        
        schema = ar.Schema({
            "id": ar.Int64(unique=True),
            "name": ar.String()
        })
        
        result = schema.validate(frame)
        assert result.passed is False
        assert any(issue.column == "id" for issue in result.issues)

    def test_validate_int64_min_constraint(self):
        """Int64 with min constraint should catch violations."""
        df = pd.DataFrame({
            "age": [10, 25, 30]
        })
        frame = ar.from_pandas(df)
        
        schema = ar.Schema({
            "age": ar.Int64(min=18)
        })
        
        result = schema.validate(frame)
        assert result.passed is False
        # First row has age=10 which is < 18
        assert any(issue.row_index == 1 for issue in result.issues)

    def test_validate_float64_max_constraint(self):
        """Float64 with max constraint should catch violations."""
        df = pd.DataFrame({
            "score": [0.5, 0.8, 1.2]
        })
        frame = ar.from_pandas(df)
        
        schema = ar.Schema({
            "score": ar.Float64(max=1.0)
        })
        
        result = schema.validate(frame)
        assert result.passed is False
        # Third row has score=1.2 which is > 1.0
        assert any(issue.row_index == 3 for issue in result.issues)

    def test_validate_string_min_length(self):
        """String min_length should catch values below minimum."""
        df = pd.DataFrame({
            "username": ["a", "bob", "alice"]
        })
        frame = ar.from_pandas(df)
        
        schema = ar.Schema({
            "username": ar.String(min_length=3)
        })
        
        result = schema.validate(frame)
        assert result.passed is False
        # First row "a" is length 1 < 3
        assert any(issue.row_index == 1 for issue in result.issues)

    def test_validate_string_max_length(self):
        """String max_length should catch values above maximum."""
        df = pd.DataFrame({
            "username": ["ab", "bob", "verylongusernamehere"]
        })
        frame = ar.from_pandas(df)
        
        schema = ar.Schema({
            "username": ar.String(max_length=10)
        })
        
        result = schema.validate(frame)
        assert result.passed is False
        # Third row is > 10 characters
        assert any(issue.row_index == 3 for issue in result.issues)

    def test_validate_email_format(self):
        """Email field should validate email format."""
        df = pd.DataFrame({
            "email": ["alice@example.com", "not-an-email", "bob@test.org"]
        })
        frame = ar.from_pandas(df)
        
        schema = ar.Schema({
            "email": ar.Email()
        })
        
        result = schema.validate(frame)
        assert result.passed is False
        # Second row has invalid email format
        assert any(issue.column == "email" for issue in result.issues)

    def test_validate_url_format(self):
        """URL field should validate URL format."""
        df = pd.DataFrame({
            "website": ["https://example.com", "not a url", "http://test.org"]
        })
        frame = ar.from_pandas(df)
        
        schema = ar.Schema({
            "website": ar.URL()
        })
        
        result = schema.validate(frame)
        assert result.passed is False

    def test_validate_type_mismatch_int64(self):
        """Int64 field should reject non-integer string values."""
        df = pd.DataFrame({
            "age": ["not_a_number", "25", "30"]
        })
        frame = ar.from_pandas(df)
        
        schema = ar.Schema({
            "age": ar.Int64()
        })
        
        result = schema.validate(frame)
        assert result.passed is False

    def test_validate_max_errors_limits_issues(self):
        """max_errors parameter should limit number of reported issues."""
        df = pd.DataFrame({
            "age": [10, 15, 20, 5, 8],
            "score": [-1, -2, -3, -4, -5]
        })
        frame = ar.from_pandas(df)
        
        schema = ar.Schema({
            "age": ar.Int64(min=18),
            "score": ar.Float64(min=0)
        })
        
        result_limited = schema.validate(frame, max_errors=3)
        assert len(result_limited.issues) <= 3

    def test_validate_row_index_one_based(self):
        """row_index should be 1-based (first data row is 1, not 0)."""
        df = pd.DataFrame({
            "id": [1, 2, "invalid"]
        })
        frame = ar.from_pandas(df)
        
        schema = ar.Schema({
            "id": ar.Int64()
        })
        
        result = schema.validate(frame)
        # Third row (index=2 in 0-based) should be row_index=3 (1-based)
        assert any(issue.row_index == 3 for issue in result.issues)

    def test_validate_multi_column_schema(self):
        """Validation should handle multiple columns with different constraints."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "email": ["alice@test.com", None, "charlie@test.com"],
            "age": [25, 30, 17]
        })
        frame = ar.from_pandas(df)
        
        schema = ar.Schema({
            "id": ar.Int64(unique=True, nullable=False),
            "email": ar.Email(nullable=True),
            "age": ar.Int64(min=18)
        })
        
        result = schema.validate(frame)
        assert result.passed is False
        # Should have issue for age=17 < 18 on row 3
        assert any(issue.column == "age" and issue.row_index == 3 for issue in result.issues)

    def test_validate_bool_type(self):
        """Bool field should validate boolean type."""
        df = pd.DataFrame({
            "active": [True, False, True]
        })
        frame = ar.from_pandas(df)
        
        schema = ar.Schema({
            "active": ar.Bool()
        })
        
        result = schema.validate(frame)
        assert result.passed is True

    def test_validate_datetime_format(self):
        """DateTime field should validate datetime format."""
        df = pd.DataFrame({
            "created_at": ["2023-01-01", "2023-02-15", "not-a-date"]
        })
        frame = ar.from_pandas(df)
        
        schema = ar.Schema({
            "created_at": ar.DateTime(format="%Y-%m-%d")
        })
        
        result = schema.validate(frame)
        assert result.passed is False
        # Third row has invalid date format
        assert any(issue.row_index == 3 for issue in result.issues)

    def test_validate_result_object_structure(self):
        """ValidationResult should have expected attributes and methods."""
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
        
        # Check result has expected attributes
        assert hasattr(result, "passed")
        assert hasattr(result, "issues")
        assert hasattr(result, "issue_count")
        
        # Check method availability
        assert callable(result.summary)
        assert callable(result.to_markdown)
        
        # Verify types
        assert isinstance(result.passed, bool)
        assert isinstance(result.issues, list)
        assert isinstance(result.issue_count, int)
