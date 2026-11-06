import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestSchemaValidate:
    """Tests for arnio.schema.Schema.validate method."""
    
    def test_validate_passing_simple_schema(self):
        """Valid data against schema should pass."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "name": ar.String(nullable=False),
        })
        
        result = schema.validate(frame)
        
        assert result.passed is True
        assert result.issue_count == 0
    
    def test_validate_failing_dtype(self):
        """Data with wrong dtype should fail validation."""
        df = pd.DataFrame({
            "age": ["not_a_number", "also_not"],
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "age": ar.Int64(),
        })
        
        result = schema.validate(frame)
        
        assert result.passed is False
        assert result.issue_count > 0
        assert any(issue.rule == "dtype" for issue in result.issues)
    
    def test_validate_nullable_field_with_nulls_passes(self):
        """Nullable fields should accept null values."""
        df = pd.DataFrame({
            "optional_email": [None, "test@example.com", None],
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "optional_email": ar.Email(nullable=True),
        })
        
        result = schema.validate(frame)
        
        assert result.passed is True
    
    def test_validate_non_nullable_with_nulls_fails(self):
        """Non-nullable fields with nulls should fail."""
        df = pd.DataFrame({
            "id": [1, None, 3],
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
        })
        
        result = schema.validate(frame)
        
        assert result.passed is False
        assert any(issue.rule == "nullable" for issue in result.issues)
    
    def test_validate_numeric_min_constraint(self):
        """Values below minimum should fail validation."""
        df = pd.DataFrame({
            "age": [5, 25, 35],
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "age": ar.Int64(min=18),
        })
        
        result = schema.validate(frame)
        
        assert result.passed is False
        assert any(issue.rule == "min" for issue in result.issues)
    
    def test_validate_numeric_max_constraint(self):
        """Values above maximum should fail validation."""
        df = pd.DataFrame({
            "score": [85.5, 95.0, 150.0],
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "score": ar.Float64(max=100.0),
        })
        
        result = schema.validate(frame)
        
        assert result.passed is False
        assert any(issue.rule == "max" for issue in result.issues)
    
    def test_validate_email_format(self):
        """Invalid email addresses should fail validation."""
        df = pd.DataFrame({
            "email": ["valid@example.com", "invalid-email", "good@test.org"],
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "email": ar.Email(nullable=False),
        })
        
        result = schema.validate(frame)
        
        assert result.passed is False
        assert any(issue.rule == "email_format" for issue in result.issues)
    
    def test_validate_url_format(self):
        """Invalid URLs should fail validation."""
        df = pd.DataFrame({
            "website": ["https://example.com", "not a url", "http://test.org"],
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "website": ar.URL(nullable=False),
        })
        
        result = schema.validate(frame)
        
        assert result.passed is False
        assert any(issue.rule == "url_format" for issue in result.issues)
    
    def test_validate_unique_constraint(self):
        """Duplicate values in unique column should fail validation."""
        df = pd.DataFrame({
            "user_id": [1, 2, 2, 3],
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "user_id": ar.Int64(nullable=False, unique=True),
        })
        
        result = schema.validate(frame)
        
        assert result.passed is False
        assert any(issue.rule == "unique" for issue in result.issues)
    
    def test_validate_missing_required_column(self):
        """Missing required column should fail validation."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "id": ar.Int64(),
            "email": ar.Email(nullable=False),
        })
        
        result = schema.validate(frame)
        
        assert result.passed is False
        assert any(issue.rule == "required_column" for issue in result.issues)
    
    def test_validate_max_errors_limits_issues(self):
        """max_errors parameter should limit number of issues returned."""
        df = pd.DataFrame({
            "age": [-5, -10, -15, -20],
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "age": ar.Int64(min=0),
        })
        
        result_unlimited = schema.validate(frame)
        assert result_unlimited.issue_count == 4
        
        result_limited = schema.validate(frame, max_errors=2)
        assert result_limited.issue_count == 2
    
    def test_validate_max_errors_none_returns_all(self):
        """max_errors=None should return all issues."""
        df = pd.DataFrame({
            "value": [1, -1, -2, -3, -4, -5],
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "value": ar.Int64(min=0),
        })
        
        result = schema.validate(frame, max_errors=None)
        
        assert result.issue_count == 5
    
    def test_validate_warning_severity_passes(self):
        """Issues with warning severity should not fail validation."""
        df = pd.DataFrame({
            "age": [5, 15, 25],
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "age": ar.Int64(min=18, severity="warning"),
        })
        
        result = schema.validate(frame)
        
        assert result.passed is True
        assert result.issue_count > 0  # Warnings still reported
    
    def test_validate_cross_field_rule(self):
        """Cross-field validation rules should be applied."""
        def date_order_rule(df):
            issues = []
            for i, row in df.iterrows():
                if str(row["end_date"]) < str(row["start_date"]):
                    issues.append(
                        ar.ValidationIssue(
                            column="end_date",
                            rule="cross_field",
                            message="end_date must be >= start_date",
                            row_index=int(i) + 1,
                        )
                    )
            return issues
        
        df = pd.DataFrame({
            "start_date": ["2024-01-01", "2024-02-01"],
            "end_date": ["2024-12-31", "2024-01-15"],
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema(
            {
                "start_date": ar.String(),
                "end_date": ar.String(),
            },
            rules=[date_order_rule],
        )
        
        result = schema.validate(frame)
        
        assert result.passed is False
        assert any(issue.rule == "cross_field" for issue in result.issues)
    
    def test_validate_empty_frame(self):
        """Validation should succeed on empty frame (no data to violate schema)."""
        df = pd.DataFrame({
            "id": pd.Series([], dtype="int64"),
            "name": pd.Series([], dtype="object"),
        })
        frame = ar.from_pandas(df)
        schema = ar.Schema({
            "id": ar.Int64(),
            "name": ar.String(),
        })
        
        result = schema.validate(frame)
        
        assert result.issue_count == 0
