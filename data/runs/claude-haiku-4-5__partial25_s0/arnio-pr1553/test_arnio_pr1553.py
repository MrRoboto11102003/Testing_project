import pytest
import pandas as pd
import arnio as ar


class TestSchemaValidate:
    """Tests for arnio.schema.Schema.validate() method."""

    def test_validate_returns_validation_result(self):
        """Schema.validate() returns a ValidationResult object."""
        frame = ar.from_pandas(pd.DataFrame({"id": [1, 2, 3]}))
        schema = ar.Schema({"id": ar.Int64()})
        result = schema.validate(frame)
        assert hasattr(result, "passed")
        assert hasattr(result, "issues")

    def test_validate_passes_for_matching_schema(self):
        """Schema.validate() returns passed=True for valid data."""
        frame = ar.from_pandas(pd.DataFrame({"id": [1, 2, 3]}))
        schema = ar.Schema({"id": ar.Int64()})
        result = schema.validate(frame)
        assert result.passed is True

    def test_validate_fails_for_dtype_mismatch(self):
        """Schema.validate() fails when column dtype doesn't match schema."""
        frame = ar.from_pandas(pd.DataFrame({"age": ["20", "30"]}))
        schema = ar.Schema({"age": ar.Int64()})
        result = schema.validate(frame)
        assert result.passed is False
        assert len(result.issues) > 0

    def test_validate_issues_contain_metadata(self):
        """Schema.validate() issues include row_index and column."""
        frame = ar.from_pandas(pd.DataFrame({"age": ["invalid", "25"]}))
        schema = ar.Schema({"age": ar.Int64()})
        result = schema.validate(frame)
        assert hasattr(result.issues[0], "row_index")
        assert hasattr(result.issues[0], "column")
        assert hasattr(result.issues[0], "message")

    def test_validate_nullable_true_allows_nulls(self):
        """Schema.validate() allows null values when nullable=True."""
        frame = ar.from_pandas(pd.DataFrame({"value": [1, None, 3]}))
        schema = ar.Schema({"value": ar.Int64(nullable=True)})
        result = schema.validate(frame)
        assert result.passed is True

    def test_validate_nullable_false_rejects_nulls(self):
        """Schema.validate() rejects null values when nullable=False."""
        frame = ar.from_pandas(pd.DataFrame({"value": [1, None, 3]}))
        schema = ar.Schema({"value": ar.Int64(nullable=False)})
        result = schema.validate(frame)
        assert result.passed is False

    def test_validate_string_length_constraints(self):
        """Schema.validate() enforces String min_length and max_length."""
        frame = ar.from_pandas(pd.DataFrame({"code": ["a", "abcde"]}))
        schema = ar.Schema({"code": ar.String(min_length=2, max_length=4)})
        result = schema.validate(frame)
        assert result.passed is False

    def test_validate_numeric_bounds_constraints(self):
        """Schema.validate() enforces min and max on numeric fields."""
        frame = ar.from_pandas(pd.DataFrame({"score": [0, 50, 150]}))
        schema = ar.Schema({"score": ar.Int64(min=10, max=100)})
        result = schema.validate(frame)
        assert result.passed is False

    def test_validate_unique_single_column(self):
        """Schema.validate() enforces unique constraint on single column."""
        frame = ar.from_pandas(pd.DataFrame({"id": [1, 2, 2, 3]}))
        schema = ar.Schema({"id": ar.Int64(unique=True)})
        result = schema.validate(frame)
        assert result.passed is False

    def test_validate_composite_uniqueness_constraint(self):
        """Schema.validate() enforces composite key uniqueness."""
        frame = ar.from_pandas(pd.DataFrame({
            "user_id": [1, 1, 2],
            "course_id": [101, 101, 101]
        }))
        schema = ar.Schema({
            "user_id": ar.Int64(),
            "course_id": ar.Int64()
        }, unique=["user_id", "course_id"])
        result = schema.validate(frame)
        assert result.passed is False

    def test_validate_email_format(self):
        """Schema.validate() validates Email format."""
        frame = ar.from_pandas(pd.DataFrame({
            "email": ["not-an-email", "user@example.com"]
        }))
        schema = ar.Schema({"email": ar.Email()})
        result = schema.validate(frame)
        assert result.passed is False

    def test_validate_url_scheme_restriction(self):
        """Schema.validate() enforces URL allowed_schemes."""
        frame = ar.from_pandas(pd.DataFrame({
            "url": ["https://example.com", "http://example.com"]
        }))
        schema = ar.Schema({
            "url": ar.URL(allowed_schemes=["https"])
        })
        result = schema.validate(frame)
        assert result.passed is False

    def test_validate_regex_pattern_matching(self):
        """Schema.validate() validates Regex patterns."""
        frame = ar.from_pandas(pd.DataFrame({
            "code": ["BAD-FORM", "GOOD-0001"]
        }))
        schema = ar.Schema({
            "code": ar.Regex(r"^GOOD-\d{4}$")
        })
        result = schema.validate(frame)
        assert result.passed is False

    def test_validate_warning_severity_passes_but_reports(self):
        """Schema.validate() with severity='warning' passes but counts issues."""
        frame = ar.from_pandas(pd.DataFrame({"age": [5, 100]}))
        schema = ar.Schema({
            "age": ar.Int64(min=18, severity="warning")
        })
        result = schema.validate(frame)
        assert result.passed is True
        assert result.issue_count > 0

    def test_validate_max_errors_parameter_limits_issues(self):
        """Schema.validate() with max_errors limits number of returned issues."""
        frame = ar.from_pandas(pd.DataFrame({
            "value": ["a", "b", "c", "d", "e"]
        }))
        schema = ar.Schema({"value": ar.Int64()})
        result_limited = schema.validate(frame, max_errors=2)
        assert len(result_limited.issues) <= 2
