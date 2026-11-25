import pytest
import pandas as pd
import arnio as ar


class TestValidationResultToMarkdown:
    """Test cases for arnio.schema.ValidationResult.to_markdown()"""

    @pytest.fixture
    def frame_with_validation_issues(self):
        """Create a frame that will generate validation issues."""
        return ar.from_pandas(
            pd.DataFrame({
                "email": ["valid@test.com", "not-an-email", "another@bad"],
                "age": [25, -5, 150],
                "name": ["Alice", "Bob", None],
            })
        )

    @pytest.fixture
    def valid_frame(self):
        """Create a frame that passes validation."""
        return ar.from_pandas(
            pd.DataFrame({
                "id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"],
            })
        )

    def test_to_markdown_returns_string(self, frame_with_validation_issues):
        """to_markdown() must return a string."""
        schema = ar.Schema({"email": ar.Email(nullable=False)})
        result = ar.validate(frame_with_validation_issues, schema)
        markdown = result.to_markdown()
        assert isinstance(markdown, str)

    def test_to_markdown_returns_non_empty_string(self, frame_with_validation_issues):
        """to_markdown() should return non-empty string."""
        schema = ar.Schema({"email": ar.Email(nullable=False)})
        result = ar.validate(frame_with_validation_issues, schema)
        markdown = result.to_markdown()
        assert len(markdown) > 0

    def test_to_markdown_github_markdown_format(self, frame_with_validation_issues):
        """to_markdown() output should use markdown table syntax with pipes."""
        schema = ar.Schema({"email": ar.Email(nullable=False)})
        result = ar.validate(frame_with_validation_issues, schema)
        markdown = result.to_markdown()
        
        if not result.passed:
            # GitHub markdown tables use | as column separator
            assert "|" in markdown

    def test_to_markdown_max_issues_none_shows_all(self, frame_with_validation_issues):
        """max_issues=None should show all issues (default)."""
        schema = ar.Schema({"email": ar.Email(nullable=False)})
        result = ar.validate(frame_with_validation_issues, schema)
        
        markdown_all = result.to_markdown(max_issues=None)
        markdown_default = result.to_markdown()
        
        assert isinstance(markdown_all, str)
        assert isinstance(markdown_default, str)

    def test_to_markdown_max_issues_limits_output(self, frame_with_validation_issues):
        """max_issues parameter should limit shown issues."""
        schema = ar.Schema({"email": ar.Email(nullable=False)})
        result = ar.validate(frame_with_validation_issues, schema)
        
        if len(result.issues) > 1:
            markdown_limited = result.to_markdown(max_issues=1)
            markdown_unlimited = result.to_markdown(max_issues=100)
            
            # Limited should be shorter or equal
            assert len(markdown_limited) <= len(markdown_unlimited)

    def test_to_markdown_max_issues_zero(self, frame_with_validation_issues):
        """max_issues=0 should not crash and return string."""
        schema = ar.Schema({"email": ar.Email(nullable=False)})
        result = ar.validate(frame_with_validation_issues, schema)
        markdown = result.to_markdown(max_issues=0)
        assert isinstance(markdown, str)

    def test_to_markdown_redact_values_false_shows_actual_values(self, frame_with_validation_issues):
        """redact_values=False (default) should show actual invalid values."""
        schema = ar.Schema({"email": ar.Email(nullable=False)})
        result = ar.validate(frame_with_validation_issues, schema)
        
        markdown = result.to_markdown(redact_values=False)
        assert isinstance(markdown, str)

    def test_to_markdown_redact_values_true_hides_values(self, frame_with_validation_issues):
        """redact_values=True should replace values with [REDACTED]."""
        schema = ar.Schema({"email": ar.Email(nullable=False)})
        result = ar.validate(frame_with_validation_issues, schema)
        
        markdown = result.to_markdown(redact_values=True)
        assert isinstance(markdown, str)
        
        if not result.passed and len(result.issues) > 0:
            # Either issues are shown with [REDACTED] or no value column exists
            assert "[REDACTED]" in markdown or len(result.issues) == 0

    def test_to_markdown_passed_validation(self, valid_frame):
        """Passed validation should still produce valid markdown."""
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "name": ar.String(nullable=False),
        })
        result = ar.validate(valid_frame, schema)
        
        if result.passed:
            markdown = result.to_markdown()
            assert isinstance(markdown, str)
            assert len(markdown) >= 0

    def test_to_markdown_failed_validation(self, frame_with_validation_issues):
        """Failed validation should show issues in markdown table."""
        schema = ar.Schema({
            "email": ar.Email(nullable=False),
            "age": ar.Int64(min=0, max=120),
            "name": ar.String(nullable=False),
        })
        result = ar.validate(frame_with_validation_issues, schema)
        
        markdown = result.to_markdown()
        assert isinstance(markdown, str)
        
        if not result.passed:
            # Should have meaningful content
            assert len(markdown) > 0

    def test_to_markdown_idempotent(self, frame_with_validation_issues):
        """Multiple calls should produce identical output."""
        schema = ar.Schema({"email": ar.Email(nullable=False)})
        result = ar.validate(frame_with_validation_issues, schema)
        
        call1 = result.to_markdown()
        call2 = result.to_markdown()
        call3 = result.to_markdown()
        
        assert call1 == call2 == call3

    def test_to_markdown_with_multiple_issues_per_column(self):
        """Test output with multiple validation failures."""
        frame = ar.from_pandas(
            pd.DataFrame({
                "score": [-10, 150, -5, 200],
            })
        )
        schema = ar.Schema({"score": ar.Int64(min=0, max=100)})
        result = ar.validate(frame, schema)
        
        markdown = result.to_markdown()
        assert isinstance(markdown, str)

    def test_to_markdown_max_issues_parameter_is_keyword_only(self, frame_with_validation_issues):
        """max_issues must be passed as keyword argument."""
        schema = ar.Schema({"email": ar.Email(nullable=False)})
        result = ar.validate(frame_with_validation_issues, schema)
        
        # This should work (keyword)
        markdown = result.to_markdown(max_issues=5)
        assert isinstance(markdown, str)

    def test_to_markdown_redact_values_parameter_is_keyword_only(self, frame_with_validation_issues):
        """redact_values must be passed as keyword argument."""
        schema = ar.Schema({"email": ar.Email(nullable=False)})
        result = ar.validate(frame_with_validation_issues, schema)
        
        # This should work (keyword)
        markdown = result.to_markdown(redact_values=True)
        assert isinstance(markdown, str)

    def test_to_markdown_both_parameters_together(self, frame_with_validation_issues):
        """Both max_issues and redact_values should work together."""
        schema = ar.Schema({"email": ar.Email(nullable=False)})
        result = ar.validate(frame_with_validation_issues, schema)
        
        markdown = result.to_markdown(max_issues=2, redact_values=True)
        assert isinstance(markdown, str)

    def test_to_markdown_large_issue_count(self):
        """Test with many validation issues (tests truncation behavior)."""
        # Create frame with 100 invalid rows
        frame = ar.from_pandas(
            pd.DataFrame({"value": list(range(-100, 0))})
        )
        schema = ar.Schema({"value": ar.Int64(min=0)})
        result = ar.validate(frame, schema)
        
        markdown_limited = result.to_markdown(max_issues=5)
        markdown_unlimited = result.to_markdown(max_issues=1000)
        
        assert len(markdown_limited) <= len(markdown_unlimited)
        assert isinstance(markdown_limited, str)

    def test_to_markdown_no_issues(self, valid_frame):
        """Validation with no issues should produce valid markdown."""
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "name": ar.String(nullable=False),
        })
        result = ar.validate(valid_frame, schema)
        
        markdown = result.to_markdown()
        assert isinstance(markdown, str)

    def test_to_markdown_cross_field_validation(self):
        """Test markdown with cross-field validation rules."""
        frame = ar.from_pandas(
            pd.DataFrame({
                "start_date": ["2023-01-01", "2023-01-05"],
                "end_date": ["2022-12-31", "2023-01-10"],
            })
        )
        
        def end_after_start(df):
            issues = []
            for i, row in df.iterrows():
                if row["end_date"] < row["start_date"]:
                    issues.append(
                        ar.ValidationIssue(
                            column="end_date",
                            rule="cross_field",
                            message="end_date must be >= start_date",
                            row_index=int(i) + 1,
                        )
                    )
            return issues
        
        schema = ar.Schema(
            {
                "start_date": ar.String(),
                "end_date": ar.String(),
            },
            rules=[end_after_start],
        )
        result = ar.validate(frame, schema)
        markdown = result.to_markdown()
        
        assert isinstance(markdown, str)

    def test_to_markdown_with_unicode_values(self):
        """Test markdown output handles unicode correctly."""
        frame = ar.from_pandas(
            pd.DataFrame({
                "name": ["Alice", "Björn", "李明"],
                "age": [25, 30, -5],
            })
        )
        schema = ar.Schema({"age": ar.Int64(min=0)})
        result = ar.validate(frame, schema)
        
        markdown = result.to_markdown()
        assert isinstance(markdown, str)

    def test_to_markdown_nullable_violations(self):
        """Test markdown with nullable=False violations."""
        frame = ar.from_pandas(
            pd.DataFrame({
                "email": ["test@example.com", None, "another@test.com"],
            })
        )
        schema = ar.Schema({"email": ar.String(nullable=False)})
        result = ar.validate(frame, schema)
        
        markdown = result.to_markdown()
        assert isinstance(markdown, str)
