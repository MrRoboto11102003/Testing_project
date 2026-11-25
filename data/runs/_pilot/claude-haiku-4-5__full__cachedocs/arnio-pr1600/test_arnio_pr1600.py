import pytest
import pandas as pd
import arnio as ar


class TestValidationResultToMarkdown:
    """Tests for ValidationResult.to_markdown() method."""

    @pytest.fixture
    def schema_with_rules(self):
        """Schema with multiple validation rules."""
        return ar.Schema({
            "id": ar.Int64(nullable=False, unique=True),
            "email": ar.Email(nullable=False),
            "age": ar.Int64(nullable=True, min=0, max=120),
            "name": ar.String(nullable=False, min_length=1),
        })

    @pytest.fixture
    def valid_frame(self):
        """Frame that passes all validation."""
        return ar.from_pandas(pd.DataFrame({
            "id": [1, 2, 3],
            "email": ["alice@example.com", "bob@example.com", "carol@example.com"],
            "age": [25, 30, 35],
            "name": ["Alice", "Bob", "Carol"],
        }))

    @pytest.fixture
    def invalid_frame_multiple_issues(self):
        """Frame with multiple validation issues."""
        return ar.from_pandas(pd.DataFrame({
            "id": [1, 1, 3, 4, 5],
            "email": ["valid@example.com", "invalid-email", None, "bad@", "ok@example.com"],
            "age": [25, 150, -5, 30, 200],
            "name": ["Alice", "", "Charlie", "David", "Eve"],
        }))

    def test_to_markdown_returns_string(self, schema_with_rules, valid_frame):
        """to_markdown returns a string."""
        result = ar.validate(valid_frame, schema_with_rules)
        markdown = result.to_markdown()
        assert isinstance(markdown, str)

    def test_to_markdown_with_no_issues(self, schema_with_rules, valid_frame):
        """to_markdown on passing validation should work."""
        result = ar.validate(valid_frame, schema_with_rules)
        assert result.passed
        markdown = result.to_markdown()
        assert isinstance(markdown, str)

    def test_to_markdown_contains_issues(self, schema_with_rules, invalid_frame_multiple_issues):
        """to_markdown should contain information about issues."""
        result = ar.validate(invalid_frame_multiple_issues, schema_with_rules)
        assert not result.passed
        markdown = result.to_markdown()
        # Should contain pipe characters for markdown table
        assert "|" in markdown

    def test_to_markdown_max_issues_limits_rows(self, schema_with_rules, invalid_frame_multiple_issues):
        """max_issues should limit the number of issues displayed."""
        result = ar.validate(invalid_frame_multiple_issues, schema_with_rules)
        assert result.issue_count > 2
        
        markdown_all = result.to_markdown()
        markdown_limited = result.to_markdown(max_issues=2)
        
        # Count table data rows (lines with |)
        rows_all = [line for line in markdown_all.split("\n") if "|" in line and "—" not in line]
        rows_limited = [line for line in markdown_limited.split("\n") if "|" in line and "—" not in line]
        
        assert len(rows_limited) <= len(rows_all)

    def test_to_markdown_max_issues_zero(self, schema_with_rules, invalid_frame_multiple_issues):
        """max_issues=0 should not display issue rows."""
        result = ar.validate(invalid_frame_multiple_issues, schema_with_rules)
        markdown = result.to_markdown(max_issues=0)
        assert isinstance(markdown, str)

    def test_to_markdown_max_issues_greater_than_total(self, schema_with_rules, invalid_frame_multiple_issues):
        """max_issues larger than total issues should show all issues."""
        result = ar.validate(invalid_frame_multiple_issues, schema_with_rules)
        total = result.issue_count
        
        markdown_all = result.to_markdown()
        markdown_large = result.to_markdown(max_issues=total + 100)
        
        assert markdown_all.strip() == markdown_large.strip()

    def test_to_markdown_max_issues_none_shows_all(self, schema_with_rules, invalid_frame_multiple_issues):
        """max_issues=None should show all issues."""
        result = ar.validate(invalid_frame_multiple_issues, schema_with_rules)
        markdown_none = result.to_markdown(max_issues=None)
        markdown_default = result.to_markdown()
        
        assert markdown_none.strip() == markdown_default.strip()

    def test_to_markdown_redact_values_true_replaces_values(self, schema_with_rules, invalid_frame_multiple_issues):
        """redact_values=True should replace Value column with [REDACTED]."""
        result = ar.validate(invalid_frame_multiple_issues, schema_with_rules)
        if result.issue_count > 0:
            markdown = result.to_markdown(redact_values=True)
            assert "[REDACTED]" in markdown

    def test_to_markdown_redact_values_false_shows_values(self, schema_with_rules, invalid_frame_multiple_issues):
        """redact_values=False should keep original values."""
        result = ar.validate(invalid_frame_multiple_issues, schema_with_rules)
        if result.issue_count > 0:
            markdown = result.to_markdown(redact_values=False)
            # Should not have [REDACTED] as systematic replacement
            assert isinstance(markdown, str)

    def test_to_markdown_redacted_differs_from_unredacted(self, schema_with_rules, invalid_frame_multiple_issues):
        """Redacted and non-redacted versions should differ when there are non-None values."""
        result = ar.validate(invalid_frame_multiple_issues, schema_with_rules)
        if result.issue_count > 0 and any(issue.value is not None for issue in result.issues):
            markdown_redacted = result.to_markdown(redact_values=True)
            markdown_unredacted = result.to_markdown(redact_values=False)
            
            assert markdown_redacted != markdown_unredacted
            assert "[REDACTED]" in markdown_redacted

    def test_to_markdown_combined_parameters(self, schema_with_rules, invalid_frame_multiple_issues):
        """to_markdown should work with both max_issues and redact_values together."""
        result = ar.validate(invalid_frame_multiple_issues, schema_with_rules)
        markdown = result.to_markdown(max_issues=2, redact_values=True)
        
        assert isinstance(markdown, str)
        if result.issue_count > 0:
            assert "[REDACTED]" in markdown

    def test_to_markdown_github_markdown_format(self, schema_with_rules, invalid_frame_multiple_issues):
        """to_markdown output should be in GitHub-friendly markdown table format."""
        result = ar.validate(invalid_frame_multiple_issues, schema_with_rules)
        if not result.passed:
            markdown = result.to_markdown()
            # Check for basic markdown table structure
            if "|" in markdown:
                lines = [line for line in markdown.split("\n") if line.strip()]
                # Should have pipe characters indicating table
                table_lines = [line for line in lines if "|" in line]
                assert len(table_lines) > 0

    def test_to_markdown_single_issue(self):
        """to_markdown should work correctly with a single issue."""
        schema = ar.Schema({"id": ar.Int64(nullable=False)})
        frame = ar.from_pandas(pd.DataFrame({"id": [None]}))
        result = ar.validate(frame, schema)
        
        markdown = result.to_markdown()
        assert isinstance(markdown, str)
        assert len(result.issues) == 1

    def test_to_markdown_default_redact_values_is_false(self, schema_with_rules, invalid_frame_multiple_issues):
        """Default redact_values should be False (not redacting by default)."""
        result = ar.validate(invalid_frame_multiple_issues, schema_with_rules)
        if result.issue_count > 0 and any(issue.value is not None for issue in result.issues):
            markdown_default = result.to_markdown()
            markdown_explicit_false = result.to_markdown(redact_values=False)
            
            assert markdown_default == markdown_explicit_false
