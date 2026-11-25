import pytest
import arnio as ar
from unittest.mock import Mock


def create_mock_issue(column="test_col", rule="test_rule", severity="error",
                      row_index=0, value="test_value", message="test_message"):
    """Create a mock issue object."""
    issue = Mock()
    issue.column = column
    issue.rule = rule
    issue.severity = severity
    issue.row_index = row_index
    issue.value = value
    issue.message = message
    return issue


def create_mock_result(passed=True, row_count=0, issue_count=0, bad_rows=None, issues=None):
    """Create a mock ValidationResult object."""
    result = Mock()
    result.passed = passed
    result.row_count = row_count
    result.issue_count = issue_count
    result.bad_rows = bad_rows or []
    result.issues = issues or []
    return result


class TestValidationResultToMarkdown:
    """Test suite for arnio.schema.ValidationResult.to_markdown method."""

    def test_passed_validation_no_issues(self):
        """Test markdown output for passed validation."""
        result = create_mock_result(passed=True, row_count=100, issue_count=0)
        markdown = ar.schema.ValidationResult.to_markdown(result)

        assert "## Validation Report" in markdown
        assert "- Status: **passed**" in markdown
        assert "- Rows checked: 100" in markdown
        assert "- Issues found: 0" in markdown
        assert "- Bad rows: 0" in markdown

    def test_failed_validation_with_issues(self):
        """Test markdown output for failed validation."""
        issue = create_mock_issue(
            column="email", rule="format_check", severity="error",
            row_index=5, value="bad_email", message="Invalid format"
        )
        result = create_mock_result(
            passed=False, row_count=100, issue_count=1, bad_rows=[5], issues=[issue]
        )
        markdown = ar.schema.ValidationResult.to_markdown(result)

        assert "## Validation Report" in markdown
        assert "- Status: **failed**" in markdown
        assert "| Column | Rule | Severity | Row | Value | Message |" in markdown
        assert "email" in markdown
        assert "bad_email" in markdown

    def test_max_issues_zero_omits_table(self):
        """Test that max_issues=0 omits the issue table."""
        issue = create_mock_issue(message="Some issue")
        result = create_mock_result(
            passed=False, row_count=100, issue_count=1, bad_rows=[0], issues=[issue]
        )
        markdown = ar.schema.ValidationResult.to_markdown(result, max_issues=0)

        assert "_Issue table omitted by `max_issues=0`._" in markdown
        assert "| Column |" not in markdown

    def test_max_issues_limits_visible_issues(self):
        """Test that max_issues limits displayed issues."""
        issues = [
            create_mock_issue(row_index=i, message=f"Issue {i}")
            for i in range(10)
        ]
        result = create_mock_result(
            passed=False, row_count=100, issue_count=10, bad_rows=list(range(10)), issues=issues
        )
        markdown = ar.schema.ValidationResult.to_markdown(result, max_issues=3)

        assert "_Showing 3 of 10 issues._" in markdown
        # First 3 should be visible
        for i in range(3):
            assert f"Issue {i}" in markdown

    def test_max_issues_none_shows_all(self):
        """Test that max_issues=None shows all issues (default)."""
        issues = [
            create_mock_issue(row_index=i, message=f"Issue {i}")
            for i in range(5)
        ]
        result = create_mock_result(
            passed=False, row_count=100, issue_count=5, bad_rows=list(range(5)), issues=issues
        )
        markdown = ar.schema.ValidationResult.to_markdown(result, max_issues=None)

        # All issues should be visible
        for i in range(5):
            assert f"Issue {i}" in markdown
        # No "showing X of Y" footer when all are shown
        assert "Showing" not in markdown

    def test_redact_values_true(self):
        """Test that redact_values=True shows [REDACTED]."""
        issue = create_mock_issue(value="secret_data", message="Has data")
        result = create_mock_result(
            passed=False, row_count=100, issue_count=1, bad_rows=[0], issues=[issue]
        )
        markdown = ar.schema.ValidationResult.to_markdown(result, redact_values=True)

        assert "[REDACTED]" in markdown
        assert "secret_data" not in markdown

    def test_redact_values_false(self):
        """Test that redact_values=False shows actual values."""
        issue = create_mock_issue(value="visible_data", message="Has data")
        result = create_mock_result(
            passed=False, row_count=100, issue_count=1, bad_rows=[0], issues=[issue]
        )
        markdown = ar.schema.ValidationResult.to_markdown(result, redact_values=False)

        assert "[REDACTED]" not in markdown
        assert "visible_data" in markdown

    def test_max_issues_type_error_string(self):
        """Test TypeError when max_issues is a string."""
        result = create_mock_result()
        with pytest.raises(TypeError, match="max_issues must be an integer or None"):
            ar.schema.ValidationResult.to_markdown(result, max_issues="5")

    def test_max_issues_type_error_bool(self):
        """Test TypeError when max_issues is a bool (bool is subclass of int)."""
        result = create_mock_result()
        with pytest.raises(TypeError, match="max_issues must be an integer or None"):
            ar.schema.ValidationResult.to_markdown(result, max_issues=True)

    def test_max_issues_type_error_float(self):
        """Test TypeError when max_issues is a float."""
        result = create_mock_result()
        with pytest.raises(TypeError, match="max_issues must be an integer or None"):
            ar.schema.ValidationResult.to_markdown(result, max_issues=3.14)

    def test_max_issues_value_error_negative(self):
        """Test ValueError when max_issues is negative."""
        result = create_mock_result()
        with pytest.raises(ValueError, match="max_issues must be non-negative"):
            ar.schema.ValidationResult.to_markdown(result, max_issues=-1)

    def test_redact_values_type_error(self):
        """Test TypeError when redact_values is not a bool."""
        result = create_mock_result()
        with pytest.raises(TypeError, match="redact_values must be a bool"):
            ar.schema.ValidationResult.to_markdown(result, redact_values="yes")

    def test_markdown_table_structure(self):
        """Test that markdown table has correct structure."""
        issue = create_mock_issue()
        result = create_mock_result(
            passed=False, row_count=100, issue_count=1, bad_rows=[0], issues=[issue]
        )
        markdown = ar.schema.ValidationResult.to_markdown(result)

        lines = markdown.split('\n')
        table_lines = [l for l in lines if l.startswith('|')]
        # Should have header and separator and data row
        assert len(table_lines) >= 2
        # Header should have 6 columns (Column, Rule, Severity, Row, Value, Message)
        assert "| Column | Rule | Severity | Row | Value | Message |" in markdown

    def test_multiple_issues_with_hidden_count(self):
        """Test correct footer message when issues are hidden."""
        issues = [
            create_mock_issue(row_index=i, message=f"Issue {i}")
            for i in range(15)
        ]
        result = create_mock_result(
            passed=False, row_count=200, issue_count=15, bad_rows=list(range(15)), issues=issues
        )
        markdown = ar.schema.ValidationResult.to_markdown(result, max_issues=5)

        assert "_Showing 5 of 15 issues._" in markdown
        assert "- Issues found: 15" in markdown
