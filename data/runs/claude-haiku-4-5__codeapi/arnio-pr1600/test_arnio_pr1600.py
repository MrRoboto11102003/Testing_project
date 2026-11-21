import pytest
from arnio.schema import ValidationResult, ValidationIssue


class TestValidationResultToMarkdown:
    """Tests for ValidationResult.to_markdown() method."""

    def test_passed_validation_basic_header(self):
        """Test markdown output for passed validation includes correct header."""
        result = ValidationResult(
            row_count=100,
            issue_count=0,
            issues=[],
            bad_rows=[]
        )
        markdown = result.to_markdown()
        
        assert "## Validation Report" in markdown
        assert "Status: **passed**" in markdown
        assert "Rows checked: 100" in markdown
        assert "Issues found: 0" in markdown
        assert "Bad rows: 0" in markdown

    def test_failed_validation_includes_table(self):
        """Test markdown output for failed validation includes issue table."""
        issues = [
            ValidationIssue(
                column="email",
                rule="Email",
                message="Invalid email format",
                row_index=0,
                value="not-an-email",
                severity="error"
            )
        ]
        result = ValidationResult(
            row_count=100,
            issue_count=1,
            issues=issues,
            bad_rows=[0]
        )
        markdown = result.to_markdown()
        
        assert "Status: **failed**" in markdown
        assert "| Column | Rule | Severity | Row | Value | Message |" in markdown
        assert "email" in markdown
        assert "Email" in markdown
        assert "Invalid email format" in markdown

    def test_max_issues_parameter_limits_table_rows(self):
        """Test max_issues parameter limits visible issues in output."""
        issues = [
            ValidationIssue(
                column=f"col{i}",
                rule="test_rule",
                message=f"Error {i}",
                row_index=i,
                value=i,
                severity="error"
            )
            for i in range(10)
        ]
        result = ValidationResult(
            row_count=100,
            issue_count=10,
            issues=issues,
            bad_rows=list(range(10))
        )
        markdown = result.to_markdown(max_issues=3)
        
        assert "Showing 3 of 10 issues" in markdown
        assert "col0" in markdown
        assert "col2" in markdown
        assert "col9" not in markdown

    def test_max_issues_zero_omits_table(self):
        """Test max_issues=0 omits issue table but shows summary."""
        issues = [
            ValidationIssue(
                column="col",
                rule="test",
                message="Issue",
                row_index=0,
                value="val",
                severity="error"
            )
        ]
        result = ValidationResult(
            row_count=10,
            issue_count=1,
            issues=issues,
            bad_rows=[0]
        )
        markdown = result.to_markdown(max_issues=0)
        
        assert "_Issue table omitted by `max_issues=0`._" in markdown
        assert "| Column | Rule |" not in markdown
        assert "Issues found: 1" in markdown

    def test_redact_values_true_hides_data(self):
        """Test redact_values=True replaces values with [REDACTED]."""
        issues = [
            ValidationIssue(
                column="ssn",
                rule="format",
                message="Invalid SSN",
                row_index=0,
                value="123-45-6789",
                severity="error"
            )
        ]
        result = ValidationResult(
            row_count=100,
            issue_count=1,
            issues=issues,
            bad_rows=[0]
        )
        
        markdown = result.to_markdown(redact_values=True)
        assert "[REDACTED]" in markdown
        assert "123-45-6789" not in markdown

    def test_redact_values_false_shows_data(self):
        """Test redact_values=False (default) shows actual values."""
        issues = [
            ValidationIssue(
                column="col",
                rule="test",
                message="Invalid",
                row_index=0,
                value="sensitive_data",
                severity="error"
            )
        ]
        result = ValidationResult(
            row_count=10,
            issue_count=1,
            issues=issues,
            bad_rows=[0]
        )
        
        markdown = result.to_markdown(redact_values=False)
        assert "sensitive_data" in markdown

    def test_max_issues_type_error_with_string(self):
        """Test max_issues with string type raises TypeError."""
        result = ValidationResult(row_count=10, issue_count=0, issues=[], bad_rows=[])
        
        with pytest.raises(TypeError, match="max_issues must be an integer or None"):
            result.to_markdown(max_issues="5")

    def test_max_issues_type_error_with_bool(self):
        """Test max_issues with bool type raises TypeError (bool subclasses int)."""
        result = ValidationResult(row_count=10, issue_count=0, issues=[], bad_rows=[])
        
        with pytest.raises(TypeError, match="max_issues must be an integer or None"):
            result.to_markdown(max_issues=True)

    def test_max_issues_type_error_with_float(self):
        """Test max_issues with float type raises TypeError."""
        result = ValidationResult(row_count=10, issue_count=0, issues=[], bad_rows=[])
        
        with pytest.raises(TypeError, match="max_issues must be an integer or None"):
            result.to_markdown(max_issues=3.14)

    def test_max_issues_negative_raises_value_error(self):
        """Test negative max_issues raises ValueError."""
        result = ValidationResult(row_count=10, issue_count=0, issues=[], bad_rows=[])
        
        with pytest.raises(ValueError, match="max_issues must be non-negative"):
            result.to_markdown(max_issues=-1)

    def test_redact_values_type_error_with_string(self):
        """Test redact_values with string type raises TypeError."""
        result = ValidationResult(row_count=10, issue_count=0, issues=[], bad_rows=[])
        
        with pytest.raises(TypeError, match="redact_values must be a bool"):
            result.to_markdown(redact_values="true")

    def test_redact_values_type_error_with_int(self):
        """Test redact_values with int type raises TypeError."""
        result = ValidationResult(row_count=10, issue_count=0, issues=[], bad_rows=[])
        
        with pytest.raises(TypeError, match="redact_values must be a bool"):
            result.to_markdown(redact_values=1)

    def test_multiple_severity_levels_displayed(self):
        """Test multiple severity levels are included in output."""
        issues = [
            ValidationIssue(
                column="col1",
                rule="test",
                message="Error",
                row_index=0,
                value="val1",
                severity="error"
            ),
            ValidationIssue(
                column="col2",
                rule="test",
                message="Warning",
                row_index=1,
                value="val2",
                severity="warning"
            )
        ]
        result = ValidationResult(
            row_count=100,
            issue_count=2,
            issues=issues,
            bad_rows=[0, 1]
        )
        
        markdown = result.to_markdown()
        assert "error" in markdown
        assert "warning" in markdown
        assert "col1" in markdown
        assert "col2" in markdown

    def test_max_issues_greater_than_total_no_truncation_message(self):
        """Test max_issues greater than total issues doesn't show truncation message."""
        issues = [
            ValidationIssue(
                column=f"col{i}",
                rule="test",
                message=f"Issue {i}",
                row_index=i,
                value=i,
                severity="error"
            )
            for i in range(3)
        ]
        result = ValidationResult(
            row_count=10,
            issue_count=3,
            issues=issues,
            bad_rows=[0, 1, 2]
        )
        
        markdown = result.to_markdown(max_issues=10)
        assert "Showing" not in markdown
        assert "col0" in markdown
        assert "col2" in markdown

    def test_none_values_in_issue_fields(self):
        """Test issue with None column and row_index is handled correctly."""
        issues = [
            ValidationIssue(
                column=None,
                rule="global_check",
                message="Validation error",
                row_index=None,
                value=None,
                severity="error"
            )
        ]
        result = ValidationResult(
            row_count=100,
            issue_count=1,
            issues=issues,
            bad_rows=[]
        )
        
        markdown = result.to_markdown()
        assert "| Column | Rule |" in markdown
        assert "global_check" in markdown
        assert "Validation error" in markdown
