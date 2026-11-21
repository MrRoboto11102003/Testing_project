import pytest
import numpy as np
import pandas as pd
from arnio.schema import ValidationIssue, ValidationResult


class TestValidationResultToMarkdown:
    """Test ValidationResult.to_markdown() method."""
    
    def test_max_issues_bool_rejected(self):
        """Boolean values are rejected for max_issues."""
        result = ValidationResult(row_count=0, issue_count=0, issues=[], bad_rows=[])
        with pytest.raises(TypeError, match="max_issues must be an integer or None"):
            result.to_markdown(max_issues=True)
    
    def test_max_issues_float_rejected(self):
        """Float values are rejected for max_issues."""
        result = ValidationResult(row_count=0, issue_count=0, issues=[], bad_rows=[])
        with pytest.raises(TypeError, match="max_issues must be an integer or None"):
            result.to_markdown(max_issues=1.5)
    
    def test_max_issues_negative_rejected(self):
        """Negative max_issues raises ValueError."""
        result = ValidationResult(row_count=0, issue_count=0, issues=[], bad_rows=[])
        with pytest.raises(ValueError, match="max_issues must be non-negative"):
            result.to_markdown(max_issues=-1)
    
    def test_redact_values_int_rejected(self):
        """Integer values are rejected for redact_values."""
        result = ValidationResult(row_count=0, issue_count=0, issues=[], bad_rows=[])
        with pytest.raises(TypeError, match="redact_values must be a bool"):
            result.to_markdown(redact_values=1)
    
    def test_header_only_when_no_issues(self):
        """No issues returns only the header section."""
        result = ValidationResult(row_count=10, issue_count=0, issues=[], bad_rows=[])
        output = result.to_markdown()
        
        expected_lines = [
            "## Validation Report",
            "",
            "- Status: **passed**",
            "- Rows checked: 10",
            "- Issues found: 0",
            "- Bad rows: 0",
        ]
        assert output == "\n".join(expected_lines)
    
    def test_failed_status_with_error_severity(self):
        """Error issues result in failed status and table."""
        issue = ValidationIssue(
            column="age",
            rule="min",
            severity="error",
            message="Age below minimum",
            row_index=2,
            value=-5
        )
        result = ValidationResult(row_count=10, issue_count=1, issues=[issue], bad_rows=[2])
        output = result.to_markdown()
        
        assert "- Status: **failed**" in output
        assert "| Column | Rule | Severity | Row | Value | Message |" in output
        assert "| age | min | error | 2 | -5 | Age below minimum |" in output
    
    def test_max_issues_zero_omits_table(self):
        """max_issues=0 shows header but adds omitted message."""
        issue = ValidationIssue(
            column="age", rule="min", severity="error",
            message="Too small", row_index=1, value=0
        )
        result = ValidationResult(row_count=5, issue_count=1, issues=[issue], bad_rows=[1])
        output = result.to_markdown(max_issues=0)
        
        assert "## Validation Report" in output
        assert "_Issue table omitted by `max_issues=0`._" in output
        assert "| Column | Rule |" not in output
    
    def test_max_issues_partial_with_footer(self):
        """Showing fewer issues than available adds footer."""
        issues = [
            ValidationIssue(column="a", rule="r1", severity="error", message="m1", row_index=1, value=1),
            ValidationIssue(column="b", rule="r2", severity="error", message="m2", row_index=2, value=2),
            ValidationIssue(column="c", rule="r3", severity="error", message="m3", row_index=3, value=3),
        ]
        result = ValidationResult(row_count=10, issue_count=3, issues=issues, bad_rows=[1, 2, 3])
        output = result.to_markdown(max_issues=2)
        
        assert "| a | r1 |" in output
        assert "| b | r2 |" in output
        assert "| c | r3 |" not in output
        assert "_Showing 2 of 3 issues._" in output
    
    def test_max_issues_all_no_footer(self):
        """When max_issues shows all issues, no footer."""
        issues = [
            ValidationIssue(column="a", rule="r1", severity="error", message="m1", row_index=1, value=1),
            ValidationIssue(column="b", rule="r2", severity="error", message="m2", row_index=2, value=2),
        ]
        result = ValidationResult(row_count=10, issue_count=2, issues=issues, bad_rows=[1, 2])
        output = result.to_markdown(max_issues=100)
        
        assert "| a | r1 |" in output
        assert "| b | r2 |" in output
        assert "Showing" not in output
    
    def test_redact_values_hides_sensitive_data(self):
        """redact_values=True replaces all values with [REDACTED]."""
        issue = ValidationIssue(
            column="password", rule="length", severity="error",
            message="Too short", row_index=1, value="secret123"
        )
        result = ValidationResult(row_count=5, issue_count=1, issues=[issue], bad_rows=[1])
        output = result.to_markdown(redact_values=True)
        
        assert "[REDACTED]" in output
        assert "secret123" not in output
    
    def test_pipe_character_escaping(self):
        """Pipe characters in values are escaped."""
        issue = ValidationIssue(
            column="data", rule="format", severity="error",
            message="Invalid", row_index=1, value="a|b"
        )
        result = ValidationResult(row_count=5, issue_count=1, issues=[issue], bad_rows=[1])
        output = result.to_markdown()
        
        assert "a\\|b" in output
    
    def test_newline_to_br_conversion(self):
        """Newlines in values are converted to <br>."""
        issue = ValidationIssue(
            column="text", rule="format", severity="error",
            message="Bad", row_index=1, value="line1\nline2"
        )
        result = ValidationResult(row_count=5, issue_count=1, issues=[issue], bad_rows=[1])
        output = result.to_markdown()
        
        assert "line1<br>line2" in output
    
    def test_numpy_scalar_conversion(self):
        """NumPy scalars are converted to Python types."""
        issue = ValidationIssue(
            column="count", rule="range", severity="error",
            message="Out of range", row_index=1, value=np.int64(42)
        )
        result = ValidationResult(row_count=5, issue_count=1, issues=[issue], bad_rows=[1])
        output = result.to_markdown()
        
        assert "| 42 |" in output
    
    def test_warning_severity_with_table(self):
        """Warning-only issues still show table with status passed."""
        issue = ValidationIssue(
            column="age", rule="recommended_min", severity="warning",
            message="Below recommended", row_index=1, value=16
        )
        result = ValidationResult(row_count=5, issue_count=1, issues=[issue], bad_rows=[1])
        output = result.to_markdown()
        
        assert "- Status: **passed**" in output
        assert "| Column | Rule | Severity | Row | Value | Message |" in output
        assert "| warning |" in output
