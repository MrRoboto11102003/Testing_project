import pytest
import pandas as pd
import numpy as np
from arnio.schema import ValidationIssue, ValidationResult


class TestValidationResultToMarkdown:
    """Tests for ValidationResult.to_markdown method."""

    # ===== Parameter Validation =====

    def test_max_issues_type_error_string(self):
        """max_issues as string raises TypeError."""
        r = ValidationResult(row_count=1, issue_count=0, issues=[], bad_rows=[])
        with pytest.raises(TypeError, match="max_issues must be an integer or None"):
            r.to_markdown(max_issues="2")

    def test_max_issues_type_error_bool(self):
        """max_issues as bool raises TypeError."""
        r = ValidationResult(row_count=1, issue_count=0, issues=[], bad_rows=[])
        with pytest.raises(TypeError, match="max_issues must be an integer or None"):
            r.to_markdown(max_issues=True)

    def test_max_issues_negative_raises_valueerror(self):
        """max_issues negative raises ValueError."""
        r = ValidationResult(row_count=1, issue_count=0, issues=[], bad_rows=[])
        with pytest.raises(ValueError, match="max_issues must be non-negative"):
            r.to_markdown(max_issues=-1)

    def test_redact_values_type_error_int(self):
        """redact_values as int raises TypeError."""
        r = ValidationResult(row_count=1, issue_count=0, issues=[], bad_rows=[])
        with pytest.raises(TypeError, match="redact_values must be a bool"):
            r.to_markdown(redact_values=1)

    def test_error_check_order_valueerror_before_typeerror(self):
        """ValueError for max_issues checked before TypeError for redact_values."""
        r = ValidationResult(row_count=1, issue_count=0, issues=[], bad_rows=[])
        with pytest.raises(ValueError, match="max_issues must be non-negative"):
            r.to_markdown(max_issues=-1, redact_values="bad")

    # ===== Basic Functionality =====

    def test_no_issues_header_only(self):
        """No issues returns header with passed status."""
        r = ValidationResult(row_count=10, issue_count=0, issues=[], bad_rows=[])
        output = r.to_markdown()
        assert "## Validation Report" in output
        assert "- Status: **passed**" in output
        assert "- Rows checked: 10" in output
        assert "- Issues found: 0" in output
        assert "- Bad rows: 0" in output
        assert "| Column |" not in output

    def test_failed_status_with_error_issues(self):
        """Error-severity issues produce failed status."""
        issue = ValidationIssue(
            column="age", rule="min", severity="error",
            message="Values below 0", row_index=2, value=-5
        )
        r = ValidationResult(row_count=5, issue_count=1, issues=[issue], bad_rows=[2])
        assert "- Status: **failed**" in r.to_markdown()

    def test_warning_only_shows_passed_with_table(self):
        """Warning-only issues show passed status and table."""
        issue = ValidationIssue(
            column="age", rule="min", severity="warning",
            message="Values are low", row_index=2, value=-5
        )
        r = ValidationResult(row_count=5, issue_count=1, issues=[issue], bad_rows=[])
        output = r.to_markdown()
        assert "- Status: **passed**" in output
        assert "| Column | Rule | Severity | Row | Value | Message |" in output

    # ===== max_issues Parameter =====

    def test_max_issues_zero_omits_table_shows_footer(self):
        """max_issues=0 omits table but includes special footer."""
        issue = ValidationIssue(
            column="age", rule="min", severity="error",
            message="m", row_index=0, value=-5
        )
        r = ValidationResult(row_count=5, issue_count=1, issues=[issue], bad_rows=[0])
        output = r.to_markdown(max_issues=0)
        assert "## Validation Report" in output
        assert "| Column |" not in output
        assert "_Issue table omitted by `max_issues=0`._" in output

    def test_max_issues_truncates_with_footer(self):
        """max_issues < issue_count shows truncation footer."""
        issues = [
            ValidationIssue(column=f"col{i}", rule=f"r{i}", severity="error",
                          message=f"m{i}", row_index=i, value=i)
            for i in range(3)
        ]
        r = ValidationResult(row_count=5, issue_count=3, issues=issues, bad_rows=[0, 1, 2])
        output = r.to_markdown(max_issues=2)
        assert "_Showing 2 of 3 issues._" in output
        assert "| col0 |" in output
        assert "| col1 |" in output
        assert "| col2 |" not in output

    def test_max_issues_equal_count_no_footer(self):
        """max_issues equal to issue_count has no footer."""
        issues = [
            ValidationIssue(column="a", rule="r", severity="error",
                          message="m", row_index=i, value=i)
            for i in range(2)
        ]
        r = ValidationResult(row_count=5, issue_count=2, issues=issues, bad_rows=[0, 1])
        output = r.to_markdown(max_issues=2)
        assert "_Showing" not in output

    # ===== redact_values Parameter =====

    def test_redact_values_true_hides_values(self):
        """redact_values=True replaces all value cells with [REDACTED]."""
        issue = ValidationIssue(
            column="age", rule="min", severity="error",
            message="m", row_index=0, value="secret_data"
        )
        r = ValidationResult(row_count=1, issue_count=1, issues=[issue], bad_rows=[0])
        output = r.to_markdown(redact_values=True)
        assert "| [REDACTED] |" in output
        assert "secret_data" not in output

    # ===== Cell Escaping and Value Cleaning =====

    def test_cell_escaping_pipe_and_newline(self):
        """Pipes escaped as \\| and newlines as <br>."""
        issue = ValidationIssue(
            column="a", rule="r", severity="error",
            message="msg|with\npipe", row_index=0, value="a|b\nc"
        )
        r = ValidationResult(row_count=1, issue_count=1, issues=[issue], bad_rows=[0])
        output = r.to_markdown()
        assert "a\\|b<br>c" in output

    def test_value_numpy_scalar_converted(self):
        """numpy scalar values converted via .item()."""
        issue = ValidationIssue(
            column="a", rule="r", severity="error",
            message="m", row_index=0, value=np.int64(42)
        )
        r = ValidationResult(row_count=1, issue_count=1, issues=[issue], bad_rows=[0])
        assert "| 42 |" in r.to_markdown()

    def test_value_timestamp_isoformat(self):
        """pd.Timestamp converted to ISO format."""
        ts = pd.Timestamp("2025-01-15 10:30:00")
        issue = ValidationIssue(
            column="a", rule="r", severity="error",
            message="m", row_index=0, value=ts
        )
        r = ValidationResult(row_count=1, issue_count=1, issues=[issue], bad_rows=[0])
        assert "2025-01-15T10:30:00" in r.to_markdown()

    # ===== Complete Example =====

    def test_complete_example_from_documentation(self):
        """Complete example matches documented output format."""
        i = ValidationIssue(
            column="age", rule="min", severity="error",
            message="Column 'age' has values below 0", row_index=2, value=-5
        )
        r = ValidationResult(row_count=3, issue_count=1, issues=[i], bad_rows=[2])
        expected = (
            "## Validation Report\n\n"
            "- Status: **failed**\n"
            "- Rows checked: 3\n"
            "- Issues found: 1\n"
            "- Bad rows: 1\n\n"
            "| Column | Rule | Severity | Row | Value | Message |\n"
            "|---|---|---|---|---|---|\n"
            "| age | min | error | 2 | -5 | Column 'age' has values below 0 |"
        )
        assert r.to_markdown() == expected
