import pytest
from unittest.mock import Mock, patch
from arnio.schema import ValidationResult


@pytest.fixture
def mock_issue():
    """Create a mock issue object with required attributes."""
    issue = Mock()
    issue.column = "age"
    issue.rule = "range"
    issue.severity = "error"
    issue.row_index = 5
    issue.value = -1
    issue.message = "Value must be >= 0"
    return issue


@pytest.fixture
def passed_result(mock_issue):
    """Create a mock passed validation result."""
    result = Mock(spec=ValidationResult)
    result.passed = True
    result.row_count = 100
    result.issue_count = 0
    result.bad_rows = []
    result.issues = []
    result.to_markdown = ValidationResult.to_markdown.__get__(result, ValidationResult)
    return result


@pytest.fixture
def failed_result_one_issue(mock_issue):
    """Create a mock failed validation result with one issue."""
    result = Mock(spec=ValidationResult)
    result.passed = False
    result.row_count = 100
    result.issue_count = 1
    result.bad_rows = [5]
    result.issues = [mock_issue]
    result.to_markdown = ValidationResult.to_markdown.__get__(result, ValidationResult)
    return result


class TestValidationResultToMarkdown:
    """Tests for ValidationResult.to_markdown method."""

    def test_passed_validation_no_issues(self, passed_result):
        """Test markdown output for passed validation with no issues."""
        markdown = passed_result.to_markdown()
        
        assert "## Validation Report" in markdown
        assert "Status: **passed**" in markdown
        assert "Rows checked: 100" in markdown
        assert "Issues found: 0" in markdown
        assert "Bad rows: 0" in markdown
        # Should not include table when no issues
        assert "| Column | Rule |" not in markdown

    def test_failed_validation_with_issues(self, failed_result_one_issue):
        """Test markdown output for failed validation with issues."""
        markdown = failed_result_one_issue.to_markdown()
        
        assert "## Validation Report" in markdown
        assert "Status: **failed**" in markdown
        assert "Issues found: 1" in markdown
        assert "Bad rows: 1" in markdown
        assert "| Column | Rule | Severity | Row | Value | Message |" in markdown
        assert "age" in markdown
        assert "range" in markdown
        assert "error" in markdown

    def test_redact_values_false_shows_values(self, failed_result_one_issue):
        """Test that values are shown when redact_values=False."""
        markdown = failed_result_one_issue.to_markdown(redact_values=False)
        
        assert "[REDACTED]" not in markdown
        # Value should be present (might be formatted)
        assert "1" in markdown or "-1" in markdown

    def test_redact_values_true_redacts_values(self, failed_result_one_issue):
        """Test that values are redacted when redact_values=True."""
        markdown = failed_result_one_issue.to_markdown(redact_values=True)
        
        assert "[REDACTED]" in markdown
        assert "-1" not in markdown

    def test_max_issues_none_shows_all_issues(self):
        """Test that max_issues=None shows all issues."""
        issues = []
        for i in range(5):
            issue = Mock()
            issue.column = f"column_{i}"
            issue.rule = "rule"
            issue.severity = "error"
            issue.row_index = i
            issue.value = i
            issue.message = f"message_{i}"
            issues.append(issue)
        
        result = Mock(spec=ValidationResult)
        result.passed = False
        result.row_count = 100
        result.issue_count = 5
        result.bad_rows = list(range(5))
        result.issues = issues
        result.to_markdown = ValidationResult.to_markdown.__get__(result, ValidationResult)
        
        markdown = result.to_markdown(max_issues=None)
        
        # All issues should be present
        for i in range(5):
            assert f"column_{i}" in markdown
        # No truncation message
        assert "_Showing" not in markdown

    def test_max_issues_zero_omits_table(self, failed_result_one_issue):
        """Test that max_issues=0 omits the issue table."""
        markdown = failed_result_one_issue.to_markdown(max_issues=0)
        
        assert "## Validation Report" in markdown
        assert "Issues found: 1" in markdown
        assert "_Issue table omitted by `max_issues=0`._" in markdown
        assert "| Column | Rule |" not in markdown

    def test_max_issues_limits_displayed_issues(self):
        """Test that max_issues limits the number of displayed issues."""
        issues = []
        for i in range(5):
            issue = Mock()
            issue.column = f"col{i}"
            issue.rule = "rule"
            issue.severity = "error"
            issue.row_index = i
            issue.value = i
            issue.message = f"msg{i}"
            issues.append(issue)
        
        result = Mock(spec=ValidationResult)
        result.passed = False
        result.row_count = 100
        result.issue_count = 5
        result.bad_rows = list(range(5))
        result.issues = issues
        result.to_markdown = ValidationResult.to_markdown.__get__(result, ValidationResult)
        
        markdown = result.to_markdown(max_issues=2)
        
        assert "col0" in markdown
        assert "col1" in markdown
        assert "col2" not in markdown
        assert "_Showing 2 of 5 issues._" in markdown

    def test_max_issues_invalid_type_string_raises_typeerror(self, failed_result_one_issue):
        """Test that string max_issues raises TypeError."""
        with pytest.raises(TypeError, match="max_issues must be an integer or None"):
            failed_result_one_issue.to_markdown(max_issues="5")

    def test_max_issues_invalid_type_float_raises_typeerror(self, failed_result_one_issue):
        """Test that float max_issues raises TypeError."""
        with pytest.raises(TypeError, match="max_issues must be an integer or None"):
            failed_result_one_issue.to_markdown(max_issues=5.0)

    def test_max_issues_invalid_type_bool_raises_typeerror(self, failed_result_one_issue):
        """Test that bool max_issues raises TypeError."""
        with pytest.raises(TypeError, match="max_issues must be an integer or None"):
            failed_result_one_issue.to_markdown(max_issues=True)

    def test_max_issues_negative_raises_valueerror(self, failed_result_one_issue):
        """Test that negative max_issues raises ValueError."""
        with pytest.raises(ValueError, match="max_issues must be non-negative"):
            failed_result_one_issue.to_markdown(max_issues=-1)

    def test_redact_values_invalid_type_raises_typeerror(self, failed_result_one_issue):
        """Test that non-bool redact_values raises TypeError."""
        with pytest.raises(TypeError, match="redact_values must be a bool"):
            failed_result_one_issue.to_markdown(redact_values="true")

    def test_markdown_table_structure(self, failed_result_one_issue):
        """Test that markdown table has correct structure with all columns."""
        markdown = failed_result_one_issue.to_markdown()
        
        # Should have all column headers
        assert "| Column | Rule | Severity | Row | Value | Message |" in markdown
        # Should have separator line
        assert "|---|---|---|---|---|---|" in markdown
        # Should have data row with pipe delimiters
        lines = markdown.split("\n")
        data_lines = [l for l in lines if l.startswith("| ") and "Column" not in l and "---|" not in l]
        assert len(data_lines) >= 1

    def test_multiple_issues_with_redaction(self):
        """Test multiple issues are all redacted when redact_values=True."""
        issues = []
        for i in range(3):
            issue = Mock()
            issue.column = f"sensitive_{i}"
            issue.rule = "rule"
            issue.severity = "error"
            issue.row_index = i
            issue.value = f"secret_{i}"
            issue.message = "Invalid"
            issues.append(issue)
        
        result = Mock(spec=ValidationResult)
        result.passed = False
        result.row_count = 100
        result.issue_count = 3
        result.bad_rows = list(range(3))
        result.issues = issues
        result.to_markdown = ValidationResult.to_markdown.__get__(result, ValidationResult)
        
        markdown = result.to_markdown(redact_values=True, max_issues=None)
        
        # Count redacted entries (should be at least 3, one per issue)
        redact_count = markdown.count("[REDACTED]")
        assert redact_count >= 3
        # Sensitive values should not appear
        assert "secret_0" not in markdown

    def test_edge_case_max_issues_equals_total_issues(self):
        """Test max_issues equal to total issue count shows no truncation message."""
        issues = []
        for i in range(3):
            issue = Mock()
            issue.column = f"col{i}"
            issue.rule = "rule"
            issue.severity = "error"
            issue.row_index = i
            issue.value = i
            issue.message = f"msg{i}"
            issues.append(issue)
        
        result = Mock(spec=ValidationResult)
        result.passed = False
        result.row_count = 100
        result.issue_count = 3
        result.bad_rows = list(range(3))
        result.issues = issues
        result.to_markdown = ValidationResult.to_markdown.__get__(result, ValidationResult)
        
        markdown = result.to_markdown(max_issues=3)
        
        # All issues shown, no truncation message
        assert "col0" in markdown
        assert "col2" in markdown
        assert "_Showing" not in markdown
