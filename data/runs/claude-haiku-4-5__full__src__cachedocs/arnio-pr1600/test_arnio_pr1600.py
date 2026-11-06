import pytest
import pandas as pd
import arnio as ar


class TestValidationResultToMarkdown:
    """Tests for ValidationResult.to_markdown method."""

    @pytest.fixture
    def passing_result(self):
        """Create a ValidationResult that passes."""
        frame = ar.from_pandas(pd.DataFrame({
            "id": [1, 2, 3],
            "email": ["alice@example.com", "bob@example.com", "charlie@example.com"]
        }))
        schema = ar.Schema({
            "id": ar.Int64(),
            "email": ar.Email()
        })
        return ar.validate(frame, schema)

    @pytest.fixture
    def failing_result(self):
        """Create a ValidationResult with issues."""
        frame = ar.from_pandas(pd.DataFrame({
            "id": [1, 2, 3],
            "email": ["alice@example.com", "invalid-email", "charlie@example.com"]
        }))
        schema = ar.Schema({
            "id": ar.Int64(),
            "email": ar.Email()
        })
        return ar.validate(frame, schema)

    # Basic structure tests
    def test_passed_validation_contains_status_passed(self, passing_result):
        """Test that passing validation shows 'passed' status."""
        markdown = passing_result.to_markdown()
        assert "Status: **passed**" in markdown

    def test_failed_validation_contains_status_failed(self, failing_result):
        """Test that failing validation shows 'failed' status."""
        markdown = failing_result.to_markdown()
        assert "Status: **failed**" in markdown

    def test_markdown_contains_header(self, passing_result):
        """Test that markdown includes validation report header."""
        markdown = passing_result.to_markdown()
        assert "## Validation Report" in markdown

    def test_markdown_contains_metadata(self, passing_result):
        """Test that markdown includes metadata about validation."""
        markdown = passing_result.to_markdown()
        assert "Rows checked:" in markdown
        assert "Issues found:" in markdown
        assert "Bad rows:" in markdown

    def test_passed_validation_no_issue_table(self, passing_result):
        """Test that passing validation doesn't include issue table."""
        markdown = passing_result.to_markdown()
        assert "| Column | Rule | Severity | Row | Value | Message |" not in markdown

    def test_failed_validation_includes_issue_table(self, failing_result):
        """Test that failing validation includes issue table with proper structure."""
        markdown = failing_result.to_markdown()
        assert "| Column | Rule | Severity | Row | Value | Message |" in markdown

    # max_issues parameter tests
    def test_max_issues_zero_omits_table(self, failing_result):
        """Test that max_issues=0 omits the issue table."""
        markdown = failing_result.to_markdown(max_issues=0)
        assert "_Issue table omitted by `max_issues=0`._" in markdown
        assert "| Column | Rule" not in markdown

    def test_max_issues_none_shows_all(self, failing_result):
        """Test that max_issues=None shows all issues (default behavior)."""
        markdown_none = failing_result.to_markdown(max_issues=None)
        markdown_default = failing_result.to_markdown()
        assert markdown_none == markdown_default

    def test_max_issues_truncates_issues_list(self, failing_result):
        """Test that max_issues limits issues displayed."""
        if failing_result.issue_count > 1:
            markdown = failing_result.to_markdown(max_issues=1)
            assert "Showing 1 of" in markdown

    def test_max_issues_negative_raises_value_error(self, passing_result):
        """Test that negative max_issues raises ValueError."""
        with pytest.raises(ValueError, match="max_issues must be non-negative"):
            passing_result.to_markdown(max_issues=-1)

    def test_max_issues_non_integer_raises_type_error(self, passing_result):
        """Test that non-integer max_issues raises TypeError."""
        with pytest.raises(TypeError, match="max_issues must be an integer or None"):
            passing_result.to_markdown(max_issues="10")

    def test_max_issues_bool_raises_type_error(self, passing_result):
        """Test that boolean max_issues raises TypeError (bool is not int)."""
        with pytest.raises(TypeError, match="max_issues must be an integer or None"):
            passing_result.to_markdown(max_issues=True)

    # redact_values parameter tests
    def test_redact_values_true_masks_values(self, failing_result):
        """Test that redact_values=True replaces values with [REDACTED]."""
        markdown = failing_result.to_markdown(redact_values=True)
        assert "[REDACTED]" in markdown

    def test_redact_values_false_shows_values(self, failing_result):
        """Test that redact_values=False shows original values in table."""
        markdown = failing_result.to_markdown(redact_values=False)
        # With redact_values=False, we should see actual values (not [REDACTED])
        assert markdown.count("[REDACTED]") == 0

    def test_redact_values_non_bool_raises_type_error(self, passing_result):
        """Test that non-bool redact_values raises TypeError."""
        with pytest.raises(TypeError, match="redact_values must be a bool"):
            passing_result.to_markdown(redact_values="true")

    def test_redact_and_max_issues_together(self, failing_result):
        """Test that redact_values and max_issues work together."""
        markdown = failing_result.to_markdown(redact_values=True, max_issues=0)
        assert "_Issue table omitted by `max_issues=0`._" in markdown
        # Table should not be present even with redaction
        assert "| Column | Rule" not in markdown
