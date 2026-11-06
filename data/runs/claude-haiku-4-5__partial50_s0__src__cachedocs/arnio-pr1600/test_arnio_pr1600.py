import pytest
import pandas as pd
import arnio as ar


class TestValidationResultToMarkdown:
    """Tests for ValidationResult.to_markdown() method."""

    @pytest.fixture
    def passed_result(self):
        """Create a validation result that passes."""
        frame = ar.from_pandas(pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
        }))
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "name": ar.String(),
        })
        return ar.validate(frame, schema)

    @pytest.fixture
    def failed_result(self):
        """Create a validation result with multiple issues."""
        frame = ar.from_pandas(pd.DataFrame({
            "email": ["valid@test.com", "not_an_email", "good@test.com"],
            "age": [25, -5, 35],
        }))
        schema = ar.Schema({
            "email": ar.Email(nullable=False),
            "age": ar.Int64(min=0),
        })
        return ar.validate(frame, schema)

    def test_passed_validation_output(self, passed_result):
        """Test markdown output when validation passes."""
        markdown = passed_result.to_markdown()

        assert "## Validation Report" in markdown
        assert "Status: **passed**" in markdown
        assert "Issues found: 0" in markdown
        # Should not include table when no issues
        assert "| Column | Rule |" not in markdown

    def test_failed_validation_output(self, failed_result):
        """Test markdown output when validation fails."""
        markdown = failed_result.to_markdown()

        assert "Status: **failed**" in markdown
        assert "Issues found:" in markdown

    def test_failed_validation_includes_table(self, failed_result):
        """Test that failed validation includes issue table."""
        markdown = failed_result.to_markdown()

        assert "| Column | Rule | Severity | Row | Value | Message |" in markdown
        assert "|---|---|---|---|---|---|" in markdown

    def test_max_issues_parameter_limits_output(self, failed_result):
        """Test max_issues parameter limits number of issues shown."""
        if failed_result.issue_count > 1:
            limited = failed_result.to_markdown(max_issues=1)
            unlimited = failed_result.to_markdown(max_issues=None)

            # Limited version should have fewer issue rows
            limited_rows = limited.count("\n| ")
            unlimited_rows = unlimited.count("\n| ")
            assert limited_rows < unlimited_rows

    def test_max_issues_zero_omits_table(self, failed_result):
        """Test max_issues=0 omits the issue table."""
        markdown = failed_result.to_markdown(max_issues=0)

        assert "Issue table omitted by `max_issues=0`" in markdown
        assert "| Column | Rule |" not in markdown

    def test_redact_values_true_hides_data(self, failed_result):
        """Test redact_values=True replaces values with [REDACTED]."""
        markdown = failed_result.to_markdown(redact_values=True)

        assert "[REDACTED]" in markdown

    def test_redact_values_false_shows_data(self, failed_result):
        """Test redact_values=False shows actual values."""
        markdown = failed_result.to_markdown(redact_values=False)

        # Should contain actual values, not redacted
        assert "[REDACTED]" not in markdown

    def test_max_issues_rejects_bool(self, passed_result):
        """Test max_issues rejects bool values."""
        with pytest.raises(TypeError, match="max_issues must be an integer or None"):
            passed_result.to_markdown(max_issues=True)

    def test_max_issues_rejects_string(self, passed_result):
        """Test max_issues rejects string values."""
        with pytest.raises(TypeError, match="max_issues must be an integer or None"):
            passed_result.to_markdown(max_issues="10")

    def test_max_issues_rejects_negative(self, passed_result):
        """Test max_issues rejects negative integers."""
        with pytest.raises(ValueError, match="max_issues must be non-negative"):
            passed_result.to_markdown(max_issues=-5)

    def test_redact_values_rejects_non_bool(self, passed_result):
        """Test redact_values rejects non-bool values."""
        with pytest.raises(TypeError, match="redact_values must be a bool"):
            passed_result.to_markdown(redact_values="yes")

    def test_output_structure_includes_metadata(self, failed_result):
        """Test output includes rows checked and bad rows counts."""
        markdown = failed_result.to_markdown()

        assert "Rows checked:" in markdown
        assert "Bad rows:" in markdown
        assert "Issues found:" in markdown

    def test_hidden_issues_message_when_truncated(self, failed_result):
        """Test message indicating hidden issues when max_issues truncates."""
        if failed_result.issue_count > 1:
            markdown = failed_result.to_markdown(max_issues=1)

            assert "Showing 1 of" in markdown

    def test_combined_parameters_max_issues_and_redact(self, failed_result):
        """Test using max_issues and redact_values together."""
        markdown = failed_result.to_markdown(max_issues=1, redact_values=True)

        assert "[REDACTED]" in markdown
        if failed_result.issue_count > 1:
            assert "Showing 1 of" in markdown
