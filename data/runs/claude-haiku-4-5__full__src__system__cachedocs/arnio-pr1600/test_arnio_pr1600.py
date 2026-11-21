"""
End-to-end pytest tests for ValidationResult.to_markdown()

Tests cover:
- Real CSV file loading and validation workflows
- Parameter validation (max_issues, redact_values)
- Markdown output format and content
- Edge cases (zero issues, many issues, truncation)
- Integration with pandas accessor
"""

import pytest
import pandas as pd
import arnio as ar


class TestValidationResultToMarkdown:
    """End-to-end tests for ValidationResult.to_markdown()"""

    def test_passing_validation_shows_no_issues_table(self, tmp_path):
        """When validation passes, markdown shows status but no issue table."""
        csv_file = tmp_path / "valid_data.csv"
        csv_file.write_text("name,age,email\nAlice,30,alice@example.com\nBob,25,bob@example.com\n")
        
        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({
            "name": ar.String(nullable=False),
            "age": ar.Int64(nullable=False, min=0, max=150),
            "email": ar.Email(nullable=False),
        })
        
        result = ar.validate(frame, schema)
        markdown = result.to_markdown()
        
        assert result.passed is True
        assert "passed" in markdown.lower()
        assert "Issues found: 0" in markdown
        # No issue table when validation passes
        assert "| Column | Rule" not in markdown

    def test_failing_validation_shows_issue_table(self, tmp_path):
        """When validation fails, markdown includes detailed issue table with all columns."""
        csv_file = tmp_path / "invalid_data.csv"
        csv_file.write_text("email,age\ninvalid-email,999\n")
        
        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({
            "email": ar.Email(nullable=False),
            "age": ar.Int64(nullable=False, max=150),
        })
        
        result = ar.validate(frame, schema)
        markdown = result.to_markdown()
        
        assert result.passed is False
        assert "failed" in markdown.lower()
        assert "| Column | Rule | Severity | Row | Value | Message |" in markdown
        assert result.issue_count > 0

    def test_max_issues_limits_table_rows(self, tmp_path):
        """max_issues parameter limits number of issues displayed in table."""
        csv_file = tmp_path / "many_errors.csv"
        lines = ["age"]
        for _ in range(10):
            lines.append("-1")
        csv_file.write_text("\n".join(lines))
        
        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({"age": ar.Int64(min=0)})
        result = ar.validate(frame, schema)
        
        # Request only 3 issues
        markdown_limited = result.to_markdown(max_issues=3)
        
        # Count pipe-delimited table rows (excluding header rows)
        table_rows = [
            line for line in markdown_limited.split("\n") 
            if line.startswith("|") and "Column" not in line and "---|" not in line
        ]
        assert len(table_rows) == 3
        assert f"Showing 3 of {result.issue_count}" in markdown_limited

    def test_max_issues_zero_omits_issue_table(self, tmp_path):
        """max_issues=0 omits the issue table but keeps summary."""
        csv_file = tmp_path / "data_with_errors.csv"
        csv_file.write_text("age\n-5\n-10\n")
        
        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({"age": ar.Int64(min=0)})
        result = ar.validate(frame, schema)
        
        markdown = result.to_markdown(max_issues=0)
        
        assert "## Validation Report" in markdown
        assert "Issues found:" in markdown
        assert "Issue table omitted" in markdown
        # Should not have issue rows
        assert "|" not in markdown.split("omitted")[0].split("Issues found")[1]

    def test_redact_values_true_hides_sensitive_data(self, tmp_path):
        """redact_values=True replaces Value column with [REDACTED]."""
        csv_file = tmp_path / "sensitive_data.csv"
        csv_file.write_text("email,ssn\ninvalid-email,123-45-6789\n")
        
        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({
            "email": ar.Email(nullable=False),
        })
        result = ar.validate(frame, schema)
        
        markdown_redacted = result.to_markdown(redact_values=True)
        
        # All values should be redacted in the Value column
        if result.issue_count > 0:
            assert "[REDACTED]" in markdown_redacted
            # Actual values should be hidden
            assert "invalid-email" not in markdown_redacted

    def test_redact_values_false_shows_values(self, tmp_path):
        """redact_values=False (default) shows actual cell values in table."""
        csv_file = tmp_path / "data_values.csv"
        csv_file.write_text("age\n999\n")
        
        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({"age": ar.Int64(max=150)})
        result = ar.validate(frame, schema)
        
        markdown = result.to_markdown(redact_values=False)
        
        # If there are issues, actual value should be in the markdown
        if result.issue_count > 0:
            # 999 should appear somewhere (might be escaped)
            assert "999" in markdown

    def test_max_issues_non_integer_raises_type_error(self, tmp_path):
        """max_issues with non-integer value raises TypeError."""
        csv_file = tmp_path / "simple.csv"
        csv_file.write_text("age\n25\n")
        
        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({"age": ar.Int64()})
        result = ar.validate(frame, schema)
        
        with pytest.raises(TypeError, match="must be an integer"):
            result.to_markdown(max_issues="5")
        
        with pytest.raises(TypeError, match="must be an integer"):
            result.to_markdown(max_issues=5.5)

    def test_max_issues_negative_raises_value_error(self, tmp_path):
        """max_issues with negative value raises ValueError."""
        csv_file = tmp_path / "simple.csv"
        csv_file.write_text("age\n25\n")
        
        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({"age": ar.Int64()})
        result = ar.validate(frame, schema)
        
        with pytest.raises(ValueError, match="non-negative"):
            result.to_markdown(max_issues=-1)

    def test_redact_values_non_boolean_raises_type_error(self, tmp_path):
        """redact_values parameter must be boolean."""
        csv_file = tmp_path / "simple.csv"
        csv_file.write_text("age\n25\n")
        
        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({"age": ar.Int64()})
        result = ar.validate(frame, schema)
        
        with pytest.raises(TypeError, match="must be a bool"):
            result.to_markdown(redact_values="true")
        
        with pytest.raises(TypeError, match="must be a bool"):
            result.to_markdown(redact_values=1)

    def test_markdown_structure_includes_all_sections(self, tmp_path):
        """Markdown output includes header, status, counts, and row summary."""
        csv_file = tmp_path / "structured_data.csv"
        csv_file.write_text("name,age\nAlice,30\nBob,-5\n")
        
        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({
            "name": ar.String(nullable=False),
            "age": ar.Int64(min=0),
        })
        result = ar.validate(frame, schema)
        
        markdown = result.to_markdown()
        
        # Check for all required structural elements
        assert "## Validation Report" in markdown
        assert "Status:" in markdown
        assert "Rows checked:" in markdown
        assert "Issues found:" in markdown
        assert "Bad rows:" in markdown

    def test_markdown_with_multiple_validation_issues(self, tmp_path):
        """Markdown displays multiple issues with correct column names and rules."""
        csv_file = tmp_path / "multi_error.csv"
        csv_file.write_text("id,name,email,age\n1,,invalid-email,-5\n")
        
        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({
            "id": ar.Int64(nullable=False),
            "name": ar.String(nullable=False),
            "email": ar.Email(nullable=False),
            "age": ar.Int64(nullable=False, min=0),
        })
        result = ar.validate(frame, schema)
        
        markdown = result.to_markdown()
        
        # Verify table header and that multiple issues are shown
        assert "| Column | Rule | Severity | Row | Value | Message |" in markdown
        assert result.issue_count >= 3
        # Count actual issue rows (lines starting with |, not headers)
        issue_rows = [
            line for line in markdown.split("\n") 
            if line.startswith("|") and "Column" not in line and "---|" not in line
        ]
        assert len(issue_rows) >= 3

    def test_markdown_through_pandas_accessor(self, tmp_path):
        """ValidationResult.to_markdown works through pandas .arnio accessor."""
        csv_file = tmp_path / "pandas_data.csv"
        csv_file.write_text("name,age\nAlice,30\nBob,-5\n")
        
        df = pd.read_csv(str(csv_file))
        schema = ar.Schema({
            "name": ar.String(nullable=False),
            "age": ar.Int64(min=0),
        })
        
        result = df.arnio.validate(schema)
        markdown = result.to_markdown()
        
        assert isinstance(markdown, str)
        assert "Validation Report" in markdown
        assert "failed" in markdown.lower() or "passed" in markdown.lower()

    def test_markdown_reports_different_rule_types(self, tmp_path):
        """Markdown includes issues for different field types and validation rules."""
        csv_file = tmp_path / "different_rules.csv"
        csv_file.write_text("email,phone,country\nbad-email,123,USA\n")
        
        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({
            "email": ar.Email(nullable=False),
            "phone": ar.PhoneNumber(nullable=False),
            "country": ar.CountryCode(nullable=False),
        })
        result = ar.validate(frame, schema)
        
        markdown = result.to_markdown()
        
        # Check that the Rule column contains validation rule names
        assert "Rule" in markdown
        if result.issue_count > 0:
            # At least some rule types should appear
            rules_text = markdown.lower()
            assert any(keyword in rules_text for keyword in ["email", "phone", "country"])

    def test_markdown_truncation_message_format(self, tmp_path):
        """Markdown shows correct truncation message when issues exceed max_issues."""
        csv_file = tmp_path / "many_errors_data.csv"
        lines = ["age"]
        for _ in range(20):
            lines.append("-1")
        csv_file.write_text("\n".join(lines))
        
        frame = ar.read_csv(str(csv_file))
        schema = ar.Schema({"age": ar.Int64(min=0)})
        result = ar.validate(frame, schema)
        
        markdown = result.to_markdown(max_issues=5)
        
        # Should show truncation message with counts
        assert "Showing 5 of" in markdown
        assert str(result.issue_count) in markdown
        # Total issues should exceed the max shown
        assert result.issue_count > 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
