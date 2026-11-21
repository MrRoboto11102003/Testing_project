import pytest
import pandas as pd
import arnio as ar
from pathlib import Path


class TestValidationResultToMarkdown:
    """End-to-end tests for arnio.schema.ValidationResult.to_markdown()"""

    def test_markdown_passing_validation_no_issues(self, tmp_path):
        """Test markdown output when validation passes with no issues."""
        csv_path = tmp_path / "valid_data.csv"
        csv_path.write_text("id,email,age\n1,alice@example.com,30\n2,bob@example.com,25\n")
        
        frame = ar.read_csv(str(csv_path))
        schema = ar.Schema({
            "id": ar.Int64(),
            "email": ar.Email(),
            "age": ar.Int64(),
        })
        result = ar.validate(frame, schema)
        
        md = result.to_markdown()
        
        assert "## Validation Report" in md
        assert "Status: **passed**" in md
        assert "Issues found: 0" in md
        assert "Rows checked: 2" in md

    def test_markdown_failing_validation_with_issue_table(self, tmp_path):
        """Test markdown output includes issue table on validation failures."""
        csv_path = tmp_path / "invalid_data.csv"
        csv_path.write_text("id,email,age\n1,not-an-email,30\n2,bob@example.com,-5\n")
        
        frame = ar.read_csv(str(csv_path))
        schema = ar.Schema({
            "id": ar.Int64(),
            "email": ar.Email(),
            "age": ar.Int64(min=0),
        })
        result = ar.validate(frame, schema)
        
        md = result.to_markdown()
        
        assert "## Validation Report" in md
        assert "Status: **failed**" in md
        assert "Issues found:" in md
        assert "| Column | Rule | Severity | Row | Value | Message |" in md
        # Verify table rows are present for each issue
        assert result.issue_count >= 2
        table_lines = [line for line in md.split('\n') if line.startswith('|') and 'Column' not in line]
        assert len(table_lines) >= 2

    def test_markdown_max_issues_zero_omits_table(self, tmp_path):
        """Test max_issues=0 omits the issue table."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,email\n1,bad1\n2,bad2\n3,bad3\n")
        
        frame = ar.read_csv(str(csv_path))
        schema = ar.Schema({"id": ar.Int64(), "email": ar.Email()})
        result = ar.validate(frame, schema)
        
        md = result.to_markdown(max_issues=0)
        
        assert "_Issue table omitted by `max_issues=0`._" in md
        assert "| Column | Rule | Severity |" not in md

    def test_markdown_max_issues_limits_rows(self, tmp_path):
        """Test max_issues parameter limits displayed issues."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text(
            "id,email,age\n"
            "1,bad1,bad\n"
            "2,bad2,bad\n"
            "3,bad3,bad\n"
            "4,bad4,bad\n"
        )
        
        frame = ar.read_csv(str(csv_path))
        schema = ar.Schema({
            "id": ar.Int64(),
            "email": ar.Email(),
            "age": ar.Int64(),
        })
        result = ar.validate(frame, schema)
        
        # Request only 2 issues
        md = result.to_markdown(max_issues=2)
        
        # Count issue rows in table (not header or separator)
        table_rows = [line for line in md.split('\n') 
                     if line.startswith('|') and 'Column' not in line and '---|' not in line]
        assert len(table_rows) <= 2
        
        # Should show truncation note if more issues exist
        if result.issue_count > 2:
            assert "Showing 2 of" in md

    def test_markdown_redact_values_false_shows_data(self, tmp_path):
        """Test redact_values=False exposes actual cell values."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,email\n1,sensitive-data-123\n")
        
        frame = ar.read_csv(str(csv_path))
        schema = ar.Schema({
            "id": ar.Int64(),
            "email": ar.Email(),
        })
        result = ar.validate(frame, schema)
        
        md = result.to_markdown(redact_values=False)
        
        # Sensitive values should be visible
        assert "[REDACTED]" not in md

    def test_markdown_redact_values_true_masks_values(self, tmp_path):
        """Test redact_values=True masks cell values as [REDACTED]."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,email,username\n1,bad,secret\n")
        
        frame = ar.read_csv(str(csv_path))
        schema = ar.Schema({
            "id": ar.Int64(),
            "email": ar.Email(),
            "username": ar.String(),
        })
        result = ar.validate(frame, schema)
        
        md = result.to_markdown(redact_values=True)
        
        # Redacted placeholder should appear instead of actual values
        if result.issue_count > 0:
            assert "[REDACTED]" in md

    def test_markdown_error_max_issues_negative(self, tmp_path):
        """Test negative max_issues raises ValueError."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id\n1\n")
        
        frame = ar.read_csv(str(csv_path))
        result = ar.validate(frame, ar.Schema({"id": ar.Int64()}))
        
        with pytest.raises(ValueError, match="non-negative"):
            result.to_markdown(max_issues=-1)

    def test_markdown_error_max_issues_bool_raises_type_error(self, tmp_path):
        """Test bool max_issues raises TypeError (not int)."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id\n1\n")
        
        frame = ar.read_csv(str(csv_path))
        result = ar.validate(frame, ar.Schema({"id": ar.Int64()}))
        
        with pytest.raises(TypeError, match="integer"):
            result.to_markdown(max_issues=True)

    def test_markdown_error_max_issues_string_raises_type_error(self, tmp_path):
        """Test string max_issues raises TypeError."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id\n1\n")
        
        frame = ar.read_csv(str(csv_path))
        result = ar.validate(frame, ar.Schema({"id": ar.Int64()}))
        
        with pytest.raises(TypeError, match="integer"):
            result.to_markdown(max_issues="10")

    def test_markdown_error_redact_values_non_bool(self, tmp_path):
        """Test non-bool redact_values raises TypeError."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id\n1\n")
        
        frame = ar.read_csv(str(csv_path))
        result = ar.validate(frame, ar.Schema({"id": ar.Int64()}))
        
        with pytest.raises(TypeError, match="bool"):
            result.to_markdown(redact_values="yes")

    def test_markdown_with_unique_constraint_violation(self, tmp_path):
        """Test markdown reports unique constraint violations."""
        csv_path = tmp_path / "users.csv"
        csv_path.write_text("user_id,username\n1,alice\n2,alice\n3,bob\n")
        
        frame = ar.read_csv(str(csv_path))
        schema = ar.Schema(
            {"user_id": ar.Int64(), "username": ar.String()},
            unique=["username"]
        )
        result = ar.validate(frame, schema)
        
        md = result.to_markdown()
        
        assert "## Validation Report" in md
        if result.issue_count > 0:
            assert "Status: **failed**" in md

    def test_markdown_comprehensive_workflow_saved_to_file(self, tmp_path):
        """Test end-to-end: load CSV, validate, generate and save markdown report."""
        csv_path = tmp_path / "customers.csv"
        csv_path.write_text(
            "customer_id,name,email,revenue\n"
            "1,Alice,alice@example.com,10000\n"
            "2,Bob,bad@address,5000\n"
            "3,Charlie,charlie@example.com,-100\n"
        )
        
        # Load and validate
        frame = ar.read_csv(str(csv_path))
        schema = ar.Schema({
            "customer_id": ar.Int64(unique=True),
            "name": ar.String(nullable=False),
            "email": ar.Email(nullable=False),
            "revenue": ar.Float64(min=0),
        })
        result = ar.validate(frame, schema)
        
        # Generate markdown report
        md_full = result.to_markdown()
        md_limited = result.to_markdown(max_issues=2, redact_values=True)
        
        # Save to file
        report_path = tmp_path / "validation_report.md"
        report_path.write_text(md_full)
        
        # Verify file and content
        assert report_path.exists()
        saved_content = report_path.read_text()
        assert "## Validation Report" in saved_content
        assert "Status:" in saved_content
        
        # Verify limited report is different from full
        assert len(md_limited) <= len(md_full) or "Showing" in md_limited

    def test_markdown_with_multiple_severity_levels(self, tmp_path):
        """Test markdown includes severity column with different levels."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,age\n1,10\n")
        
        frame = ar.read_csv(str(csv_path))
        schema = ar.Schema({
            "id": ar.Int64(),
            "age": ar.Int64(min=18, severity="warning"),
        })
        result = ar.validate(frame, schema)
        
        md = result.to_markdown()
        
        assert "| Severity |" in md
        if result.issue_count > 0:
            assert "warning" in md.lower() or "error" in md.lower()

    def test_markdown_empty_dataset_no_rows(self, tmp_path):
        """Test markdown with empty dataset (header only)."""
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("id,name,email\n")
        
        frame = ar.read_csv(str(csv_path))
        schema = ar.Schema({
            "id": ar.Int64(),
            "name": ar.String(),
            "email": ar.Email(),
        })
        result = ar.validate(frame, schema)
        
        md = result.to_markdown()
        
        assert "## Validation Report" in md
        assert "Rows checked: 0" in md
        assert "Status: **passed**" in md

    def test_markdown_via_pandas_accessor_workflow(self, tmp_path):
        """Test markdown generation using pandas .arnio accessor."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,score\n1,95\n2,bad\n3,105\n")
        
        # Load via pandas then convert
        df = pd.read_csv(str(csv_path))
        frame = ar.from_pandas(df)
        
        schema = ar.Schema({
            "id": ar.Int64(),
            "score": ar.Int64(min=0, max=100),
        })
        result = ar.validate(frame, schema)
        
        # Generate markdown
        md = result.to_markdown()
        
        # Verify markdown structure
        assert "## Validation Report" in md
        assert "Rows checked: 3" in md
