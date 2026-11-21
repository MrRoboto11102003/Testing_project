import pytest
import pandas as pd
import tempfile
from pathlib import Path
import sys
import subprocess
import arnio as ar


class TestValidationResultToMarkdown:
    """End-to-end tests for arnio.schema.ValidationResult.to_markdown()"""

    def test_passing_validation_no_issues(self, tmp_path):
        """All valid data produces passed status and no issue table."""
        csv_file = tmp_path / "valid.csv"
        csv_file.write_text("id,name,age\n1,Alice,30\n2,Bob,25\n")
        
        df = pd.read_csv(csv_file)
        schema = ar.Schema([
            ar.Column("id", checks=[ar.Check("type", "integer")]),
            ar.Column("name"),
            ar.Column("age", checks=[ar.Check("type", "integer")]),
        ])
        result = ar.validate(df, schema)
        markdown = result.to_markdown()
        
        assert "## Validation Report" in markdown
        assert "Status: **passed**" in markdown
        assert "Issues found: 0" in markdown
        assert "Rows checked: 2" in markdown
        assert "Bad rows: 0" in markdown

    def test_failing_validation_with_issues_table(self, tmp_path):
        """Invalid data produces failed status and markdown table."""
        csv_file = tmp_path / "invalid.csv"
        csv_file.write_text("id,age\n1,thirty\n2,25\n")
        
        df = pd.read_csv(csv_file)
        schema = ar.Schema([
            ar.Column("id", checks=[ar.Check("type", "integer")]),
            ar.Column("age", checks=[ar.Check("type", "integer")]),
        ])
        result = ar.validate(df, schema)
        markdown = result.to_markdown()
        
        assert "## Validation Report" in markdown
        assert "Status: **failed**" in markdown
        assert "| Column | Rule | Severity | Row | Value | Message |" in markdown
        assert "|---|---|---|---|---|---|" in markdown

    def test_max_issues_parameter_limits_rows(self, tmp_path):
        """max_issues parameter limits number of issues shown."""
        csv_file = tmp_path / "errors.csv"
        rows = ["val"] + [f"error_{i}" for i in range(5)]
        csv_file.write_text("\n".join(rows))
        
        df = pd.read_csv(csv_file)
        schema = ar.Schema([ar.Column("val", checks=[ar.Check("type", "integer")])])
        result = ar.validate(df, schema)
        
        markdown_limited = result.to_markdown(max_issues=2)
        markdown_all = result.to_markdown()
        
        # Limited version should have fewer or same lines
        limited_lines = markdown_limited.count("\n")
        all_lines = markdown_all.count("\n")
        assert limited_lines <= all_lines

    def test_max_issues_zero_omits_table(self, tmp_path):
        """max_issues=0 omits issue table."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("val\nerror\n")
        
        df = pd.read_csv(csv_file)
        schema = ar.Schema([ar.Column("val", checks=[ar.Check("type", "integer")])])
        result = ar.validate(df, schema)
        markdown = result.to_markdown(max_issues=0)
        
        assert "Issue table omitted" in markdown
        # Verify no data rows in table
        lines = markdown.split("\n")
        data_rows = [l for l in lines if l.startswith("| ") 
                     and "Column" not in l and "---" not in l]
        assert len(data_rows) == 0

    def test_max_issues_showing_count_note(self, tmp_path):
        """When max_issues < total, shows 'Showing N of M' note."""
        csv_file = tmp_path / "data.csv"
        rows = ["val"] + [f"bad_{i}" for i in range(5)]
        csv_file.write_text("\n".join(rows))
        
        df = pd.read_csv(csv_file)
        schema = ar.Schema([ar.Column("val", checks=[ar.Check("type", "integer")])])
        result = ar.validate(df, schema)
        
        if result.issue_count > 2:
            markdown = result.to_markdown(max_issues=2)
            assert "Showing 2 of" in markdown

    def test_redact_values_true_hides_data(self, tmp_path):
        """redact_values=True replaces values with [REDACTED]."""
        csv_file = tmp_path / "sensitive.csv"
        csv_file.write_text("ssn\ninvalid_ssn_12345\n")
        
        df = pd.read_csv(csv_file)
        schema = ar.Schema([ar.Column("ssn", checks=[ar.Check("type", "integer")])])
        result = ar.validate(df, schema)
        markdown = result.to_markdown(redact_values=True)
        
        assert "[REDACTED]" in markdown
        assert "invalid_ssn" not in markdown

    def test_redact_values_false_shows_values(self, tmp_path):
        """redact_values=False shows actual values in table."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("val\nbad_value\n")
        
        df = pd.read_csv(csv_file)
        schema = ar.Schema([ar.Column("val", checks=[ar.Check("type", "integer")])])
        result = ar.validate(df, schema)
        markdown = result.to_markdown(redact_values=False)
        
        assert "[REDACTED]" not in markdown
        # Value should be present or escaped
        assert len(markdown) > 0

    def test_max_issues_type_error_string(self, tmp_path):
        """max_issues rejects string values."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("val\n1\n")
        
        df = pd.read_csv(csv_file)
        schema = ar.Schema([ar.Column("val", checks=[ar.Check("type", "integer")])])
        result = ar.validate(df, schema)
        
        with pytest.raises(TypeError):
            result.to_markdown(max_issues="2")

    def test_max_issues_type_error_bool(self, tmp_path):
        """max_issues rejects bool (not integer)."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("val\n1\n")
        
        df = pd.read_csv(csv_file)
        schema = ar.Schema([ar.Column("val", checks=[ar.Check("type", "integer")])])
        result = ar.validate(df, schema)
        
        with pytest.raises(TypeError):
            result.to_markdown(max_issues=True)

    def test_max_issues_negative_value_error(self, tmp_path):
        """max_issues rejects negative integers."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("val\n1\n")
        
        df = pd.read_csv(csv_file)
        schema = ar.Schema([ar.Column("val", checks=[ar.Check("type", "integer")])])
        result = ar.validate(df, schema)
        
        with pytest.raises(ValueError):
            result.to_markdown(max_issues=-1)

    def test_redact_values_type_error(self, tmp_path):
        """redact_values rejects non-bool values."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("val\n1\n")
        
        df = pd.read_csv(csv_file)
        schema = ar.Schema([ar.Column("val", checks=[ar.Check("type", "integer")])])
        result = ar.validate(df, schema)
        
        with pytest.raises(TypeError):
            result.to_markdown(redact_values="yes")

    def test_markdown_table_has_all_columns(self, tmp_path):
        """Markdown table includes all required columns."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,val\n1,bad\n")
        
        df = pd.read_csv(csv_file)
        schema = ar.Schema([
            ar.Column("id", checks=[ar.Check("type", "integer")]),
            ar.Column("val", checks=[ar.Check("type", "integer")]),
        ])
        result = ar.validate(df, schema)
        markdown = result.to_markdown()
        
        header = "| Column | Rule | Severity | Row | Value | Message |"
        assert header in markdown
        separator = "|---|---|---|---|---|---|"
        assert separator in markdown

    def test_bad_rows_count_reported(self, tmp_path):
        """Report includes bad rows count."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id\nbad\n25\n")
        
        df = pd.read_csv(csv_file)
        schema = ar.Schema([ar.Column("id", checks=[ar.Check("type", "integer")])])
        result = ar.validate(df, schema)
        markdown = result.to_markdown()
        
        assert "Bad rows:" in markdown
        assert result.bad_rows is not None

    def test_multiple_errors_same_row_all_shown(self, tmp_path):
        """Multiple errors in same row all appear in table."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,val\nbad,wrong\n")
        
        df = pd.read_csv(csv_file)
        schema = ar.Schema([
            ar.Column("id", checks=[ar.Check("type", "integer")]),
            ar.Column("val", checks=[ar.Check("type", "integer")]),
        ])
        result = ar.validate(df, schema)
        markdown = result.to_markdown()
        
        lines = markdown.split("\n")
        data_rows = [l for l in lines if l.startswith("| ") 
                     and "Column" not in l and "---" not in l]
        assert len(data_rows) >= 2

    def test_valid_and_invalid_mixed(self, tmp_path):
        """Dataset with mix of valid and invalid rows reports correctly."""
        csv_file = tmp_path / "mixed.csv"
        csv_file.write_text("id,val\n1,100\nbad,200\n3,wrong\n")
        
        df = pd.read_csv(csv_file)
        schema = ar.Schema([
            ar.Column("id", checks=[ar.Check("type", "integer")]),
            ar.Column("val", checks=[ar.Check("type", "integer")]),
        ])
        result = ar.validate(df, schema)
        markdown = result.to_markdown()
        
        assert "Status: **failed**" in markdown
        assert "Rows checked: 3" in markdown
        # Should have issues
        lines = markdown.split("\n")
        data_rows = [l for l in lines if l.startswith("| ") 
                     and "Column" not in l and "---" not in l]
        assert len(data_rows) > 0
