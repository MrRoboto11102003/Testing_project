import pytest
import pandas as pd
import numpy as np
import arnio as ar
import json


class TestProfileBasic:
    """Basic profiling functionality."""
    
    def test_profile_simple_csv(self, tmp_path):
        """Profile a simple CSV and verify report structure."""
        csv_file = tmp_path / "simple.csv"
        csv_file.write_text("id,name,score\n1,Alice,95\n2,Bob,87\n3,Charlie,92\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        
        assert report.row_count == 3
        assert report.column_count == 3
        assert report.duplicate_rows == 0
        assert 0 <= report.quality_score <= 100
        assert "id" in report.columns
        assert "name" in report.columns
        assert "score" in report.columns
    
    def test_profile_with_nulls_and_duplicates(self, tmp_path):
        """Profile a CSV with null values and duplicate rows."""
        csv_file = tmp_path / "messy.csv"
        csv_file.write_text("id,value\n1,10\n2,\n2,\n3,30\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        
        assert report.row_count == 4
        assert report.columns["value"].null_count == 2
        assert 0.49 < report.columns["value"].null_ratio < 0.51
        assert report.duplicate_rows == 1
        assert 0.24 < report.duplicate_ratio < 0.26
    
    def test_profile_exclude_columns(self, tmp_path):
        """Exclude specific columns from profiling."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name,debug_col\n1,Alice,x\n2,Bob,y\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame, exclude_columns=["debug_col"])
        
        assert report.column_count == 2
        assert "debug_col" not in report.columns
        assert "id" in report.columns
        assert "name" in report.columns
    
    def test_profile_exclude_nonexistent_column_raises(self, tmp_path):
        """Excluding a nonexistent column raises ValueError."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name\n1,Alice\n")
        
        frame = ar.read_csv(str(csv_file))
        with pytest.raises(ValueError, match="not found"):
            ar.profile(frame, exclude_columns=["missing"])


class TestProfileParameters:
    """Test profile function parameter validation."""
    
    def test_profile_custom_sample_size(self, tmp_path):
        """Profile with custom sample_size parameter."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("text\nval1\nval2\nval3\nval4\nval5\nval6\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame, sample_size=2)
        
        # sample_size limits collected sample values
        col_profile = report.columns["text"]
        assert len(col_profile.sample_values) <= 2
    
    def test_profile_sample_size_negative_raises(self, tmp_path):
        """Negative sample_size raises ValueError."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id\n1\n")
        
        frame = ar.read_csv(str(csv_file))
        with pytest.raises(ValueError, match="non-negative"):
            ar.profile(frame, sample_size=-1)
    
    def test_profile_sample_size_not_integer_raises(self, tmp_path):
        """Non-integer sample_size raises TypeError."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id\n1\n")
        
        frame = ar.read_csv(str(csv_file))
        with pytest.raises(TypeError, match="sample_size must be an integer"):
            ar.profile(frame, sample_size=3.14)
    
    def test_profile_approx_top_values_parameters(self, tmp_path):
        """Test approximate top values parameters validation."""
        csv_file = tmp_path / "highcard.csv"
        lines = ["id\n"] + [f"val_{i}\n" for i in range(2500)]
        csv_file.write_text("".join(lines))
        
        frame = ar.read_csv(str(csv_file))
        
        # Valid parameters should not raise
        report = ar.profile(
            frame,
            approx_top_values=True,
            approx_top_values_min_unique=1000,
            approx_top_values_min_ratio=0.2,
            approx_top_values_sample_size=1000,
        )
        
        assert report.row_count == 2500


class TestProfileReportOutput:
    """Test DataQualityReport output methods and serialization."""
    
    def test_report_summary_dict(self, tmp_path):
        """Report.summary() returns a dict with key metrics."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,value\n1,10\n2,20\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        summary = report.summary()
        
        assert isinstance(summary, dict)
        assert summary["row_count"] == 2
        assert summary["column_count"] == 2
        assert 0 <= summary["quality_score"] <= 100
    
    def test_report_to_dict_json_serializable(self, tmp_path):
        """Report.to_dict() produces JSON-serializable output."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,text\n1,hello\n2,world\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        data = report.to_dict()
        
        # Should be JSON serializable
        json_str = json.dumps(data)
        reloaded = json.loads(json_str)
        
        assert reloaded["row_count"] == 2
        assert reloaded["column_count"] == 2
        assert "columns" in reloaded
    
    def test_report_to_dict_with_redaction(self, tmp_path):
        """Report.to_dict() with redact_sample_values=True redacts values."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("email\ntest@example.com\nalice@example.com\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame, sample_size=2)
        data_redacted = report.to_dict(redact_sample_values=True)
        
        # Sample values should be redacted strings
        email_col = data_redacted["columns"]["email"]
        if "sample_values" in email_col and email_col["sample_values"]:
            for val in email_col["sample_values"]:
                assert val == "[REDACTED]" or val is None
    
    def test_report_to_markdown(self, tmp_path):
        """Report.to_markdown() produces markdown text output."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name\n1,Alice\n2,Bob\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        markdown = report.to_markdown()
        
        assert isinstance(markdown, str)
        assert len(markdown) > 0
    
    def test_report_to_html(self, tmp_path):
        """Report.to_html() produces valid HTML output."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,value\n1,100\n2,200\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        html = report.to_html()
        
        assert isinstance(html, str)
        assert "<" in html and ">" in html
    
    def test_report_to_html_with_file_output(self, tmp_path):
        """Report.to_html() can write to a file."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,value\n1,100\n2,200\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        
        html_file = tmp_path / "report.html"
        report.to_html(file_path=str(html_file))
        
        assert html_file.exists()
        content = html_file.read_text()
        assert "<" in content and ">" in content


class TestProfileEdgeCases:
    """Test edge cases and special data scenarios."""
    
    def test_profile_empty_frame(self, tmp_path):
        """Profile an empty frame (header only)."""
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("id,name\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        
        assert report.row_count == 0
        assert report.column_count == 2
        assert report.duplicate_rows == 0
    
    def test_profile_all_null_column(self, tmp_path):
        """Profile a column where all values are null."""
        csv_file = tmp_path / "all_nulls.csv"
        csv_file.write_text("id,empty\n1,\n2,\n3,\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        
        assert report.columns["empty"].null_count == 3
        assert report.columns["empty"].null_ratio == 1.0


class TestProfileIntegration:
    """Integration tests combining profiling with other arnio workflows."""
    
    def test_profile_then_clean_via_suggestions(self, tmp_path):
        """Profile data, review suggestions, and apply cleaning."""
        csv_file = tmp_path / "messy.csv"
        csv_file.write_text("id, name , score\n1, Alice ,95\n2, Bob ,87\n2, Bob ,87\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        
        # Report should have cleaning suggestions
        assert isinstance(report.suggestions, list)
        
        # Apply suggestions to clean the data
        if report.suggestions:
            clean = ar.pipeline(frame, report.suggestions)
            clean_report = ar.profile(clean)
            
            # Quality should improve or stay same after cleaning
            assert clean_report.quality_score >= report.quality_score * 0.9
    
    def test_profile_via_pandas_accessor(self, tmp_path):
        """Profile data using pandas .arnio accessor."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,value\n1,10\n2,20\n3,30\n")
        
        df = pd.read_csv(str(csv_file))
        report = df.arnio.profile()
        
        assert report.row_count == 3
        assert report.column_count == 2
        assert isinstance(report, ar.DataQualityReport)
    
    def test_profile_roundtrip_workflow(self, tmp_path):
        """Test profiling in a complete data pipeline."""
        input_csv = tmp_path / "input.csv"
        input_csv.write_text("id,name,revenue\n1,Alice,1000\n2,Bob,\n3,Charlie,3000\n")
        
        # Load, profile, clean, convert to pandas
        frame = ar.read_csv(str(input_csv))
        report1 = ar.profile(frame)
        
        # Apply cleaning from suggestions
        if report1.suggestions:
            clean_frame = ar.pipeline(frame, report1.suggestions)
        else:
            clean_frame = frame
        
        # Profile again and convert to pandas
        report2 = ar.profile(clean_frame)
        df_result = ar.to_pandas(clean_frame)
        
        # Verify results
        assert len(df_result) == report2.row_count
        assert len(df_result.columns) == report2.column_count
        assert report2.quality_score >= report1.quality_score * 0.9
