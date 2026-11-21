import pytest
import pandas as pd
import arnio as ar
import json
import io
from pathlib import Path


class TestColumnProfileToDict:
    """Tests for ColumnProfile.to_dict()."""
    
    def test_column_profile_to_dict_basic(self, tmp_path):
        """Profile CSV and verify column profile to_dict contains expected fields."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name,age,salary\nAlice,30,50000\nBob,25,45000\nCharlie,35,60000\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        col_dict = report.columns["age"].to_dict()
        
        assert col_dict["name"] == "age"
        assert col_dict["dtype"] == "int64"
        assert col_dict["row_count"] == 3
        assert col_dict["null_count"] == 0
        assert col_dict["unique_count"] == 3
        assert isinstance(col_dict["sample_values"], list)
        assert isinstance(col_dict["warnings"], list)
    
    def test_column_profile_to_dict_redact_sample_values(self, tmp_path):
        """Verify redact_sample_values flag redacts sensitive data in to_dict."""
        csv_file = tmp_path / "sensitive.csv"
        csv_file.write_text("email,phone\nalice@example.com,555-1234\nbob@example.com,555-5678\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        
        col_dict_unredacted = report.columns["email"].to_dict(redact_sample_values=False)
        col_dict_redacted = report.columns["email"].to_dict(redact_sample_values=True)
        
        # Redacted should contain [REDACTED]
        assert any("[REDACTED]" in str(v) for v in col_dict_redacted["sample_values"])
        # Unredacted should have real values
        assert any("example.com" in str(v) for v in col_dict_unredacted["sample_values"])


class TestDataQualityReportToDict:
    """Tests for DataQualityReport.to_dict()."""
    
    def test_report_to_dict_basic_structure(self, tmp_path):
        """Profile data and verify report to_dict has required structure."""
        csv_file = tmp_path / "report.csv"
        csv_file.write_text("id,name,active\n1,Alice,true\n2,Bob,false\n1,Alice,true\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        report_dict = report.to_dict()
        
        assert report_dict["row_count"] == 3
        assert report_dict["column_count"] == 3
        assert "quality_score" in report_dict
        assert "columns" in report_dict
        assert isinstance(report_dict["columns"], dict)
        assert set(report_dict["columns"].keys()) == {"id", "name", "active"}
    
    def test_report_to_dict_exclude_columns(self, tmp_path):
        """Test to_dict correctly excludes specified columns."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("public,secret,internal\nA,hidden1,debug1\nB,hidden2,debug2\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        
        report_dict = report.to_dict(exclude_columns=["secret"])
        
        assert "public" in report_dict["columns"]
        assert "secret" not in report_dict["columns"]
        assert "internal" in report_dict["columns"]
    
    def test_report_to_dict_exclude_columns_validation_unknown(self, tmp_path):
        """Verify to_dict raises KeyError for non-existent exclude_columns."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2\n3,4\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        
        with pytest.raises(KeyError, match="Unknown exclude_columns"):
            report.to_dict(exclude_columns=["nonexistent"])
    
    def test_report_to_dict_exclude_columns_validation_type(self, tmp_path):
        """Verify to_dict raises TypeError for invalid exclude_columns type."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        
        with pytest.raises(TypeError, match="exclude_columns must be"):
            report.to_dict(exclude_columns="a")
    
    def test_report_to_dict_with_redaction(self, tmp_path):
        """Verify redact_sample_values redacts sensitive column values."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("ssn,name\n123-45-6789,Alice\n987-65-4321,Bob\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        
        report_dict_redacted = report.to_dict(redact_sample_values=True)
        report_dict_normal = report.to_dict(redact_sample_values=False)
        
        redacted_samples = report_dict_redacted["columns"]["ssn"]["sample_values"]
        normal_samples = report_dict_normal["columns"]["ssn"]["sample_values"]
        
        assert any("[REDACTED]" in str(v) for v in redacted_samples)
        assert any("123" in str(v) or "987" in str(v) for v in normal_samples)
    
    def test_report_to_dict_includes_suggestions(self, tmp_path):
        """Verify to_dict includes cleaning suggestions."""
        csv_file = tmp_path / "messy.csv"
        csv_file.write_text("name,age\n  Alice  ,30\n  Bob  ,25\nAlice,30\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        report_dict = report.to_dict()
        
        assert "suggestions" in report_dict
        assert isinstance(report_dict["suggestions"], list)


class TestDataQualityReportToJson:
    """Tests for DataQualityReport.to_json()."""
    
    def test_report_to_json_returns_valid_json_string(self, tmp_path):
        """Verify to_json returns valid JSON string."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,value\n1,100\n2,200\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        json_str = report.to_json()
        
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["row_count"] == 2
        assert "id" in parsed["columns"]
    
    def test_report_to_json_with_indent(self, tmp_path):
        """Verify to_json respects indent parameter."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        json_str = report.to_json(indent=2)
        
        # Pretty-printed JSON should have newlines
        assert "\n" in json_str
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
    
    def test_report_to_json_to_file_output(self, tmp_path):
        """Verify to_json writes to file and returns None when output provided."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name\n1,Alice\n2,Bob\n")
        json_file = tmp_path / "report.json"
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        
        with open(json_file, "w") as f:
            result = report.to_json(output=f)
        
        # Should return None when output provided
        assert result is None
        # File should contain valid JSON
        assert json_file.exists()
        with open(json_file) as f:
            parsed = json.load(f)
        assert parsed["row_count"] == 2
    
    def test_report_to_json_with_exclude_columns(self, tmp_path):
        """Verify to_json respects exclude_columns parameter."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("public,secret\nA,hidden\nB,shh\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        json_str = report.to_json(exclude_columns=["secret"])
        
        parsed = json.loads(json_str)
        assert "public" in parsed["columns"]
        assert "secret" not in parsed["columns"]
    
    def test_report_to_json_invalid_output_type(self, tmp_path):
        """Verify to_json raises TypeError for non-writable output."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        
        with pytest.raises(TypeError, match="output must be a writable"):
            report.to_json(output="not_a_stream")


class TestScoreBreakdown:
    """Tests for DataQualityReport.score_breakdown()."""
    
    def test_score_breakdown_structure(self, tmp_path):
        """Verify score_breakdown returns dict with penalty components."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name\n1,Alice\n2,Bob\n1,Alice\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        breakdown = report.score_breakdown()
        
        assert isinstance(breakdown, dict)
        assert "null_penalty" in breakdown
        assert "duplicate_penalty" in breakdown
        assert "type_mismatch_penalty" in breakdown
        assert "final_score" in breakdown
        
        # All penalties should be non-negative floats
        for key in ["null_penalty", "duplicate_penalty", "type_mismatch_penalty"]:
            assert isinstance(breakdown[key], (int, float))
            assert breakdown[key] >= 0.0
        
        # Final score should be between 0 and 1
        assert 0.0 <= breakdown["final_score"] <= 1.0
    
    def test_score_breakdown_with_duplicates(self, tmp_path):
        """Verify score_breakdown reflects duplicate row penalty."""
        csv_file = tmp_path / "dups.csv"
        csv_file.write_text("id,val\n1,100\n1,100\n1,100\n2,200\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        breakdown = report.score_breakdown()
        
        # Should have duplicate penalty when duplicates exist
        assert breakdown["duplicate_penalty"] > 0.0


class TestIntegrationWorkflows:
    """Integration tests combining profiling, cleaning, and serialization."""
    
    def test_messy_to_clean_profile_serialize_workflow(self, tmp_path):
        """Full workflow: load messy -> clean -> profile -> to_dict -> to_json."""
        csv_file = tmp_path / "messy.csv"
        csv_file.write_text(
            "name,score,active\n"
            "  alice  ,95,  true  \n"
            "  bob  ,88,false\n"
            "  alice  ,95,  true  \n"
        )
        
        # Load and profile before cleaning
        frame = ar.read_csv(str(csv_file))
        report_before = ar.profile(frame)
        breakdown_before = report_before.score_breakdown()
        
        # Clean the data
        clean = ar.pipeline(frame, [
            ("strip_whitespace",),
            ("drop_duplicates",),
        ])
        
        # Profile after cleaning
        report_after = ar.profile(clean)
        breakdown_after = report_after.score_breakdown()
        
        # Quality should improve
        assert breakdown_after["final_score"] >= breakdown_before["final_score"]
        
        # Serialize to dict and JSON
        dict_repr = report_after.to_dict()
        assert dict_repr["row_count"] == 2
        
        json_str = report_after.to_json(indent=2)
        parsed = json.loads(json_str)
        assert parsed["duplicate_ratio"] == 0.0
    
    def test_pandas_accessor_profile_and_serialize(self, tmp_path):
        """Test pandas .arnio accessor with profiling and serialization."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("x,y\n10,20\n30,40\n50,60\n")
        
        pdf = pd.read_csv(csv_file)
        quality = pdf.arnio.profile()
        
        # Serialize to dict
        quality_dict = quality.to_dict()
        assert quality_dict["row_count"] == 3
        assert quality_dict["column_count"] == 2
        
        # Serialize to JSON
        json_str = quality.to_json()
        parsed = json.loads(json_str)
        assert isinstance(parsed["columns"]["x"]["sample_values"], list)
    
    def test_profile_exclude_suggest_exclude_workflow(self, tmp_path):
        """Test excluding columns from profile and suggestions."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,debug_col,value\n1,temp,100\n2,temp,200\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        
        # Get dict excluding debug column
        report_dict = report.to_dict(exclude_columns=["debug_col"])
        
        assert "id" in report_dict["columns"]
        assert "debug_col" not in report_dict["columns"]
        assert "value" in report_dict["columns"]
        
        # Suggestions should not reference excluded columns
        for suggestion in report_dict["suggestions"]:
            if "subset" in suggestion["kwargs"]:
                assert "debug_col" not in suggestion["kwargs"]["subset"]
    
    def test_empty_frame_profile_to_json(self, tmp_path):
        """Edge case: profile empty frame and serialize."""
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("a,b,c\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        
        assert report.row_count == 0
        
        # Should serialize cleanly
        breakdown = report.score_breakdown()
        assert isinstance(breakdown, dict)
        
        report_dict = report.to_dict()
        assert report_dict["row_count"] == 0
        
        json_str = report.to_json()
        parsed = json.loads(json_str)
        assert parsed["row_count"] == 0
    
    def test_profile_with_nulls_and_redaction(self, tmp_path):
        """Test profile with null values and redaction+exclusion."""
        csv_file = tmp_path / "nulls.csv"
        csv_file.write_text(
            "id,email,score\n"
            "1,alice@example.com,95\n"
            "2,,88\n"
            "3,charlie@example.com,\n"
        )
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        
        # Redact sensitive data and exclude score column
        report_dict = report.to_dict(
            redact_sample_values=True,
            exclude_columns=["score"]
        )
        
        # Email should be redacted
        email_samples = report_dict["columns"]["email"]["sample_values"]
        assert any("[REDACTED]" in str(v) for v in email_samples)
        
        # Score should be excluded
        assert "score" not in report_dict["columns"]
        
        # Should reflect null info
        assert report_dict["columns"]["email"]["null_count"] == 1
