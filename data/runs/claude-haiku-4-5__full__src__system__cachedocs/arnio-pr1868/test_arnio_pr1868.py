import json
import io
import pytest
import pandas as pd
import arnio as ar


class TestColumnProfileToDict:
    """Test ColumnProfile.to_dict method from real workflow."""

    def test_column_profile_to_dict_basic(self, tmp_path):
        """End-to-end: load CSV, profile, extract column dict."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name,age,score\nAlice,30,95.5\nBob,25,87.3\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame, sample_size=10)
        
        age_profile = report.columns["age"]
        result = age_profile.to_dict()
        
        assert result["name"] == "age"
        assert result["dtype"] == "int64"
        assert result["null_count"] == 0
        assert result["unique_count"] == 2
        assert isinstance(result["sample_values"], list)

    def test_column_profile_to_dict_with_redaction(self, tmp_path):
        """Verify redact_sample_values flag redacts sensitive data."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("email\nuser@example.com\nadmin@test.org\nsupport@domain.io\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame, sample_size=5)
        
        email_profile = report.columns["email"]
        
        # Unredacted should have real email addresses
        unredacted = email_profile.to_dict(redact_sample_values=False)
        assert any("@" in str(v) for v in unredacted["sample_values"])
        
        # Redacted should replace all sample values
        redacted = email_profile.to_dict(redact_sample_values=True)
        assert all(v == "[REDACTED]" for v in redacted["sample_values"])
        # Top values should also be redacted
        if redacted["top_values"]:
            assert all(item["value"] == "[REDACTED]" for item in redacted["top_values"])

    def test_column_profile_numeric_has_quantiles(self, tmp_path):
        """Verify numeric column profiles include quantile statistics."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("score\n10\n20\n30\n40\n50\n60\n70\n80\n90\n100\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        
        profile = report.columns["score"]
        result = profile.to_dict()
        
        assert result["dtype"] == "int64"
        assert "q25" in result
        assert "q50" in result
        assert "q75" in result
        assert "iqr" in result
        assert "outlier_lower_bound" in result

    def test_column_profile_string_no_quantiles(self, tmp_path):
        """Verify string column profiles do not include quantiles."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("city\nNew York\nLos Angeles\nChicago\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        
        profile = report.columns["city"]
        result = profile.to_dict()
        
        assert result["dtype"] == "string"
        assert "q25" not in result
        assert "iqr" not in result


class TestDataQualityReportScoreBreakdown:
    """Test DataQualityReport.score_breakdown from real workflow."""

    def test_score_breakdown_clean_data(self, tmp_path):
        """Verify score_breakdown returns expected penalties for clean data."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name,value\n1,Alice,100\n2,Bob,200\n3,Charlie,300\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        
        breakdown = report.score_breakdown()
        
        assert isinstance(breakdown, dict)
        assert "null_penalty" in breakdown
        assert "duplicate_penalty" in breakdown
        assert "type_mismatch_penalty" in breakdown
        assert "final_score" in breakdown
        assert breakdown["final_score"] == report.quality_score
        # Clean data should have high score
        assert breakdown["final_score"] >= 90.0

    def test_score_breakdown_with_nulls_reduces_score(self, tmp_path):
        """Verify nulls incur null_penalty in score breakdown."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name,email\n1,Alice,alice@example.com\n2,,\n3,Bob,bob@example.com\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        
        breakdown = report.score_breakdown()
        
        # Should have incurred some penalty
        assert isinstance(breakdown["null_penalty"], (int, float))
        assert breakdown["final_score"] < 100.0

    def test_score_breakdown_with_duplicates_reduces_score(self, tmp_path):
        """Verify duplicates incur duplicate_penalty."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name\n1,Alice\n1,Alice\n2,Bob\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        
        breakdown = report.score_breakdown()
        
        assert report.duplicate_ratio > 0
        assert breakdown["final_score"] < 100.0


class TestDataQualityReportToDict:
    """Test DataQualityReport.to_dict from real workflow."""

    def test_report_to_dict_basic_structure(self, tmp_path):
        """End-to-end: load CSV, profile, convert to dict."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name,score\n1,Alice,95\n2,Bob,87\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        
        result = report.to_dict()
        
        assert result["row_count"] == 2
        assert result["column_count"] == 3
        assert "id" in result["columns"]
        assert "name" in result["columns"]
        assert "score" in result["columns"]
        assert result["quality_score"] > 0
        assert result["duplicate_ratio"] == 0.0

    def test_report_to_dict_with_sample_redaction(self, tmp_path):
        """Verify redaction of sample values in to_dict output."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("ssn,name\n123-45-6789,Alice\n987-65-4321,Bob\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame, sample_size=10)
        
        # Without redaction
        unredacted = report.to_dict(redact_sample_values=False)
        ssn_col = unredacted["columns"]["ssn"]
        assert any("-" in str(v) for v in ssn_col["sample_values"])
        
        # With redaction
        redacted = report.to_dict(redact_sample_values=True)
        ssn_col_red = redacted["columns"]["ssn"]
        assert all(v == "[REDACTED]" for v in ssn_col_red["sample_values"])

    def test_report_to_dict_exclude_columns(self, tmp_path):
        """Verify exclude_columns parameter removes specified columns."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("ssn,credit_card,name,age\n123-45-6789,4111-1111-1111-1111,Alice,30\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        
        result = report.to_dict(exclude_columns=["ssn", "credit_card"])
        
        assert "ssn" not in result["columns"]
        assert "credit_card" not in result["columns"]
        assert "name" in result["columns"]
        assert "age" in result["columns"]

    def test_report_to_dict_exclude_columns_validation(self, tmp_path):
        """Verify error when excluding non-existent columns."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name\n1,Alice\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        
        with pytest.raises(KeyError, match="Unknown exclude_columns"):
            report.to_dict(exclude_columns=["nonexistent"])

    def test_report_to_dict_exclude_columns_wrong_type(self, tmp_path):
        """Verify error when exclude_columns is wrong type."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name\n1,Alice\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        
        with pytest.raises(TypeError, match="exclude_columns must be"):
            report.to_dict(exclude_columns="id")

    def test_report_to_dict_with_missing_values(self, tmp_path):
        """End-to-end: profile data with missing values, check dict."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,score,category\n1,95,A\n2,,B\n3,87,\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        
        result = report.to_dict()
        
        # Check that null info is present
        assert result["columns"]["score"]["null_count"] > 0
        assert result["columns"]["category"]["null_count"] > 0
        assert result["quality_score"] < 100.0


class TestDataQualityReportToJson:
    """Test DataQualityReport.to_json from real workflow."""

    def test_report_to_json_returns_string(self, tmp_path):
        """Verify to_json returns valid JSON string."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name\n1,Alice\n2,Bob\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        
        json_str = report.to_json()
        
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["row_count"] == 2
        assert parsed["column_count"] == 2
        assert "columns" in parsed

    def test_report_to_json_with_indent(self, tmp_path):
        """Verify indent parameter produces formatted JSON."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,value\n1,100\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        
        compact = report.to_json(indent=None)
        formatted = report.to_json(indent=2)
        
        assert isinstance(compact, str)
        assert isinstance(formatted, str)
        # Formatted should have more whitespace
        assert len(formatted) > len(compact)
        assert "\n" in formatted

    def test_report_to_json_write_to_file(self, tmp_path):
        """Verify to_json writes to file when output provided."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,score\n1,95\n2,87\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        
        output_file = tmp_path / "report.json"
        with open(output_file, "w") as f:
            result = report.to_json(output=f, indent=2)
        
        # When output is provided, returns None
        assert result is None
        
        # File should exist and contain valid JSON
        assert output_file.exists()
        content = output_file.read_text()
        parsed = json.loads(content)
        assert parsed["row_count"] == 2

    def test_report_to_json_invalid_output_param(self, tmp_path):
        """Verify error when output is not writable."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id\n1\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        
        with pytest.raises(TypeError, match="output must be a writable text stream"):
            report.to_json(output="not_a_file_object")

    def test_report_to_json_with_redaction_and_exclusion(self, tmp_path):
        """End-to-end: JSON export with both redaction and column exclusion."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("email,ssn,name\nuser@example.com,123-45-6789,Alice\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame, sample_size=10)
        
        json_str = report.to_json(
            redact_sample_values=True,
            exclude_columns=["ssn"]
        )
        
        parsed = json.loads(json_str)
        
        # SSN should be excluded
        assert "ssn" not in parsed["columns"]
        # Email should be present but redacted
        assert "email" in parsed["columns"]
        email_col = parsed["columns"]["email"]
        assert all(v == "[REDACTED]" for v in email_col["sample_values"])

    def test_report_to_json_empty_frame(self, tmp_path):
        """Verify to_json handles empty dataframe correctly."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name,value\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        
        json_str = report.to_json()
        parsed = json.loads(json_str)
        
        assert parsed["row_count"] == 0
        assert parsed["column_count"] == 3

    def test_report_to_json_roundtrip_consistency(self, tmp_path):
        """Verify to_json and to_dict produce consistent results."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,score,status\n1,95,active\n2,87,inactive\n3,92,active\n")
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        
        # Get results both ways
        dict_result = report.to_dict()
        json_result = json.loads(report.to_json())
        
        # Should be identical
        assert dict_result == json_result
        
        # Test with options too
        dict_redacted = report.to_dict(redact_sample_values=True, exclude_columns=["id"])
        json_redacted = json.loads(
            report.to_json(redact_sample_values=True, exclude_columns=["id"])
        )
        assert dict_redacted == json_redacted

    def test_report_to_json_with_various_dtypes(self, tmp_path):
        """End-to-end: JSON export of mixed data types."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            "int_col,float_col,str_col\n10,1.5,hello\n20,2.5,world\n"
        )
        
        frame = ar.read_csv(str(csv_file))
        report = ar.profile(frame)
        
        json_str = report.to_json(indent=2)
        parsed = json.loads(json_str)
        
        # Verify all columns are represented
        assert "int_col" in parsed["columns"]
        assert "float_col" in parsed["columns"]
        assert "str_col" in parsed["columns"]
        
        # Verify correct dtypes
        assert parsed["columns"]["int_col"]["dtype"] == "int64"
        assert parsed["columns"]["float_col"]["dtype"] == "float64"
        assert parsed["columns"]["str_col"]["dtype"] == "string"
