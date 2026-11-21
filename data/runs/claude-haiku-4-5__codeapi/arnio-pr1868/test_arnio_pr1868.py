import pytest
import json
import io
from arnio.quality import ColumnProfile, DataQualityReport


class TestColumnProfileToDict:
    """Tests for ColumnProfile.to_dict method."""
    
    def test_numeric_dtype_includes_quantiles(self):
        """Numeric columns should include quantile statistics in the output."""
        profile = ColumnProfile(
            name="age",
            dtype="int64",
            semantic_type="numeric",
            row_count=100,
            null_count=5,
            null_ratio=0.05,
            unique_count=50,
            unique_ratio=0.5,
            q25=30,
            q50=45,
            q75=60,
            q95=90,
            outlier_count=2,
        )
        result = profile.to_dict()
        assert result["q25"] == 30
        assert result["q50"] == 45
        assert result["q75"] == 60
        assert result["outlier_count"] == 2
    
    def test_string_dtype_excludes_quantiles(self):
        """String columns should not include quantile statistics."""
        profile = ColumnProfile(
            name="name",
            dtype="string",
            semantic_type="text",
            row_count=100,
            null_count=0,
            null_ratio=0.0,
            unique_count=95,
            unique_ratio=0.95,
        )
        result = profile.to_dict()
        assert "q25" not in result
        assert "q75" not in result
    
    def test_sample_values_redaction(self):
        """Sample values and top_values should be redacted when redact_sample_values=True."""
        profile = ColumnProfile(
            name="secret",
            dtype="string",
            semantic_type="text",
            row_count=10,
            null_count=0,
            null_ratio=0.0,
            unique_count=5,
            unique_ratio=0.5,
            sample_values=["secret1", "secret2"],
            top_values=[("val1", 5, 0.5), ("val2", 5, 0.5)],
        )
        result_redacted = profile.to_dict(redact_sample_values=True)
        assert all(v == "[REDACTED]" for v in result_redacted["sample_values"])
        assert all(v["value"] == "[REDACTED]" for v in result_redacted["top_values"])
        
        result_plain = profile.to_dict(redact_sample_values=False)
        assert result_plain["sample_values"] == ["secret1", "secret2"]


class TestDataQualityReportScoreBreakdown:
    """Tests for DataQualityReport.score_breakdown method."""
    
    def test_score_breakdown_structure_with_penalties(self):
        """Score breakdown should return dict with penalty keys and final_score."""
        report = DataQualityReport(
            row_count=100,
            column_count=1,
            memory_usage=10000,
            duplicate_rows=0,
            duplicate_ratio=0.0,
            columns={},
            quality_score=85.0,
            score_components={
                "null_penalty": 10.0,
                "duplicate_penalty": 3.0,
                "type_mismatch_penalty": 2.0,
            },
        )
        breakdown = report.score_breakdown()
        assert breakdown["null_penalty"] == 10.0
        assert breakdown["duplicate_penalty"] == 3.0
        assert breakdown["type_mismatch_penalty"] == 2.0
        assert breakdown["final_score"] == 85.0
    
    def test_score_breakdown_missing_penalties_default_to_zero(self):
        """Missing penalty components should default to 0.0."""
        report = DataQualityReport(
            row_count=100,
            column_count=1,
            memory_usage=10000,
            duplicate_rows=0,
            duplicate_ratio=0.0,
            columns={},
            quality_score=100.0,
        )
        breakdown = report.score_breakdown()
        assert breakdown["null_penalty"] == 0.0
        assert breakdown["duplicate_penalty"] == 0.0
        assert breakdown["type_mismatch_penalty"] == 0.0


class TestDataQualityReportToDict:
    """Tests for DataQualityReport.to_dict method."""
    
    @pytest.fixture
    def report_with_columns(self):
        col1 = ColumnProfile(
            name="col1",
            dtype="int64",
            semantic_type="numeric",
            row_count=100,
            null_count=0,
            null_ratio=0.0,
            unique_count=100,
            unique_ratio=1.0,
        )
        col2 = ColumnProfile(
            name="col2",
            dtype="string",
            semantic_type="text",
            row_count=100,
            null_count=5,
            null_ratio=0.05,
            unique_count=95,
            unique_ratio=0.95,
        )
        return DataQualityReport(
            row_count=100,
            column_count=2,
            memory_usage=10000,
            duplicate_rows=0,
            duplicate_ratio=0.0,
            columns={"col1": col1, "col2": col2},
            quality_score=95.0,
        )
    
    def test_to_dict_basic_output(self, report_with_columns):
        """to_dict should return a complete dictionary representation."""
        result = report_with_columns.to_dict()
        assert result["row_count"] == 100
        assert result["column_count"] == 2
        assert result["quality_score"] == 95.0
        assert "col1" in result["columns"]
        assert "col2" in result["columns"]
    
    def test_exclude_columns_validation_and_filtering(self, report_with_columns):
        """exclude_columns should validate inputs and filter columns."""
        # Unknown columns should raise KeyError
        with pytest.raises(KeyError):
            report_with_columns.to_dict(exclude_columns=["unknown_col"])
        
        # Valid exclusion should work
        result = report_with_columns.to_dict(exclude_columns=["col1"])
        assert "col1" not in result["columns"]
        assert "col2" in result["columns"]
        
        # Set and tuple should work
        result_set = report_with_columns.to_dict(exclude_columns={"col2"})
        assert "col2" not in result_set["columns"]
    
    def test_exclude_columns_type_validation(self, report_with_columns):
        """exclude_columns type validation should enforce list/tuple/set."""
        with pytest.raises(TypeError):
            report_with_columns.to_dict(exclude_columns="col1")
        
        with pytest.raises(TypeError):
            report_with_columns.to_dict(exclude_columns=[1, "col1"])


class TestDataQualityReportToJson:
    """Tests for DataQualityReport.to_json method."""
    
    @pytest.fixture
    def simple_report(self):
        col = ColumnProfile(
            name="value",
            dtype="int64",
            semantic_type="numeric",
            row_count=50,
            null_count=2,
            null_ratio=0.04,
            unique_count=48,
            unique_ratio=0.96,
        )
        return DataQualityReport(
            row_count=50,
            column_count=1,
            memory_usage=5000,
            duplicate_rows=0,
            duplicate_ratio=0.0,
            columns={"value": col},
            quality_score=98.0,
        )
    
    def test_to_json_returns_valid_json_string(self, simple_report):
        """to_json should return a valid JSON string by default."""
        result = simple_report.to_json()
        assert isinstance(result, str)
        data = json.loads(result)
        assert data["row_count"] == 50
        assert data["quality_score"] == 98.0
    
    def test_to_json_with_indent(self, simple_report):
        """to_json should apply indent parameter for formatting."""
        json_str = simple_report.to_json(indent=2)
        assert isinstance(json_str, str)
        assert "\n" in json_str  # Indented JSON contains newlines
        data = json.loads(json_str)
        assert data["row_count"] == 50
    
    def test_to_json_with_output_parameter(self, simple_report):
        """to_json should write to output stream and return None when output is provided."""
        output = io.StringIO()
        result = simple_report.to_json(output=output)
        assert result is None
        output_str = output.getvalue()
        assert len(output_str) > 0
        data = json.loads(output_str)
        assert data["row_count"] == 50
    
    def test_to_json_invalid_output_parameter(self, simple_report):
        """to_json should raise TypeError for invalid output parameter."""
        with pytest.raises(TypeError):
            simple_report.to_json(output="not_a_file")
        
        with pytest.raises(TypeError):
            simple_report.to_json(output=123)
    
    def test_to_json_exclude_columns_and_redaction(self, simple_report):
        """to_json should support exclude_columns and redact_sample_values parameters."""
        # Test exclude_columns
        json_str = simple_report.to_json(exclude_columns=["value"])
        data = json.loads(json_str)
        assert "value" not in data["columns"]
        
        # Test redact_sample_values
        report_with_samples = DataQualityReport(
            row_count=10,
            column_count=1,
            memory_usage=1000,
            duplicate_rows=0,
            duplicate_ratio=0.0,
            columns={"secret": ColumnProfile(
                name="secret",
                dtype="string",
                semantic_type="text",
                row_count=10,
                null_count=0,
                null_ratio=0.0,
                unique_count=10,
                unique_ratio=1.0,
                sample_values=["password1", "password2"],
            )},
        )
        json_str_redacted = report_with_samples.to_json(redact_sample_values=True)
        data_redacted = json.loads(json_str_redacted)
        assert all(v == "[REDACTED]" for v in data_redacted["columns"]["secret"]["sample_values"])
