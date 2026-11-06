import pytest
import pandas as pd
import numpy as np
import arnio as ar
import json
import io


class TestColumnProfileToDict:
    """Test ColumnProfile.to_dict() method."""
    
    def test_to_dict_basic_keys_present(self):
        """Test that to_dict returns essential structure."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        col = report.columns["col"]
        
        result = col.to_dict()
        
        assert isinstance(result, dict)
        assert "name" in result
        assert "dtype" in result
        assert "row_count" in result
        assert "sample_values" in result
    
    def test_to_dict_redact_sample_values_true(self):
        """Test that redact_sample_values=True replaces all sample values."""
        df = pd.DataFrame({"col": ["secret1", "secret2", "secret3"]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        col = report.columns["col"]
        
        result = col.to_dict(redact_sample_values=True)
        
        assert all(v == "[REDACTED]" for v in result["sample_values"])
    
    def test_to_dict_redact_sample_values_false(self):
        """Test that redact_sample_values=False includes actual values."""
        df = pd.DataFrame({"col": ["value1", "value2"]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        col = report.columns["col"]
        
        result = col.to_dict(redact_sample_values=False)
        
        # Should have actual values, not redacted
        assert "[REDACTED]" not in result["sample_values"]
    
    def test_to_dict_numeric_includes_quantile_fields(self):
        """Test that numeric columns include quantile and outlier fields."""
        df = pd.DataFrame({"num": [1, 2, 3, 4, 5, 100]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        col = report.columns["num"]
        
        result = col.to_dict()
        
        # Numeric-specific fields should be present
        assert "q25" in result
        assert "q50" in result
        assert "q75" in result
        assert "iqr" in result
        assert "outlier_count" in result
        assert "outlier_upper_bound" in result


class TestDataQualityReportScoreBreakdown:
    """Test DataQualityReport.score_breakdown() method."""
    
    def test_score_breakdown_returns_required_keys(self):
        """Test that score_breakdown returns all documented keys."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        breakdown = report.score_breakdown()
        
        assert isinstance(breakdown, dict)
        assert "null_penalty" in breakdown
        assert "duplicate_penalty" in breakdown
        assert "type_mismatch_penalty" in breakdown
        assert "final_score" in breakdown
    
    def test_score_breakdown_all_values_numeric(self):
        """Test that all breakdown values are numeric (int or float)."""
        df = pd.DataFrame({"a": [1, None, 3], "b": ["x", "y", "x"]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        breakdown = report.score_breakdown()
        
        for key, value in breakdown.items():
            assert isinstance(value, (int, float)), f"{key} value {value} is not numeric"
    
    def test_score_breakdown_final_score_matches_quality_score(self):
        """Test that final_score in breakdown equals report.quality_score."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        breakdown = report.score_breakdown()
        
        assert breakdown["final_score"] == report.quality_score


class TestDataQualityReportToDict:
    """Test DataQualityReport.to_dict() method."""
    
    def test_to_dict_includes_top_level_fields(self):
        """Test that to_dict includes all documented top-level fields."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        result = report.to_dict()
        
        assert "row_count" in result
        assert "column_count" in result
        assert "memory_usage" in result
        assert "quality_score" in result
        assert "columns" in result
        assert "suggestions" in result
        assert isinstance(result["columns"], dict)
    
    def test_to_dict_exclude_columns_filters_columns(self):
        """Test that exclude_columns parameter removes specified columns."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        result = report.to_dict(exclude_columns=["b"])
        
        assert "a" in result["columns"]
        assert "b" not in result["columns"]
        assert "c" in result["columns"]
    
    def test_to_dict_exclude_columns_unknown_raises_keyerror(self):
        """Test that unknown column in exclude_columns raises KeyError."""
        df = pd.DataFrame({"a": [1, 2]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        with pytest.raises(KeyError, match="Unknown exclude_columns"):
            report.to_dict(exclude_columns=["nonexistent"])
    
    def test_to_dict_exclude_columns_accepts_multiple_types(self):
        """Test that exclude_columns accepts list, tuple, and set."""
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        result_list = report.to_dict(exclude_columns=["b"])
        result_tuple = report.to_dict(exclude_columns=("b",))
        result_set = report.to_dict(exclude_columns={"b"})
        
        assert "b" not in result_list["columns"]
        assert "b" not in result_tuple["columns"]
        assert "b" not in result_set["columns"]
    
    def test_to_dict_redact_sample_values_true(self):
        """Test that redact_sample_values=True redacts column sample values."""
        df = pd.DataFrame({"secret": ["x", "y", "z"]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        result = report.to_dict(redact_sample_values=True)
        
        col_dict = result["columns"]["secret"]
        assert all(v == "[REDACTED]" for v in col_dict["sample_values"])
    
    def test_to_dict_invalid_exclude_columns_type_raises_typeerror(self):
        """Test that exclude_columns with wrong type raises TypeError."""
        df = pd.DataFrame({"a": [1]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        with pytest.raises(TypeError, match="exclude_columns must be"):
            report.to_dict(exclude_columns="a")  # string instead of container


class TestDataQualityReportToJson:
    """Test DataQualityReport.to_json() method."""
    
    def test_to_json_returns_json_string_by_default(self):
        """Test that to_json returns valid JSON string when output=None."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        result = report.to_json()
        
        assert isinstance(result, str)
        # Should be valid JSON that can be parsed
        parsed = json.loads(result)
        assert "row_count" in parsed
    
    def test_to_json_indent_parameter_formats_output(self):
        """Test that indent parameter affects JSON formatting."""
        df = pd.DataFrame({"a": [1, 2]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        compact = report.to_json(indent=None)
        formatted = report.to_json(indent=2)
        
        # Formatted should be longer due to indentation
        assert len(formatted) > len(compact)
        # Both should parse to equivalent data
        assert json.loads(compact) == json.loads(formatted)
    
    def test_to_json_writes_to_output_parameter(self):
        """Test that output parameter writes to file-like object and returns None."""
        df = pd.DataFrame({"a": [1, 2]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        output = io.StringIO()
        result = report.to_json(output=output)
        
        # Should return None when writing to output
        assert result is None
        # Should have written valid JSON to output
        output.seek(0)
        written = output.read()
        assert json.loads(written)
    
    def test_to_json_invalid_output_type_raises_typeerror(self):
        """Test that non-writable output parameter raises TypeError."""
        df = pd.DataFrame({"a": [1]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        with pytest.raises(TypeError, match="output must be a writable text stream"):
            report.to_json(output=123)  # not file-like
    
    def test_to_json_respects_exclude_columns(self):
        """Test that exclude_columns parameter filters columns in JSON."""
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        json_str = report.to_json(exclude_columns=["b"])
        parsed = json.loads(json_str)
        
        assert "a" in parsed["columns"]
        assert "b" not in parsed["columns"]
        assert "c" in parsed["columns"]
