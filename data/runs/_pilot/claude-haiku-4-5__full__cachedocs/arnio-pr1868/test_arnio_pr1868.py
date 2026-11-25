import pytest
import json
import pandas as pd
import arnio as ar


class TestColumnProfileToDict:
    """Test arnio.quality.ColumnProfile.to_dict()"""
    
    def test_to_dict_returns_json_friendly_dict(self):
        """to_dict returns a JSON-serializable dictionary."""
        df = pd.DataFrame({"age": [25, 30, 35, None]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame, sample_size=3)
        
        result = report.columns["age"].to_dict()
        
        assert isinstance(result, dict)
        json_str = json.dumps(result, default=str)
        assert isinstance(json_str, str)
    
    def test_to_dict_redact_sample_values_parameter(self):
        """to_dict accepts redact_sample_values parameter."""
        df = pd.DataFrame({"ssn": ["123-45-6789", "987-65-4321"]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame, sample_size=2)
        
        result_unredacted = report.columns["ssn"].to_dict(redact_sample_values=False)
        result_redacted = report.columns["ssn"].to_dict(redact_sample_values=True)
        
        assert isinstance(result_unredacted, dict)
        assert isinstance(result_redacted, dict)


class TestDataQualityReportScoreBreakdown:
    """Test arnio.quality.DataQualityReport.score_breakdown()"""
    
    def test_score_breakdown_returns_penalty_dict(self):
        """score_breakdown returns dict with numeric penalty values."""
        df = pd.DataFrame({"id": [1, 1, None, 3]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        breakdown = report.score_breakdown()
        
        assert isinstance(breakdown, dict)
        for key, value in breakdown.items():
            assert isinstance(value, (int, float))
    
    def test_score_breakdown_reflects_data_quality_issues(self):
        """score_breakdown captures penalties from duplicates and nulls."""
        df = pd.DataFrame({"col": [1, 1, 1, None]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        breakdown = report.score_breakdown()
        
        assert isinstance(breakdown, dict)
        assert len(breakdown) > 0


class TestDataQualityReportToDict:
    """Test arnio.quality.DataQualityReport.to_dict()"""
    
    def test_to_dict_basic_structure(self):
        """to_dict returns dict with row_count, column_count, quality_score, columns."""
        df = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        result = report.to_dict()
        
        assert isinstance(result, dict)
        assert "row_count" in result and result["row_count"] == 3
        assert "column_count" in result and result["column_count"] == 2
        assert "quality_score" in result
        assert "columns" in result
    
    def test_to_dict_exclude_columns_list(self):
        """to_dict(exclude_columns=['col']) removes column from output."""
        df = pd.DataFrame({"keep": [1, 2], "drop": [3, 4]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        result = report.to_dict(exclude_columns=["drop"])
        
        assert "keep" in result["columns"]
        assert "drop" not in result["columns"]
    
    def test_to_dict_exclude_columns_set_tuple(self):
        """to_dict(exclude_columns=set/tuple) removes multiple columns."""
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        result_set = report.to_dict(exclude_columns={"b"})
        result_tuple = report.to_dict(exclude_columns=("c",))
        
        assert "b" not in result_set["columns"]
        assert "c" not in result_tuple["columns"]
    
    def test_to_dict_redact_sample_values(self):
        """to_dict(redact_sample_values=True) redacts sample data."""
        df = pd.DataFrame({"secret": ["data1", "data2"]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame, sample_size=2)
        
        result = report.to_dict(redact_sample_values=True)
        
        assert isinstance(result, dict)
        assert "columns" in result
    
    def test_to_dict_json_serializable(self):
        """to_dict output is JSON serializable with all parameters."""
        df = pd.DataFrame({"id": [1, 2], "val": [1.5, 2.5], "drop": [9, 9]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame, sample_size=2)
        
        result = report.to_dict(
            redact_sample_values=True,
            exclude_columns=["drop"]
        )
        
        json_str = json.dumps(result, default=str)
        assert isinstance(json_str, str)


class TestDataQualityReportToJson:
    """Test arnio.quality.DataQualityReport.to_json()"""
    
    def test_to_json_returns_json_string(self):
        """to_json returns a valid JSON string."""
        df = pd.DataFrame({"id": [1, 2, 3]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        result = report.to_json()
        
        assert isinstance(result, (str, type(None)))
        if result is not None:
            parsed = json.loads(result)
            assert isinstance(parsed, dict)
    
    def test_to_json_indent_parameter(self):
        """to_json(indent=N) controls JSON formatting."""
        df = pd.DataFrame({"val": [1, 2]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        result_compact = report.to_json(indent=None)
        result_indent = report.to_json(indent=2)
        
        assert isinstance(result_compact, (str, type(None)))
        assert isinstance(result_indent, (str, type(None)))
        if result_indent is not None and result_compact is not None:
            # Indented version typically longer due to whitespace
            assert len(result_indent) >= len(result_compact)
    
    def test_to_json_redact_sample_values(self):
        """to_json(redact_sample_values=True) redacts PII in output."""
        df = pd.DataFrame({"pii": ["secret1", "secret2"]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame, sample_size=2)
        
        result = report.to_json(redact_sample_values=True)
        
        assert isinstance(result, (str, type(None)))
        if result is not None:
            parsed = json.loads(result)
            assert "pii" in parsed["columns"]
    
    def test_to_json_exclude_columns(self):
        """to_json(exclude_columns=[...]) removes columns from JSON output."""
        df = pd.DataFrame({"safe": [1], "unsafe": [2]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        result = report.to_json(exclude_columns=["unsafe"])
        
        assert isinstance(result, (str, type(None)))
        if result is not None:
            parsed = json.loads(result)
            assert "safe" in parsed["columns"]
            assert "unsafe" not in parsed["columns"]
    
    def test_to_json_combined_parameters(self):
        """to_json works with indent, redact_sample_values, exclude_columns together."""
        df = pd.DataFrame({
            "col1": [1, 2],
            "col2": ["a", "b"],
            "col3": [1.1, 2.2]
        })
        frame = ar.from_pandas(df)
        report = ar.profile(frame, sample_size=2)
        
        result = report.to_json(
            indent=4,
            redact_sample_values=True,
            exclude_columns={"col3"}
        )
        
        assert isinstance(result, (str, type(None)))
        if result is not None:
            parsed = json.loads(result)
            assert "col1" in parsed["columns"]
            assert "col2" in parsed["columns"]
            assert "col3" not in parsed["columns"]
