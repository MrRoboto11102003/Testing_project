import pytest
import json
import io
import pandas as pd
import numpy as np
import arnio as ar


class TestColumnProfileToDict:
    """Tests for ColumnProfile.to_dict"""

    def test_to_dict_basic_structure(self):
        """Test basic to_dict output contains required fields"""
        df = pd.DataFrame({"age": [25, 30, None]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        profile = report.columns["age"]

        result = profile.to_dict()

        assert isinstance(result, dict)
        assert result["name"] == "age"
        assert "dtype" in result
        assert "null_count" in result
        assert "sample_values" in result
        assert "warnings" in result

    def test_to_dict_sample_values_redaction(self):
        """Test that redact_sample_values=True redacts sample values"""
        df = pd.DataFrame({"email": ["alice@ex.com", "bob@ex.com"]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame, sample_size=2)
        profile = report.columns["email"]

        result = profile.to_dict(redact_sample_values=True)

        assert all(v == "[REDACTED]" for v in result["sample_values"])

    def test_to_dict_numeric_includes_quantiles(self):
        """Test numeric columns include quantile and outlier fields"""
        df = pd.DataFrame({"score": [1.0, 2.0, 3.0, 4.0, 5.0]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        profile = report.columns["score"]

        result = profile.to_dict()

        assert "q25" in result
        assert "q50" in result
        assert "iqr" in result
        assert "outlier_lower_bound" in result

    def test_to_dict_string_excludes_quantiles(self):
        """Test string columns exclude quantile fields"""
        df = pd.DataFrame({"name": ["alice", "bob", "charlie"]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        profile = report.columns["name"]

        result = profile.to_dict()

        assert "q25" not in result
        assert "iqr" not in result
        assert "outlier_lower_bound" not in result


class TestDataQualityReportScoreBreakdown:
    """Tests for DataQualityReport.score_breakdown"""

    def test_score_breakdown_has_required_keys(self):
        """Test score_breakdown returns all required penalty keys"""
        df = pd.DataFrame({"id": [1, 2, 3]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)

        breakdown = report.score_breakdown()

        assert "null_penalty" in breakdown
        assert "duplicate_penalty" in breakdown
        assert "type_mismatch_penalty" in breakdown
        assert "final_score" in breakdown

    def test_score_breakdown_final_score_matches_quality_score(self):
        """Test final_score equals report quality_score"""
        df = pd.DataFrame({"col": [1, 2, None]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)

        breakdown = report.score_breakdown()

        assert breakdown["final_score"] == report.quality_score

    def test_score_breakdown_all_values_numeric(self):
        """Test all breakdown values are numeric"""
        df = pd.DataFrame({"col": [1, 1, 1, None, None]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)

        breakdown = report.score_breakdown()

        for value in breakdown.values():
            assert isinstance(value, (int, float))


class TestDataQualityReportToDict:
    """Tests for DataQualityReport.to_dict"""

    def test_to_dict_basic_structure(self):
        """Test to_dict includes expected top-level fields"""
        df = pd.DataFrame({"id": [1, 2], "name": ["a", "b"]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)

        result = report.to_dict()

        assert "row_count" in result
        assert "column_count" in result
        assert "memory_usage" in result
        assert "quality_score" in result
        assert "columns" in result
        assert isinstance(result["columns"], dict)

    def test_to_dict_exclude_columns_filters_output(self):
        """Test exclude_columns removes columns from output"""
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)

        result = report.to_dict(exclude_columns=["a", "b"])

        assert "a" not in result["columns"]
        assert "b" not in result["columns"]
        assert "c" in result["columns"]

    def test_to_dict_exclude_columns_accepts_multiple_types(self):
        """Test exclude_columns accepts list, tuple, and set"""
        df = pd.DataFrame({"x": [1], "y": [2]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)

        for exclude in [["x"], ("x",), {"x"}]:
            result = report.to_dict(exclude_columns=exclude)
            assert "x" not in result["columns"]
            assert "y" in result["columns"]

    def test_to_dict_exclude_columns_invalid_type_raises_typeerror(self):
        """Test invalid exclude_columns type raises TypeError"""
        df = pd.DataFrame({"col": [1]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)

        with pytest.raises(TypeError, match="exclude_columns must be"):
            report.to_dict(exclude_columns="col")

    def test_to_dict_exclude_columns_nonstring_items_raises_typeerror(self):
        """Test non-string column names in exclude_columns raise TypeError"""
        df = pd.DataFrame({"col": [1]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)

        with pytest.raises(TypeError, match="exclude_columns must contain only string"):
            report.to_dict(exclude_columns=[1, 2])

    def test_to_dict_exclude_columns_unknown_raises_keyerror(self):
        """Test unknown column names in exclude_columns raise KeyError"""
        df = pd.DataFrame({"col": [1]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)

        with pytest.raises(KeyError, match="Unknown exclude_columns"):
            report.to_dict(exclude_columns=["nonexistent"])

    def test_to_dict_redact_sample_values_propagates(self):
        """Test redact_sample_values parameter propagates to columns"""
        df = pd.DataFrame({"text": ["hello", "world"]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame, sample_size=2)

        result = report.to_dict(redact_sample_values=True)

        if result["columns"]["text"]["sample_values"]:
            assert all(v == "[REDACTED]" for v in result["columns"]["text"]["sample_values"])


class TestDataQualityReportToJson:
    """Tests for DataQualityReport.to_json"""

    def test_to_json_returns_valid_json_string(self):
        """Test to_json returns valid JSON when no output is provided"""
        df = pd.DataFrame({"col": [1, 2, 3]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)

        result = report.to_json()

        assert isinstance(result, str)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)
        assert "row_count" in parsed

    def test_to_json_with_output_stream_returns_none(self):
        """Test to_json writes to stream and returns None when output is provided"""
        df = pd.DataFrame({"col": [1, 2]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)

        output = io.StringIO()
        result = report.to_json(output=output)

        assert result is None
        output.seek(0)
        parsed = json.loads(output.read())
        assert isinstance(parsed, dict)

    def test_to_json_with_invalid_output_raises_typeerror(self):
        """Test to_json raises TypeError for non-writable output"""
        df = pd.DataFrame({"col": [1]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)

        with pytest.raises(TypeError, match="output must be a writable text stream"):
            report.to_json(output="not_a_stream")

    def test_to_json_respects_exclude_columns(self):
        """Test exclude_columns parameter filters JSON output"""
        df = pd.DataFrame({"a": [1], "b": [2]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)

        result = report.to_json(exclude_columns=["a"])

        parsed = json.loads(result)
        assert "a" not in parsed["columns"]
        assert "b" in parsed["columns"]

    def test_to_json_respects_redact_sample_values(self):
        """Test redact_sample_values parameter redacts in JSON output"""
        df = pd.DataFrame({"secret": ["data1", "data2"]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame, sample_size=2)

        result = report.to_json(redact_sample_values=True)

        parsed = json.loads(result)
        if parsed["columns"]["secret"]["sample_values"]:
            assert all(v == "[REDACTED]" for v in parsed["columns"]["secret"]["sample_values"])
