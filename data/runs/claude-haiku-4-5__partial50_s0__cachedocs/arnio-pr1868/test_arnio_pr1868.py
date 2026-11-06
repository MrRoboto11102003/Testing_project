import pytest
import pandas as pd
import io
import json
import arnio as ar


@pytest.fixture
def sample_frame():
    """Create a sample ArFrame with various data types and null values."""
    data = {
        "id": [1, 2, 3, None, 5],
        "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        "score": [95.5, 87.0, 92.3, None, 88.9],
        "active": [True, False, True, True, None],
    }
    df = pd.DataFrame(data)
    return ar.from_pandas(df)


@pytest.fixture
def quality_report(sample_frame):
    """Create a DataQualityReport from sample frame."""
    return ar.profile(sample_frame)


class TestColumnProfileToDict:
    """Tests for ColumnProfile.to_dict()"""

    def test_to_dict_returns_dict(self, quality_report):
        """to_dict should return a dictionary."""
        col_profile = list(quality_report.columns.values())[0]
        result = col_profile.to_dict()
        assert isinstance(result, dict)

    def test_to_dict_with_redaction_true(self, quality_report):
        """to_dict with redact_sample_values=True should be JSON-friendly."""
        col_profile = list(quality_report.columns.values())[0]
        result = col_profile.to_dict(redact_sample_values=True)
        assert isinstance(result, dict)
        # Should be serializable to JSON
        json_str = json.dumps(result, default=str)
        assert isinstance(json_str, str)

    def test_to_dict_with_redaction_false(self, quality_report):
        """to_dict with redact_sample_values=False includes samples."""
        col_profile = list(quality_report.columns.values())[0]
        result = col_profile.to_dict(redact_sample_values=False)
        assert isinstance(result, dict)
        json_str = json.dumps(result, default=str)
        assert isinstance(json_str, str)


class TestDataQualityReportScoreBreakdown:
    """Tests for DataQualityReport.score_breakdown()"""

    def test_score_breakdown_returns_dict_of_floats(self, quality_report):
        """score_breakdown should return dict with float values."""
        result = quality_report.score_breakdown()
        assert isinstance(result, dict)
        assert len(result) > 0
        for value in result.values():
            assert isinstance(value, (int, float))

    def test_score_breakdown_values_in_valid_range(self, quality_report):
        """score_breakdown values should be non-negative."""
        result = quality_report.score_breakdown()
        for key, value in result.items():
            assert value >= 0, f"Score '{key}' has invalid value: {value}"

    def test_score_breakdown_consistent_calls(self, quality_report):
        """Multiple calls to score_breakdown should be consistent."""
        result1 = quality_report.score_breakdown()
        result2 = quality_report.score_breakdown()
        assert result1 == result2


class TestDataQualityReportToDict:
    """Tests for DataQualityReport.to_dict()"""

    def test_to_dict_returns_dict(self, quality_report):
        """to_dict should return a dictionary."""
        result = quality_report.to_dict()
        assert isinstance(result, dict)

    def test_to_dict_with_redaction(self, quality_report):
        """to_dict with redact_sample_values=True should still be valid."""
        result = quality_report.to_dict(redact_sample_values=True)
        assert isinstance(result, dict)
        # Should be JSON serializable
        json.dumps(result, default=str)

    def test_to_dict_exclude_single_column(self, quality_report):
        """to_dict should exclude specified columns."""
        result_excluded = quality_report.to_dict(exclude_columns=["name"])
        result_full = quality_report.to_dict()
        
        excluded_count = len(result_excluded.get("columns", {}))
        full_count = len(result_full.get("columns", {}))
        assert excluded_count < full_count

    def test_to_dict_exclude_multiple_columns(self, quality_report):
        """to_dict should handle multiple excluded columns."""
        result = quality_report.to_dict(exclude_columns=["name", "score"])
        assert isinstance(result, dict)
        assert len(result.get("columns", {})) <= 2

    def test_to_dict_exclude_with_set_and_tuple(self, quality_report):
        """to_dict should accept set and tuple for exclude_columns."""
        result_set = quality_report.to_dict(exclude_columns={"name"})
        result_tuple = quality_report.to_dict(exclude_columns=("name",))
        result_list = quality_report.to_dict(exclude_columns=["name"])
        
        assert isinstance(result_set, dict)
        assert isinstance(result_tuple, dict)
        assert isinstance(result_list, dict)


class TestDataQualityReportToJson:
    """Tests for DataQualityReport.to_json()"""

    def test_to_json_returns_valid_json_string(self, quality_report):
        """to_json should return valid JSON string."""
        result = quality_report.to_json()
        assert isinstance(result, str)
        data = json.loads(result)
        assert isinstance(data, dict)

    def test_to_json_with_indent(self, quality_report):
        """to_json with indent should produce formatted output."""
        result = quality_report.to_json(indent=2)
        assert isinstance(result, str)
        assert "\n" in result
        data = json.loads(result)
        assert isinstance(data, dict)

    def test_to_json_with_redaction_and_exclusion(self, quality_report):
        """to_json should handle redaction and exclusion together."""
        result = quality_report.to_json(
            indent=2,
            redact_sample_values=True,
            exclude_columns=["score"]
        )
        assert isinstance(result, str)
        data = json.loads(result)
        assert isinstance(data, dict)

    def test_to_json_with_output_file(self, quality_frame):
        """to_json with output parameter should write to file object."""
        output = io.StringIO()
        result = quality_report.to_json(output=output)
        
        # When output provided, may return None or the string
        if result is None:
            content = output.getvalue()
            assert isinstance(content, str)
            json.loads(content)
        else:
            assert isinstance(result, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
