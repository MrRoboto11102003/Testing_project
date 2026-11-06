import pytest
import pandas as pd
import numpy as np
import arnio as ar
import json


class TestDataQualityReportToDict:
    """Test DataQualityReport.to_dict() method."""

    @pytest.fixture
    def sample_frame(self):
        """Create a sample ArFrame for testing."""
        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "score": [85.5, 90.0, 88.2, None, 92.1],
            "active": [True, False, True, True, None],
        })
        return ar.from_pandas(df)

    @pytest.fixture
    def report(self, sample_frame):
        """Generate a DataQualityReport from sample frame."""
        return ar.profile(sample_frame)

    def test_to_dict_returns_dict(self, report):
        """to_dict() should return a dictionary."""
        result = report.to_dict()
        assert isinstance(result, dict)

    def test_to_dict_is_json_serializable(self, report):
        """to_dict() output should be JSON-serializable."""
        result = report.to_dict()
        json_str = json.dumps(result)
        assert isinstance(json_str, str)

    def test_to_dict_contains_aggregate_metrics(self, report):
        """to_dict() should include aggregate metrics like row_count, quality_score."""
        result = report.to_dict()
        assert "row_count" in result
        assert "column_count" in result
        assert "quality_score" in result
        assert isinstance(result["row_count"], int)
        assert isinstance(result["quality_score"], (int, float))

    def test_to_dict_contains_columns_dict(self, report):
        """to_dict() should include columns dictionary with column profiles."""
        result = report.to_dict()
        assert "columns" in result
        assert isinstance(result["columns"], dict)
        assert len(result["columns"]) > 0

    def test_to_dict_no_redaction_by_default(self, report):
        """By default, sample_values should not be redacted."""
        result = report.to_dict(redact_sample_values=False)
        for col_profile in result["columns"].values():
            if "sample_values" in col_profile and col_profile["sample_values"]:
                # At least some values should not be the redaction marker
                sample_values = col_profile["sample_values"]
                assert not all(v == "[REDACTED]" for v in sample_values)

    def test_to_dict_with_redact_sample_values_true(self, report):
        """With redact_sample_values=True, sample_values should be replaced."""
        result = report.to_dict(redact_sample_values=True)
        for col_profile in result["columns"].values():
            if "sample_values" in col_profile and col_profile["sample_values"]:
                sample_values = col_profile["sample_values"]
                assert all(v == "[REDACTED]" for v in sample_values)

    def test_to_dict_redaction_preserves_list_length(self, report):
        """Redaction should preserve sample_values list length."""
        normal = report.to_dict(redact_sample_values=False)
        redacted = report.to_dict(redact_sample_values=True)
        
        for col_name in normal["columns"]:
            if "sample_values" in normal["columns"][col_name]:
                normal_len = len(normal["columns"][col_name]["sample_values"])
                redacted_len = len(redacted["columns"][col_name]["sample_values"])
                assert normal_len == redacted_len

    def test_to_dict_redaction_of_top_values(self, report):
        """With redact_sample_values=True, top_values.value should be redacted."""
        result = report.to_dict(redact_sample_values=True)
        for col_profile in result["columns"].values():
            if "top_values" in col_profile:
                for item in col_profile["top_values"]:
                    if isinstance(item, dict):
                        assert item.get("value") == "[REDACTED]"
                        # Counts and ratios should still be present
                        assert ("count" in item) or ("ratio" in item)

    def test_to_dict_exclude_columns_single(self, report):
        """exclude_columns with single column should remove it."""
        result = report.to_dict(exclude_columns=["score"])
        assert "score" not in result["columns"]
        assert "id" in result["columns"]

    def test_to_dict_exclude_columns_multiple(self, report):
        """exclude_columns should accept multiple columns as list."""
        result = report.to_dict(exclude_columns=["score", "active"])
        assert "score" not in result["columns"]
        assert "active" not in result["columns"]
        assert "id" in result["columns"]
        assert "name" in result["columns"]

    def test_to_dict_exclude_columns_as_set(self, report):
        """exclude_columns should accept a set."""
        result = report.to_dict(exclude_columns={"score", "active"})
        assert "score" not in result["columns"]
        assert "active" not in result["columns"]

    def test_to_dict_exclude_columns_as_tuple(self, report):
        """exclude_columns should accept a tuple."""
        result = report.to_dict(exclude_columns=("score", "active"))
        assert "score" not in result["columns"]
        assert "active" not in result["columns"]

    def test_to_dict_combine_redaction_and_exclusion(self, report):
        """Redaction and exclusion should work together."""
        result = report.to_dict(
            redact_sample_values=True,
            exclude_columns=["score"]
        )
        # Column should be excluded
        assert "score" not in result["columns"]
        # Remaining columns should have redacted samples
        for col_profile in result["columns"].values():
            if "sample_values" in col_profile and col_profile["sample_values"]:
                assert all(v == "[REDACTED]" for v in col_profile["sample_values"])

    def test_to_dict_empty_exclude_columns_list(self, report):
        """Empty exclude_columns should include all columns."""
        result_with_empty = report.to_dict(exclude_columns=[])
        result_with_none = report.to_dict(exclude_columns=None)
        assert set(result_with_empty["columns"].keys()) == set(result_with_none["columns"].keys())

    def test_to_dict_exclude_nonexistent_column_no_error(self, report):
        """Excluding non-existent column should not raise error."""
        # Should handle gracefully
        result = report.to_dict(exclude_columns=["nonexistent_column"])
        assert "nonexistent_column" not in result["columns"]
        # Other columns should still be present
        assert len(result["columns"]) > 0

    def test_to_dict_redaction_counts_preserved(self, report):
        """Redaction should preserve count/ratio information."""
        normal = report.to_dict(redact_sample_values=False)
        redacted = report.to_dict(redact_sample_values=True)
        
        for col_name in normal["columns"]:
            if "top_values" in normal["columns"][col_name]:
                normal_top = normal["columns"][col_name]["top_values"]
                redacted_top = redacted["columns"][col_name]["top_values"]
                
                # Length should be preserved
                assert len(normal_top) == len(redacted_top)
                
                # Counts and ratios should be identical
                for normal_item, redacted_item in zip(normal_top, redacted_top):
                    if isinstance(normal_item, dict):
                        if "count" in normal_item:
                            assert normal_item["count"] == redacted_item.get("count")
                        if "ratio" in normal_item:
                            assert normal_item["ratio"] == redacted_item.get("ratio")
