import pytest
import pandas as pd
import arnio as ar
import json


@pytest.fixture
def sample_df():
    """Create a sample pandas DataFrame."""
    return pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie", None, "Eve"],
        "age": [25, 30, 35, 40, 45],
        "salary": [50000.0, 60000.0, None, 70000.0, 80000.0],
        "city": ["NYC", "LA", "NYC", "Chicago", "Boston"],
    })


@pytest.fixture
def quality_report(sample_df):
    """Create a DataQualityReport from sample data."""
    frame = ar.from_dict(sample_df.to_dict('list'))
    return frame.profile()


class TestDataQualityReportToDict:
    """Tests for DataQualityReport.to_dict() method."""

    def test_returns_dict(self, quality_report):
        """Test that to_dict returns a dictionary."""
        result = quality_report.to_dict()
        assert isinstance(result, dict)

    def test_json_serializable(self, quality_report):
        """Test that output is JSON-serializable."""
        result = quality_report.to_dict()
        json.dumps(result)  # Should not raise

    def test_default_call(self, quality_report):
        """Test to_dict with no parameters uses defaults."""
        result = quality_report.to_dict()
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_redact_sample_values_false(self, quality_report):
        """Test with redact_sample_values=False."""
        result = quality_report.to_dict(redact_sample_values=False)
        assert isinstance(result, dict)

    def test_redact_sample_values_true(self, quality_report):
        """Test with redact_sample_values=True."""
        result = quality_report.to_dict(redact_sample_values=True)
        assert isinstance(result, dict)

    def test_exclude_columns_list(self, quality_report):
        """Test exclude_columns with list."""
        result = quality_report.to_dict(exclude_columns=["name", "age"])
        assert isinstance(result, dict)

    def test_exclude_columns_set(self, quality_report):
        """Test exclude_columns with set."""
        result = quality_report.to_dict(exclude_columns={"salary", "city"})
        assert isinstance(result, dict)

    def test_exclude_columns_tuple(self, quality_report):
        """Test exclude_columns with tuple."""
        result = quality_report.to_dict(exclude_columns=("name", "salary"))
        assert isinstance(result, dict)

    def test_exclude_columns_none(self, quality_report):
        """Test exclude_columns with None."""
        result = quality_report.to_dict(exclude_columns=None)
        assert isinstance(result, dict)

    def test_exclude_columns_empty(self, quality_report):
        """Test exclude_columns with empty collection."""
        result = quality_report.to_dict(exclude_columns=[])
        assert isinstance(result, dict)

    def test_combined_parameters(self, quality_report):
        """Test with both redact_sample_values and exclude_columns."""
        result = quality_report.to_dict(
            redact_sample_values=True,
            exclude_columns=["name"]
        )
        assert isinstance(result, dict)

    def test_multiple_calls_consistent(self, quality_report):
        """Test that multiple calls return consistent results."""
        result1 = quality_report.to_dict()
        result2 = quality_report.to_dict()
        assert result1 == result2

    def test_redact_produces_valid_dict(self, quality_report):
        """Test that redacted output is still a valid dict."""
        result_redacted = quality_report.to_dict(redact_sample_values=True)
        result_normal = quality_report.to_dict(redact_sample_values=False)
        assert isinstance(result_redacted, dict)
        assert isinstance(result_normal, dict)
        json.dumps(result_redacted)
        json.dumps(result_normal)

    def test_with_sparse_data(self):
        """Test with DataFrame containing mostly null values."""
        data = pd.DataFrame({
            "col1": [1, None, None, None, None],
            "col2": [None, None, 2, None, None],
            "col3": [None, None, None, None, 3],
        })
        frame = ar.from_dict(data.to_dict('list'))
        report = frame.profile()
        result = report.to_dict()
        assert isinstance(result, dict)
        json.dumps(result)

    def test_with_diverse_types(self):
        """Test with DataFrame containing multiple data types."""
        data = pd.DataFrame({
            "ints": [1, 2, 3, 4, 5],
            "floats": [1.1, 2.2, 3.3, 4.4, 5.5],
            "strings": ["a", "b", "c", "d", "e"],
            "mixed_nulls": [1, None, 3, None, 5],
        })
        frame = ar.from_dict(data.to_dict('list'))
        report = frame.profile()
        result = report.to_dict(exclude_columns=["ints"])
        assert isinstance(result, dict)
        json.dumps(result)
