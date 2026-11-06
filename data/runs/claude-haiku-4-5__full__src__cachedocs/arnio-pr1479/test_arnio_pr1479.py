import pytest
import pandas as pd
import arnio as ar


class TestDataQualityReportToDict:
    """Tests for DataQualityReport.to_dict()"""

    @pytest.fixture
    def sample_frame(self):
        """Create a sample ArFrame for testing."""
        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "score": [85.5, 90.0, 88.2, None, 92.1],
            "email": ["a@x.com", "b@x.com", None, "d@x.com", "e@x.com"],
        })
        return ar.from_pandas(df)

    @pytest.fixture
    def report(self, sample_frame):
        """Create a DataQualityReport for testing."""
        return ar.profile(sample_frame)

    def test_to_dict_basic_structure(self, report):
        """Test that to_dict returns all required keys."""
        result = report.to_dict()
        required_keys = {
            "row_count", "column_count", "memory_usage",
            "duplicate_rows", "duplicate_ratio", "quality_score",
            "score_components", "columns", "suggestions"
        }
        assert required_keys.issubset(result.keys())
        assert isinstance(result["columns"], dict)
        assert isinstance(result["suggestions"], list)

    def test_to_dict_default_includes_all_columns(self, report):
        """Test default parameters include all columns."""
        result = report.to_dict()
        assert len(result["columns"]) == 4
        assert all(col in result["columns"] for col in ["id", "name", "score", "email"])

    def test_to_dict_columns_are_sorted(self, report):
        """Test that columns in result are sorted alphabetically by name."""
        result = report.to_dict()
        column_names = list(result["columns"].keys())
        assert column_names == sorted(column_names)

    def test_to_dict_exclude_columns_as_list(self, report):
        """Test exclude_columns with list removes specified columns."""
        result = report.to_dict(exclude_columns=["id", "name"])
        assert "id" not in result["columns"]
        assert "name" not in result["columns"]
        assert "score" in result["columns"]
        assert "email" in result["columns"]

    def test_to_dict_exclude_columns_as_tuple(self, report):
        """Test exclude_columns with tuple removes specified columns."""
        result = report.to_dict(exclude_columns=("score", "email"))
        assert len(result["columns"]) == 2
        assert "score" not in result["columns"]
        assert "email" not in result["columns"]

    def test_to_dict_exclude_columns_as_set(self, report):
        """Test exclude_columns with set removes specified columns."""
        result = report.to_dict(exclude_columns={"id"})
        assert "id" not in result["columns"]
        assert len(result["columns"]) == 3

    def test_to_dict_exclude_columns_none_or_empty(self, report):
        """Test exclude_columns=None and empty set both include all columns."""
        result_none = report.to_dict(exclude_columns=None)
        result_empty = report.to_dict(exclude_columns=set())
        assert len(result_none["columns"]) == 4
        assert len(result_empty["columns"]) == 4

    def test_to_dict_exclude_columns_invalid_type(self, report):
        """Test exclude_columns with invalid type raises TypeError."""
        with pytest.raises(TypeError, match="exclude_columns must be a list, tuple, set, or None"):
            report.to_dict(exclude_columns="id")
        
        with pytest.raises(TypeError, match="exclude_columns must be a list, tuple, set, or None"):
            report.to_dict(exclude_columns=123)

    def test_to_dict_exclude_columns_non_string_value(self, report):
        """Test exclude_columns with non-string values raises TypeError."""
        with pytest.raises(TypeError, match="exclude_columns must contain only string column names"):
            report.to_dict(exclude_columns=["id", 123])
        
        with pytest.raises(TypeError, match="exclude_columns must contain only string column names"):
            report.to_dict(exclude_columns=[None])

    def test_to_dict_exclude_columns_unknown_column(self, report):
        """Test exclude_columns with unknown column raises KeyError."""
        with pytest.raises(KeyError, match="Unknown exclude_columns"):
            report.to_dict(exclude_columns=["unknown_column"])
        
        with pytest.raises(KeyError, match="Unknown exclude_columns.*Available columns"):
            report.to_dict(exclude_columns=["id", "nonexistent"])

    def test_to_dict_redact_sample_values_true(self, report):
        """Test redact_sample_values=True is accepted and returns valid dict."""
        result = report.to_dict(redact_sample_values=True)
        assert isinstance(result, dict)
        assert "columns" in result
        # Structure should be valid
        assert all(isinstance(v, dict) for v in result["columns"].values())

    def test_to_dict_redact_and_exclude_combined(self, report):
        """Test redact_sample_values and exclude_columns work together."""
        result = report.to_dict(redact_sample_values=True, exclude_columns=["id", "score"])
        assert len(result["columns"]) == 2
        assert "id" not in result["columns"]
        assert "score" not in result["columns"]
        assert "name" in result["columns"]
        assert "email" in result["columns"]

    def test_to_dict_suggestions_structure(self, report):
        """Test that suggestions have the expected structure."""
        result = report.to_dict()
        for suggestion in result["suggestions"]:
            assert isinstance(suggestion, dict)
            assert "step" in suggestion
            assert "kwargs" in suggestion
            assert isinstance(suggestion["step"], str)
            assert isinstance(suggestion["kwargs"], dict)

    def test_to_dict_quality_metrics_valid(self, report):
        """Test that quality metrics are valid numeric values."""
        result = report.to_dict()
        assert isinstance(result["row_count"], int)
        assert result["row_count"] > 0
        assert isinstance(result["quality_score"], (int, float))
        assert 0.0 <= result["quality_score"] <= 100.0
        assert isinstance(result["duplicate_ratio"], (int, float))
        assert 0.0 <= result["duplicate_ratio"] <= 1.0
