import pytest
import pandas as pd
import arnio as ar


class TestDataQualityReportToDict:
    """Tests for DataQualityReport.to_dict method"""

    @pytest.fixture
    def sample_report(self):
        """Create a sample DataQualityReport for testing"""
        df = pd.DataFrame({
            "col_a": ["x", "y", "z"],
            "col_b": [1, 2, 3],
            "col_c": [1.0, 2.0, 3.0],
        })
        return df.arnio.profile()

    def test_to_dict_returns_dict_with_all_keys(self, sample_report):
        """Test basic structure and required keys"""
        result = sample_report.to_dict()
        assert isinstance(result, dict)
        
        required_keys = {
            "row_count", "column_count", "memory_usage", "duplicate_rows",
            "duplicate_ratio", "quality_score", "score_components",
            "columns", "suggestions"
        }
        assert required_keys.issubset(result.keys())

    def test_to_dict_columns_contains_all_column_names(self, sample_report):
        """Test that columns dict includes all DataFrame columns"""
        result = sample_report.to_dict()
        assert set(result["columns"].keys()) == {"col_a", "col_b", "col_c"}

    def test_to_dict_exclude_columns_none_and_empty_list(self, sample_report):
        """Test that None and empty list both include all columns"""
        result_none = sample_report.to_dict(exclude_columns=None)
        result_empty = sample_report.to_dict(exclude_columns=[])
        
        assert len(result_none["columns"]) == 3
        assert len(result_empty["columns"]) == 3

    def test_to_dict_exclude_columns_list_type(self, sample_report):
        """Test exclude_columns with list"""
        result = sample_report.to_dict(exclude_columns=["col_a"])
        assert "col_a" not in result["columns"]
        assert "col_b" in result["columns"]
        assert "col_c" in result["columns"]

    def test_to_dict_exclude_columns_tuple_type(self, sample_report):
        """Test exclude_columns with tuple"""
        result = sample_report.to_dict(exclude_columns=("col_b", "col_c"))
        assert "col_a" in result["columns"]
        assert "col_b" not in result["columns"]
        assert "col_c" not in result["columns"]

    def test_to_dict_exclude_columns_set_type(self, sample_report):
        """Test exclude_columns with set"""
        result = sample_report.to_dict(exclude_columns={"col_a", "col_b"})
        assert "col_a" not in result["columns"]
        assert "col_b" not in result["columns"]
        assert "col_c" in result["columns"]

    def test_to_dict_exclude_all_columns(self, sample_report):
        """Test excluding all columns results in empty columns dict"""
        result = sample_report.to_dict(
            exclude_columns=["col_a", "col_b", "col_c"]
        )
        assert result["columns"] == {}

    def test_to_dict_exclude_columns_unknown_raises_keyerror(self, sample_report):
        """Test that unknown column in exclude_columns raises KeyError"""
        with pytest.raises(KeyError, match="Unknown exclude_columns"):
            sample_report.to_dict(exclude_columns=["unknown_col"])

    def test_to_dict_exclude_columns_wrong_type_raises_typeerror(self, sample_report):
        """Test that invalid type for exclude_columns raises TypeError"""
        with pytest.raises(TypeError, match="exclude_columns must be a list, tuple, set, or None"):
            sample_report.to_dict(exclude_columns="col_a")

    def test_to_dict_exclude_columns_non_string_raises_typeerror(self, sample_report):
        """Test that non-string values in exclude_columns raise TypeError"""
        with pytest.raises(TypeError, match="exclude_columns must contain only string column names"):
            sample_report.to_dict(exclude_columns=[1, 2, 3])

    def test_to_dict_redact_sample_values_true(self, sample_report):
        """Test to_dict with redact_sample_values=True"""
        result = sample_report.to_dict(redact_sample_values=True)
        assert isinstance(result, dict)
        assert set(result["columns"].keys()) == {"col_a", "col_b", "col_c"}

    def test_to_dict_redact_sample_values_false(self, sample_report):
        """Test to_dict with redact_sample_values=False (default)"""
        result = sample_report.to_dict(redact_sample_values=False)
        assert isinstance(result, dict)
        assert set(result["columns"].keys()) == {"col_a", "col_b", "col_c"}

    def test_to_dict_combined_exclude_and_redact(self, sample_report):
        """Test combining exclude_columns and redact_sample_values parameters"""
        result = sample_report.to_dict(
            exclude_columns=["col_b"],
            redact_sample_values=True
        )
        assert "col_b" not in result["columns"]
        assert "col_a" in result["columns"]
        assert "col_c" in result["columns"]

    def test_to_dict_suggestions_is_list(self, sample_report):
        """Test that suggestions is a list with expected structure"""
        result = sample_report.to_dict()
        assert isinstance(result["suggestions"], list)
        
        for suggestion in result["suggestions"]:
            assert isinstance(suggestion, dict)
            assert "step" in suggestion
            assert "kwargs" in suggestion
            assert "confidence_score" in suggestion
            assert "confidence_reason" in suggestion
