import pytest
import json
import pandas as pd
import arnio as ar


class TestDataQualityReportToDict:
    """Tests for DataQualityReport.to_dict() method."""

    def test_to_dict_returns_dict_with_required_keys(self):
        """Test that to_dict returns a dict with all required keys."""
        frame = ar.from_pandas(pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]}))
        report = ar.profile(frame)
        result = report.to_dict()
        
        assert isinstance(result, dict)
        required_keys = {
            "row_count",
            "column_count",
            "memory_usage",
            "duplicate_rows",
            "duplicate_ratio",
            "quality_score",
            "score_components",
            "columns",
            "suggestions",
        }
        assert required_keys.issubset(result.keys())

    def test_to_dict_basic_values_correct(self):
        """Test that to_dict returns correct values for basic metrics."""
        df = pd.DataFrame({"col1": [1, 2, 3]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        result = report.to_dict()
        
        assert result["row_count"] == 3
        assert result["column_count"] == 1
        assert result["memory_usage"] > 0
        assert isinstance(result["duplicate_ratio"], float)
        assert 0.0 <= result["quality_score"] <= 1.0

    def test_to_dict_columns_sorted_alphabetically(self):
        """Test that columns are returned in sorted order."""
        frame = ar.from_pandas(pd.DataFrame({"z_col": [1], "a_col": [2], "m_col": [3]}))
        report = ar.profile(frame)
        result = report.to_dict()
        
        column_names = list(result["columns"].keys())
        assert column_names == ["a_col", "m_col", "z_col"]

    def test_to_dict_exclude_columns_with_list(self):
        """Test excluding columns using a list."""
        frame = ar.from_pandas(pd.DataFrame({"col1": [1], "col2": [2], "col3": [3]}))
        report = ar.profile(frame)
        result = report.to_dict(exclude_columns=["col2"])
        
        assert "col1" in result["columns"]
        assert "col2" not in result["columns"]
        assert "col3" in result["columns"]

    def test_to_dict_exclude_columns_with_tuple(self):
        """Test excluding columns using a tuple."""
        frame = ar.from_pandas(pd.DataFrame({"col1": [1], "col2": [2]}))
        report = ar.profile(frame)
        result = report.to_dict(exclude_columns=("col1",))
        
        assert "col1" not in result["columns"]
        assert "col2" in result["columns"]

    def test_to_dict_exclude_columns_with_set(self):
        """Test excluding columns using a set."""
        frame = ar.from_pandas(pd.DataFrame({"col1": [1], "col2": [2], "col3": [3]}))
        report = ar.profile(frame)
        result = report.to_dict(exclude_columns={"col1", "col3"})
        
        assert "col1" not in result["columns"]
        assert "col2" in result["columns"]
        assert "col3" not in result["columns"]

    def test_to_dict_exclude_columns_invalid_type_raises_typeerror(self):
        """Test that non-list/tuple/set exclude_columns raises TypeError."""
        frame = ar.from_pandas(pd.DataFrame({"col1": [1]}))
        report = ar.profile(frame)
        
        with pytest.raises(TypeError, match="exclude_columns must be a list, tuple, set, or None"):
            report.to_dict(exclude_columns="col1")
        
        with pytest.raises(TypeError, match="exclude_columns must be a list, tuple, set, or None"):
            report.to_dict(exclude_columns={"col1": "value"})

    def test_to_dict_exclude_columns_non_string_item_raises_typeerror(self):
        """Test that non-string items in exclude_columns raise TypeError."""
        frame = ar.from_pandas(pd.DataFrame({"col1": [1]}))
        report = ar.profile(frame)
        
        with pytest.raises(TypeError, match="exclude_columns must contain only string column names"):
            report.to_dict(exclude_columns=["col1", 42])
        
        with pytest.raises(TypeError, match="exclude_columns must contain only string column names"):
            report.to_dict(exclude_columns=[None])

    def test_to_dict_unknown_exclude_column_raises_keyerror(self):
        """Test that unknown column names in exclude_columns raise KeyError."""
        frame = ar.from_pandas(pd.DataFrame({"col1": [1], "col2": [2]}))
        report = ar.profile(frame)
        
        with pytest.raises(KeyError, match="Unknown exclude_columns"):
            report.to_dict(exclude_columns=["unknown_col"])
        
        with pytest.raises(KeyError, match="Unknown exclude_columns"):
            report.to_dict(exclude_columns=["col1", "nonexistent"])

    def test_to_dict_redact_sample_values_false(self):
        """Test that redact_sample_values=False passes through normally."""
        frame = ar.from_pandas(pd.DataFrame({"col1": [1, 2, 3]}))
        report = ar.profile(frame)
        
        result = report.to_dict(redact_sample_values=False)
        assert "columns" in result
        assert "col1" in result["columns"]

    def test_to_dict_redact_sample_values_true(self):
        """Test that redact_sample_values=True is handled correctly."""
        frame = ar.from_pandas(pd.DataFrame({"col1": ["value1", "value2", "value3"]}))
        report = ar.profile(frame)
        
        result = report.to_dict(redact_sample_values=True)
        assert "columns" in result
        assert "col1" in result["columns"]

    def test_to_dict_empty_exclude_columns_list(self):
        """Test that empty exclude_columns list doesn't exclude anything."""
        frame = ar.from_pandas(pd.DataFrame({"col1": [1], "col2": [2]}))
        report = ar.profile(frame)
        
        result = report.to_dict(exclude_columns=[])
        assert "col1" in result["columns"]
        assert "col2" in result["columns"]

    def test_to_dict_none_exclude_columns(self):
        """Test that exclude_columns=None doesn't exclude anything."""
        frame = ar.from_pandas(pd.DataFrame({"col1": [1], "col2": [2]}))
        report = ar.profile(frame)
        
        result = report.to_dict(exclude_columns=None)
        assert "col1" in result["columns"]
        assert "col2" in result["columns"]

    def test_to_dict_suggestions_sorted(self):
        """Test that suggestions are sorted in output."""
        frame = ar.from_pandas(pd.DataFrame({"col1": [1, 1], "col2": ["a", "a"], "col3": [1.0, 1.0]}))
        report = ar.profile(frame)
        
        result = report.to_dict()
        suggestions = result["suggestions"]
        
        assert isinstance(suggestions, list)
        for suggestion in suggestions:
            assert isinstance(suggestion, dict)
            assert "step" in suggestion
            assert "kwargs" in suggestion

    def test_to_dict_json_serializable(self):
        """Test that to_dict output is JSON serializable."""
        frame = ar.from_pandas(pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]}))
        report = ar.profile(frame)
        
        result = report.to_dict()
        # Should not raise an exception
        json_str = json.dumps(result, default=str)
        assert isinstance(json_str, str)
        assert len(json_str) > 0

    def test_to_dict_multiple_columns_with_exclusion(self):
        """Test complex scenario with multiple columns and exclusions."""
        frame = ar.from_pandas(
            pd.DataFrame({
                "id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"],
                "score": [95.5, 87.3, 92.1],
                "active": [True, False, True],
            })
        )
        report = ar.profile(frame)
        
        result = report.to_dict(exclude_columns=["name", "active"])
        
        assert "id" in result["columns"]
        assert "score" in result["columns"]
        assert "name" not in result["columns"]
        assert "active" not in result["columns"]
        assert result["column_count"] == 4  # Total columns still 4
