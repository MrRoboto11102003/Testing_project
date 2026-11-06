import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestDataQualityReportToDict:
    """Test suite for DataQualityReport.to_dict() method."""

    @pytest.fixture
    def sample_frame(self):
        """Create a sample frame for testing."""
        df = pd.DataFrame({
            "id": [1, 2, 3, None],
            "name": ["Alice", "Bob", "Charlie", "Diana"],
            "score": [95.5, 87.3, None, 92.1],
            "active": [True, False, True, None],
        })
        return ar.from_pandas(df)

    @pytest.fixture
    def sample_report(self, sample_frame):
        """Create a data quality report."""
        return ar.profile(sample_frame)

    def test_to_dict_returns_dict(self, sample_report):
        """Test that to_dict returns a dictionary."""
        result = sample_report.to_dict()
        assert isinstance(result, dict)

    def test_to_dict_has_columns_key(self, sample_report):
        """Test that returned dict includes 'columns' key."""
        result = sample_report.to_dict()
        assert "columns" in result
        assert isinstance(result["columns"], dict)

    def test_to_dict_includes_all_columns_by_default(self, sample_report):
        """Test that all columns are included by default."""
        result = sample_report.to_dict()
        expected_cols = {"id", "name", "score", "active"}
        assert set(result["columns"].keys()) == expected_cols

    def test_to_dict_redact_sample_values_false(self, sample_report):
        """Test that sample values are preserved when redact_sample_values=False."""
        result_unredacted = sample_report.to_dict(redact_sample_values=False)
        result_redacted = sample_report.to_dict(redact_sample_values=True)
        
        # Both should return dicts, but redacted should have hidden/masked sample values
        assert isinstance(result_unredacted, dict)
        assert isinstance(result_redacted, dict)

    def test_to_dict_exclude_columns_list(self, sample_report):
        """Test exclude_columns parameter with a list."""
        result = sample_report.to_dict(exclude_columns=["id", "score"])
        
        assert "id" not in result["columns"]
        assert "score" not in result["columns"]
        assert "name" in result["columns"]
        assert "active" in result["columns"]

    def test_to_dict_exclude_columns_set(self, sample_report):
        """Test exclude_columns parameter with a set."""
        result = sample_report.to_dict(exclude_columns={"id", "name"})
        
        assert "id" not in result["columns"]
        assert "name" not in result["columns"]
        assert "score" in result["columns"]
        assert "active" in result["columns"]

    def test_to_dict_exclude_columns_tuple(self, sample_report):
        """Test exclude_columns parameter with a tuple."""
        result = sample_report.to_dict(exclude_columns=("score", "active"))
        
        assert "score" not in result["columns"]
        assert "active" not in result["columns"]
        assert "id" in result["columns"]
        assert "name" in result["columns"]

    def test_to_dict_exclude_columns_empty_collection(self, sample_report):
        """Test exclude_columns with empty collection includes all columns."""
        result = sample_report.to_dict(exclude_columns=[])
        
        expected_cols = {"id", "name", "score", "active"}
        assert set(result["columns"].keys()) == expected_cols

    def test_to_dict_exclude_nonexistent_column(self, sample_report):
        """Test that excluding non-existent columns is handled gracefully."""
        result = sample_report.to_dict(exclude_columns=["nonexistent_col"])
        
        expected_cols = {"id", "name", "score", "active"}
        assert set(result["columns"].keys()) == expected_cols

    def test_to_dict_combined_parameters(self, sample_report):
        """Test using both redact_sample_values and exclude_columns together."""
        result = sample_report.to_dict(
            redact_sample_values=True,
            exclude_columns=["score", "active"]
        )
        
        assert "score" not in result["columns"]
        assert "active" not in result["columns"]
        assert "id" in result["columns"]
        assert "name" in result["columns"]

    def test_to_dict_keyword_only_enforcement(self, sample_report):
        """Test that parameters are keyword-only (not positional)."""
        with pytest.raises(TypeError):
            sample_report.to_dict(False)  # type: ignore

    def test_to_dict_exclude_all_columns(self, sample_report):
        """Test excluding all columns results in empty columns dict."""
        all_cols = ["id", "name", "score", "active"]
        result = sample_report.to_dict(exclude_columns=all_cols)
        
        assert result["columns"] == {}

    def test_to_dict_empty_dataframe(self):
        """Test to_dict with an empty DataFrame."""
        empty_df = pd.DataFrame()
        empty_frame = ar.from_pandas(empty_df)
        empty_report = ar.profile(empty_frame)
        
        result = empty_report.to_dict()
        assert isinstance(result, dict)
        assert result["columns"] == {}

    def test_to_dict_with_mixed_dtypes(self):
        """Test to_dict with various data types."""
        df = pd.DataFrame({
            "int_col": [1, 2, 3],
            "float_col": [1.5, 2.5, 3.5],
            "str_col": ["x", "y", "z"],
            "bool_col": [True, False, True],
            "mixed_null": [1.0, None, 3.0],
        })
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        
        result = report.to_dict()
        expected_cols = {"int_col", "float_col", "str_col", "bool_col", "mixed_null"}
        assert set(result["columns"].keys()) == expected_cols
