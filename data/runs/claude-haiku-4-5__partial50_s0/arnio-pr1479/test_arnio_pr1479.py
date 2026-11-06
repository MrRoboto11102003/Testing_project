import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestDataQualityReportToDict:
    """Test suite for arnio.quality.DataQualityReport.to_dict() method."""
    
    @pytest.fixture
    def test_frame(self):
        """Create a test DataFrame with mixed data types and nulls."""
        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", None, "David", "Eve"],
            "score": [95.5, 88.0, None, 92.5, 87.0],
            "active": [True, False, True, None, True],
        })
        return ar.from_pandas(df)
    
    @pytest.fixture
    def report(self, test_frame):
        """Generate a DataQualityReport from the test frame."""
        return ar.profile(test_frame)
    
    def test_to_dict_returns_dict(self, report):
        """to_dict() should return a dictionary."""
        result = report.to_dict()
        assert isinstance(result, dict)
    
    def test_to_dict_default_parameters(self, report):
        """to_dict() with no parameters should work."""
        result = report.to_dict()
        assert isinstance(result, dict)
        assert len(result) > 0
    
    def test_to_dict_redact_sample_values_false(self, report):
        """Test with redact_sample_values=False (includes sample values)."""
        result = report.to_dict(redact_sample_values=False)
        assert isinstance(result, dict)
    
    def test_to_dict_redact_sample_values_true(self, report):
        """Test with redact_sample_values=True (redacts sample values)."""
        result = report.to_dict(redact_sample_values=True)
        assert isinstance(result, dict)
    
    def test_to_dict_exclude_columns_list(self, report):
        """exclude_columns accepts list of column names."""
        result = report.to_dict(exclude_columns=["id", "name"])
        assert isinstance(result, dict)
    
    def test_to_dict_exclude_columns_set(self, report):
        """exclude_columns accepts set of column names."""
        result = report.to_dict(exclude_columns={"id", "name"})
        assert isinstance(result, dict)
    
    def test_to_dict_exclude_columns_tuple(self, report):
        """exclude_columns accepts tuple of column names."""
        result = report.to_dict(exclude_columns=("id", "name"))
        assert isinstance(result, dict)
    
    def test_to_dict_exclude_columns_none(self, report):
        """exclude_columns=None means no exclusion (default)."""
        result = report.to_dict(exclude_columns=None)
        assert isinstance(result, dict)
    
    def test_to_dict_exclude_columns_empty(self, report):
        """exclude_columns=[] means no exclusion."""
        result = report.to_dict(exclude_columns=[])
        assert isinstance(result, dict)
    
    def test_to_dict_exclude_nonexistent_column(self, report):
        """Excluding nonexistent columns should not raise."""
        result = report.to_dict(exclude_columns=["nonexistent_col"])
        assert isinstance(result, dict)
    
    def test_to_dict_both_parameters_combined(self, report):
        """Test with both redact_sample_values=True and exclude_columns."""
        result = report.to_dict(
            redact_sample_values=True,
            exclude_columns=["id"]
        )
        assert isinstance(result, dict)
    
    def test_to_dict_empty_frame(self):
        """to_dict() works on a report from an empty frame."""
        df = pd.DataFrame({"col1": [], "col2": []})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        result = report.to_dict()
        assert isinstance(result, dict)
    
    def test_to_dict_frame_with_all_nulls(self):
        """to_dict() works on a frame with a fully null column."""
        df = pd.DataFrame({
            "all_null": [None, None, None],
            "some_data": [1, 2, 3],
        })
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        result = report.to_dict()
        assert isinstance(result, dict)
    
    def test_to_dict_single_column(self):
        """to_dict() works on a single-column frame."""
        df = pd.DataFrame({"only_col": [1, 2, 3, 4, 5]})
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        result = report.to_dict()
        assert isinstance(result, dict)
    
    def test_to_dict_many_columns(self):
        """to_dict() works on a frame with many columns."""
        data = {f"col_{i}": range(10) for i in range(20)}
        df = pd.DataFrame(data)
        frame = ar.from_pandas(df)
        report = ar.profile(frame)
        result = report.to_dict()
        assert isinstance(result, dict)
    
    def test_to_dict_consistency(self, report):
        """Multiple calls to to_dict() produce consistent structure."""
        result1 = report.to_dict()
        result2 = report.to_dict()
        assert result1.keys() == result2.keys()
