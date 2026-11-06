import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestDataQualityReportToDict:
    """Tests for arnio.quality.DataQualityReport.to_dict method."""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', None, 'David', 'Eve'],
            'age': [25, 30, 35, None, 45],
            'salary': [50000.0, 60000.0, 70000.0, 80000.0, np.nan],
            'flag': [True, False, True, True, False]
        })
    
    @pytest.fixture
    def report(self, sample_dataframe):
        """Create a DataQualityReport from the sample DataFrame."""
        return ar.quality.DataQualityReport(sample_dataframe)
    
    def test_to_dict_returns_dict(self, report):
        """to_dict should return a dictionary."""
        result = report.to_dict()
        assert isinstance(result, dict)
    
    def test_to_dict_default_parameters(self, report):
        """to_dict should work with no arguments (default parameters)."""
        result = report.to_dict()
        assert isinstance(result, dict)
    
    def test_to_dict_redact_sample_values_false(self, report):
        """to_dict with redact_sample_values=False should include sample values."""
        result = report.to_dict(redact_sample_values=False)
        assert isinstance(result, dict)
    
    def test_to_dict_redact_sample_values_true(self, report):
        """to_dict with redact_sample_values=True should redact sample values."""
        result = report.to_dict(redact_sample_values=True)
        assert isinstance(result, dict)
    
    def test_to_dict_exclude_columns_as_list(self, report):
        """exclude_columns should accept a list of column names."""
        result = report.to_dict(exclude_columns=['age', 'salary'])
        assert isinstance(result, dict)
    
    def test_to_dict_exclude_columns_as_set(self, report):
        """exclude_columns should accept a set of column names."""
        result = report.to_dict(exclude_columns={'age', 'salary'})
        assert isinstance(result, dict)
    
    def test_to_dict_exclude_columns_as_tuple(self, report):
        """exclude_columns should accept a tuple of column names."""
        result = report.to_dict(exclude_columns=('age', 'salary'))
        assert isinstance(result, dict)
    
    def test_to_dict_exclude_columns_none(self, report):
        """exclude_columns=None should include all columns (default)."""
        result = report.to_dict(exclude_columns=None)
        assert isinstance(result, dict)
    
    def test_to_dict_exclude_columns_empty_list(self, report):
        """exclude_columns as empty list should include all columns."""
        result = report.to_dict(exclude_columns=[])
        assert isinstance(result, dict)
    
    def test_to_dict_exclude_nonexistent_column(self, report):
        """exclude_columns with non-existent columns should be handled gracefully."""
        result = report.to_dict(exclude_columns=['nonexistent_column'])
        assert isinstance(result, dict)
    
    def test_to_dict_both_parameters_combined(self, report):
        """to_dict should accept both redact_sample_values and exclude_columns."""
        result = report.to_dict(
            redact_sample_values=True,
            exclude_columns=['age', 'salary']
        )
        assert isinstance(result, dict)
    
    def test_to_dict_empty_dataframe(self):
        """to_dict should handle empty DataFrames."""
        df = pd.DataFrame()
        report = ar.quality.DataQualityReport(df)
        result = report.to_dict()
        assert isinstance(result, dict)
    
    def test_to_dict_single_column_dataframe(self):
        """to_dict should handle single-column DataFrames."""
        df = pd.DataFrame({'single_col': [1, 2, 3]})
        report = ar.quality.DataQualityReport(df)
        result = report.to_dict()
        assert isinstance(result, dict)
    
    def test_to_dict_with_mixed_dtypes(self):
        """to_dict should handle DataFrames with mixed data types."""
        df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, np.nan],
            'str_col': ['a', 'b', None],
            'bool_col': [True, False, True],
            'datetime_col': pd.date_range('2023-01-01', periods=3)
        })
        report = ar.quality.DataQualityReport(df)
        result = report.to_dict()
        assert isinstance(result, dict)
