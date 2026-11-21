import pytest
import pandas as pd
import numpy as np
import sys
import subprocess
from pathlib import Path

import arnio as ar


class TestCompareProfilesWorkflow:
    """End-to-end workflow tests for compare_profiles."""
    
    def test_basic_profile_comparison_workflow(self, tmp_path):
        """Basic workflow: read CSVs, profile, and compare."""
        baseline_data = pd.DataFrame({
            'customer_id': [1, 2, 3, 4, 5],
            'purchase_amount': [100.0, 200.0, 150.0, 300.0, 250.0],
            'region': ['US', 'EU', 'US', 'ASIA', 'EU']
        })
        
        current_data = baseline_data.copy()
        
        baseline_file = tmp_path / "baseline.csv"
        current_file = tmp_path / "current.csv"
        baseline_data.to_csv(baseline_file, index=False)
        current_data.to_csv(current_file, index=False)
        
        baseline_profile = ar.profile(ar.read_csv(str(baseline_file)))
        current_profile = ar.profile(ar.read_csv(str(current_file)))
        
        comparison = ar.compare_profiles(baseline_profile, current_profile)
        
        assert isinstance(comparison.drift_report, dict)
        assert set(comparison.drift_report.keys()) == {'customer_id', 'purchase_amount', 'region'}
        assert all(entry['status'] == 'ok' for entry in comparison.drift_report.values())
    
    def test_profile_comparison_detects_numeric_drift(self, tmp_path):
        """Compare profiles with changed numeric distributions."""
        np.random.seed(42)
        baseline_data = pd.DataFrame({
            'id': np.arange(100),
            'metric': np.random.normal(100, 15, 100)
        })
        
        np.random.seed(43)
        current_data = pd.DataFrame({
            'id': np.arange(100),
            'metric': np.random.normal(150, 25, 100)
        })
        
        baseline_file = tmp_path / "baseline.csv"
        current_file = tmp_path / "current.csv"
        baseline_data.to_csv(baseline_file, index=False)
        current_data.to_csv(current_file, index=False)
        
        baseline_profile = ar.profile(ar.read_csv(str(baseline_file)))
        current_profile = ar.profile(ar.read_csv(str(current_file)))
        
        comparison = ar.compare_profiles(baseline_profile, current_profile)
        
        assert comparison.drift_report['metric']['status'] in ['warning', 'changed']
        assert comparison.status_counts['changed'] > 0 or comparison.status_counts['warning'] > 0
    
    def test_profile_comparison_detects_null_drift(self, tmp_path):
        """Compare profiles with changed null ratios."""
        baseline_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'value': [10, 20, 30, 40, 50]
        })
        
        current_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'value': [10, None, 30, None, 50]
        })
        
        baseline_file = tmp_path / "baseline.csv"
        current_file = tmp_path / "current.csv"
        baseline_data.to_csv(baseline_file, index=False)
        current_data.to_csv(current_file, index=False)
        
        baseline_profile = ar.profile(ar.read_csv(str(baseline_file)))
        current_profile = ar.profile(ar.read_csv(str(current_file)))
        
        comparison = ar.compare_profiles(baseline_profile, current_profile)
        
        assert comparison.drift_report['value']['status'] in ['warning', 'changed']


class TestCompareProfilesSchemaValidation:
    """Tests for schema validation and error handling."""
    
    def test_raises_error_on_mismatched_columns_missing_from_a(self, tmp_path):
        """Raise ValueError when profile_a missing columns present in profile_b."""
        data_a = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})
        data_b = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob'], 'age': [25, 30]})
        
        file_a = tmp_path / "a.csv"
        file_b = tmp_path / "b.csv"
        data_a.to_csv(file_a, index=False)
        data_b.to_csv(file_b, index=False)
        
        profile_a = ar.profile(ar.read_csv(str(file_a)))
        profile_b = ar.profile(ar.read_csv(str(file_b)))
        
        with pytest.raises(ValueError) as exc_info:
            ar.compare_profiles(profile_a, profile_b)
        
        error = str(exc_info.value)
        assert "incompatible schemas" in error
        assert "age" in error
    
    def test_raises_error_on_mismatched_columns_missing_from_b(self, tmp_path):
        """Raise ValueError when profile_b missing columns present in profile_a."""
        data_a = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob'], 'age': [25, 30]})
        data_b = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})
        
        file_a = tmp_path / "a.csv"
        file_b = tmp_path / "b.csv"
        data_a.to_csv(file_a, index=False)
        data_b.to_csv(file_b, index=False)
        
        profile_a = ar.profile(ar.read_csv(str(file_a)))
        profile_b = ar.profile(ar.read_csv(str(file_b)))
        
        with pytest.raises(ValueError) as exc_info:
            ar.compare_profiles(profile_a, profile_b)
        
        error = str(exc_info.value)
        assert "incompatible schemas" in error
        assert "age" in error
    
    def test_raises_type_error_on_invalid_first_argument(self, tmp_path):
        """Raise TypeError when first argument is not DataQualityReport."""
        data = pd.DataFrame({'id': [1, 2], 'value': [1.5, 2.5]})
        csv_file = tmp_path / "data.csv"
        data.to_csv(csv_file, index=False)
        
        profile = ar.profile(ar.read_csv(str(csv_file)))
        df = pd.DataFrame({'id': [1, 2], 'value': [1.5, 2.5]})
        
        with pytest.raises(TypeError):
            ar.compare_profiles(df, profile)
    
    def test_raises_type_error_on_invalid_second_argument(self, tmp_path):
        """Raise TypeError when second argument is not DataQualityReport."""
        data = pd.DataFrame({'id': [1, 2], 'value': [1.5, 2.5]})
        csv_file = tmp_path / "data.csv"
        data.to_csv(csv_file, index=False)
        
        profile = ar.profile(ar.read_csv(str(csv_file)))
        df = pd.DataFrame({'id': [1, 2], 'value': [1.5, 2.5]})
        
        with pytest.raises(TypeError):
            ar.compare_profiles(profile, df)


class TestCompareProfilesReturnValue:
    """Tests for the ProfileComparison return value structure."""
    
    def test_returns_profile_comparison_with_required_fields(self, tmp_path):
        """ProfileComparison has drift_report, left_profile, right_profile, status_counts."""
        data = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        csv_file = tmp_path / "data.csv"
        data.to_csv(csv_file, index=False)
        
        profile = ar.profile(ar.read_csv(str(csv_file)))
        comparison = ar.compare_profiles(profile, profile)
        
        assert hasattr(comparison, 'drift_report')
        assert hasattr(comparison, 'left_profile')
        assert hasattr(comparison, 'right_profile')
        assert hasattr(comparison, 'status_counts')
        
        assert comparison.left_profile is profile
        assert comparison.right_profile is profile
        assert isinstance(comparison.drift_report, dict)
        assert isinstance(comparison.status_counts, dict)
    
    def test_drift_report_contains_all_columns(self, tmp_path):
        """drift_report should have entries for all columns in profiles."""
        data = pd.DataFrame({
            'col_a': [1, 2, 3],
            'col_b': ['x', 'y', 'z'],
            'col_c': [1.1, 2.2, 3.3]
        })
        csv_file = tmp_path / "data.csv"
        data.to_csv(csv_file, index=False)
        
        profile = ar.profile(ar.read_csv(str(csv_file)))
        comparison = ar.compare_profiles(profile, profile)
        
        assert len(comparison.drift_report) == 3
        assert all(col in comparison.drift_report for col in ['col_a', 'col_b', 'col_c'])
    
    def test_status_counts_has_ok_warning_changed_keys(self, tmp_path):
        """status_counts should have 'ok', 'warning', 'changed' keys with int values."""
        data = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})
        csv_file = tmp_path / "data.csv"
        data.to_csv(csv_file, index=False)
        
        profile = ar.profile(ar.read_csv(str(csv_file)))
        comparison = ar.compare_profiles(profile, profile)
        
        assert 'ok' in comparison.status_counts
        assert 'warning' in comparison.status_counts
        assert 'changed' in comparison.status_counts
        assert all(isinstance(v, int) for v in comparison.status_counts.values())
        assert sum(comparison.status_counts.values()) == 3


class TestCompareProfilesEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_single_row_datasets(self, tmp_path):
        """Compare profiles from single-row datasets."""
        data_a = pd.DataFrame({'id': [1], 'value': [100]})
        data_b = pd.DataFrame({'id': [1], 'value': [100]})
        
        file_a = tmp_path / "a.csv"
        file_b = tmp_path / "b.csv"
        data_a.to_csv(file_a, index=False)
        data_b.to_csv(file_b, index=False)
        
        profile_a = ar.profile(ar.read_csv(str(file_a)))
        profile_b = ar.profile(ar.read_csv(str(file_b)))
        
        comparison = ar.compare_profiles(profile_a, profile_b)
        
        assert len(comparison.drift_report) == 2
        assert isinstance(comparison.drift_report, dict)
    
    def test_all_null_column_comparison(self, tmp_path):
        """Compare profiles where a column is all nulls."""
        data_a = pd.DataFrame({
            'id': [1, 2, 3],
            'all_nulls': [None, None, None]
        })
        data_b = pd.DataFrame({
            'id': [1, 2, 3],
            'all_nulls': [None, None, None]
        })
        
        file_a = tmp_path / "a.csv"
        file_b = tmp_path / "b.csv"
        data_a.to_csv(file_a, index=False)
        data_b.to_csv(file_b, index=False)
        
        profile_a = ar.profile(ar.read_csv(str(file_a)))
        profile_b = ar.profile(ar.read_csv(str(file_b)))
        
        comparison = ar.compare_profiles(profile_a, profile_b)
        
        assert 'all_nulls' in comparison.drift_report
    
    def test_many_columns(self, tmp_path):
        """Compare profiles with 50 columns."""
        data_dict = {f'col_{i:03d}': np.arange(50) for i in range(50)}
        data_a = pd.DataFrame(data_dict)
        data_b = pd.DataFrame(data_dict)
        
        file_a = tmp_path / "a.csv"
        file_b = tmp_path / "b.csv"
        data_a.to_csv(file_a, index=False)
        data_b.to_csv(file_b, index=False)
        
        profile_a = ar.profile(ar.read_csv(str(file_a)))
        profile_b = ar.profile(ar.read_csv(str(file_b)))
        
        comparison = ar.compare_profiles(profile_a, profile_b)
        
        assert len(comparison.drift_report) == 50
        assert sum(comparison.status_counts.values()) == 50
    
    def test_mixed_dtype_columns(self, tmp_path):
        """Compare profiles with mixed data types."""
        data_a = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'str_col': ['a', 'b', 'c'],
            'bool_col': [True, False, True]
        })
        data_b = data_a.copy()
        
        file_a = tmp_path / "a.csv"
        file_b = tmp_path / "b.csv"
        data_a.to_csv(file_a, index=False)
        data_b.to_csv(file_b, index=False)
        
        profile_a = ar.profile(ar.read_csv(str(file_a)))
        profile_b = ar.profile(ar.read_csv(str(file_b)))
        
        comparison = ar.compare_profiles(profile_a, profile_b)
        
        assert len(comparison.drift_report) == 4
