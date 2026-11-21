import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestCompareProfiles:
    """Tests for arnio.quality.compare_profiles function."""

    def test_identical_profiles_succeeds(self):
        """Comparing identical profiles should complete without error."""
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': ['x', 'y', 'z']
        })
        profile_a = ar.profile(df)
        profile_b = ar.profile(df)
        
        result = ar.compare_profiles(profile_a, profile_b)
        
        assert result is not None
        assert len(result.drift_report) == 2

    def test_different_columns_raises_valueerror(self):
        """ValueError raised when profiles have different columns."""
        df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        df2 = pd.DataFrame({'a': [1, 2, 3], 'c': [4, 5, 6]})
        
        profile_a = ar.profile(df1)
        profile_b = ar.profile(df2)
        
        with pytest.raises(ValueError, match="incompatible schemas"):
            ar.compare_profiles(profile_a, profile_b)

    def test_column_missing_from_first_profile(self):
        """Error message specifies column missing from profile_a."""
        df1 = pd.DataFrame({'a': [1, 2, 3]})
        df2 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        
        profile_a = ar.profile(df1)
        profile_b = ar.profile(df2)
        
        with pytest.raises(ValueError, match="missing from profile_a.*\\['b'\\]"):
            ar.compare_profiles(profile_a, profile_b)

    def test_column_missing_from_second_profile(self):
        """Error message specifies column missing from profile_b."""
        df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        df2 = pd.DataFrame({'a': [1, 2, 3]})
        
        profile_a = ar.profile(df1)
        profile_b = ar.profile(df2)
        
        with pytest.raises(ValueError, match="missing from profile_b.*\\['b'\\]"):
            ar.compare_profiles(profile_a, profile_b)

    def test_invalid_type_first_argument(self):
        """TypeError raised if first argument is not DataQualityReport."""
        df = pd.DataFrame({'a': [1, 2, 3]})
        profile = ar.profile(df)
        
        with pytest.raises(TypeError, match="DataQualityReport"):
            ar.compare_profiles(None, profile)

    def test_invalid_type_second_argument(self):
        """TypeError raised if second argument is not DataQualityReport."""
        df = pd.DataFrame({'a': [1, 2, 3]})
        profile = ar.profile(df)
        
        with pytest.raises(TypeError, match="DataQualityReport"):
            ar.compare_profiles(profile, "not a profile")

    def test_invalid_type_both_arguments(self):
        """TypeError raised if both arguments are invalid."""
        with pytest.raises(TypeError, match="DataQualityReport"):
            ar.compare_profiles({}, [])

    def test_result_preserves_profiles(self):
        """Result contains references to input profiles."""
        df = pd.DataFrame({'x': [1, 2, 3]})
        profile_a = ar.profile(df)
        profile_b = ar.profile(df)
        
        result = ar.compare_profiles(profile_a, profile_b)
        
        assert result.left_profile is profile_a
        assert result.right_profile is profile_b

    def test_drift_report_has_all_columns(self):
        """Drift report contains entry for each column."""
        df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'str_col': ['a', 'b', 'c'],
            'float_col': [1.5, 2.5, 3.5]
        })
        profile_a = ar.profile(df)
        profile_b = ar.profile(df)
        
        result = ar.compare_profiles(profile_a, profile_b)
        
        assert set(result.drift_report.keys()) == {'int_col', 'str_col', 'float_col'}

    def test_drift_report_entries_have_status(self):
        """Each drift report entry contains a status field."""
        df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
        profile_a = ar.profile(df)
        profile_b = ar.profile(df)
        
        result = ar.compare_profiles(profile_a, profile_b)
        
        for column, entry in result.drift_report.items():
            assert 'status' in entry
            assert entry['status'] in ['ok', 'warning', 'changed']

    def test_status_counts_sums_correctly(self):
        """Status counts total equals number of columns."""
        df = pd.DataFrame({
            'col_a': [1, 2, 3],
            'col_b': ['x', 'y', 'z'],
            'col_c': [1.0, 2.0, None]
        })
        profile_a = ar.profile(df)
        profile_b = ar.profile(df)
        
        result = ar.compare_profiles(profile_a, profile_b)
        
        total = sum(result.status_counts.values())
        assert total == len(result.drift_report) == 3

    def test_null_ratio_changes_detected(self):
        """Changes in null ratios between profiles are captured."""
        df1 = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        df2 = pd.DataFrame({'a': [1, 2, 3, None, None]})
        
        profile_a = ar.profile(df1)
        profile_b = ar.profile(df2)
        
        result = ar.compare_profiles(profile_a, profile_b)
        
        assert 'a' in result.drift_report
        assert result.drift_report['a']['status'] in ['ok', 'warning', 'changed']

    def test_numeric_distribution_changes_detected(self):
        """Changes in numeric distributions (mean, std, min, max) detected."""
        df1 = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
        df2 = pd.DataFrame({'value': [100, 200, 300, 400, 500]})
        
        profile_a = ar.profile(df1)
        profile_b = ar.profile(df2)
        
        result = ar.compare_profiles(profile_a, profile_b)
        
        assert 'value' in result.drift_report
        assert isinstance(result.drift_report['value']['status'], str)

    def test_single_column_comparison(self):
        """Comparison works correctly with single column dataframes."""
        df = pd.DataFrame({'col': range(10)})
        
        profile_a = ar.profile(df)
        profile_b = ar.profile(df)
        
        result = ar.compare_profiles(profile_a, profile_b)
        
        assert len(result.drift_report) == 1
        assert 'col' in result.drift_report

    def test_many_columns_comparison(self):
        """Comparison handles dataframes with many columns."""
        df = pd.DataFrame({f'col_{i}': range(5) for i in range(15)})
        
        profile_a = ar.profile(df)
        profile_b = ar.profile(df)
        
        result = ar.compare_profiles(profile_a, profile_b)
        
        assert len(result.drift_report) == 15
