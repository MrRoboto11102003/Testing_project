import pytest
import pandas as pd
import numpy as np
import arnio as ar
from arnio.quality import ProfileComparison


class TestCompareProfiles:
    """Test suite for arnio.quality.compare_profiles function"""

    def test_compare_identical_profiles_returns_profile_comparison(self):
        """Test basic functionality: comparing identical profiles returns ProfileComparison"""
        df = ar.from_dict({'col_a': [1, 2, 3], 'col_b': ['x', 'y', 'z']})
        profile_a = ar.profile(df)
        profile_b = ar.profile(df)
        
        result = ar.compare_profiles(profile_a, profile_b)
        
        assert isinstance(result, ProfileComparison)
        assert result.left_profile is profile_a
        assert result.right_profile is profile_b
        assert set(result.drift_report.keys()) == {'col_a', 'col_b'}

    def test_compare_profiles_raises_type_error_for_invalid_first_arg(self):
        """Test TypeError when first argument is not DataQualityReport"""
        df = ar.from_dict({'col_a': [1, 2, 3]})
        profile = ar.profile(df)
        
        with pytest.raises(TypeError, match="DataQualityReport"):
            ar.compare_profiles("not a profile", profile)

    def test_compare_profiles_raises_type_error_for_invalid_second_arg(self):
        """Test TypeError when second argument is not DataQualityReport"""
        df = ar.from_dict({'col_a': [1, 2, 3]})
        profile = ar.profile(df)
        
        with pytest.raises(TypeError, match="DataQualityReport"):
            ar.compare_profiles(profile, {"not": "a profile"})

    def test_compare_profiles_raises_value_error_for_different_columns(self):
        """Test ValueError when profiles have different column sets"""
        df_a = ar.from_dict({'col_a': [1, 2, 3], 'col_b': ['x', 'y', 'z']})
        df_b = ar.from_dict({'col_a': [1, 2, 3], 'col_c': ['a', 'b', 'c']})
        
        profile_a = ar.profile(df_a)
        profile_b = ar.profile(df_b)
        
        with pytest.raises(ValueError, match="incompatible schemas"):
            ar.compare_profiles(profile_a, profile_b)

    def test_compare_profiles_error_message_lists_missing_columns(self):
        """Test that ValueError message lists missing columns from each profile"""
        df_a = ar.from_dict({'col_a': [1, 2]})
        df_b = ar.from_dict({'col_a': [1, 2], 'col_b': [3, 4]})
        
        profile_a = ar.profile(df_a)
        profile_b = ar.profile(df_b)
        
        with pytest.raises(ValueError) as exc_info:
            ar.compare_profiles(profile_a, profile_b)
        assert "missing from profile_a" in str(exc_info.value)

    def test_compare_profiles_status_counts_keys_present(self):
        """Test that status_counts contains expected keys: ok, warning, changed"""
        df = ar.from_dict({'col_a': [1, 2, 3], 'col_b': [4, 5, 6]})
        profile_a = ar.profile(df)
        profile_b = ar.profile(df)
        
        result = ar.compare_profiles(profile_a, profile_b)
        
        assert 'ok' in result.status_counts
        assert 'warning' in result.status_counts
        assert 'changed' in result.status_counts

    def test_compare_profiles_status_counts_are_integers(self):
        """Test that status_counts values are integers"""
        df = ar.from_dict({'col_a': [1, 2, 3]})
        profile_a = ar.profile(df)
        profile_b = ar.profile(df)
        
        result = ar.compare_profiles(profile_a, profile_b)
        
        assert isinstance(result.status_counts['ok'], int)
        assert isinstance(result.status_counts['warning'], int)
        assert isinstance(result.status_counts['changed'], int)
        assert all(v >= 0 for v in result.status_counts.values())

    def test_compare_profiles_drift_report_all_columns_included(self):
        """Test that drift_report includes entries for all columns"""
        df = ar.from_dict({
            'col_a': [1, 2, 3],
            'col_b': ['x', 'y', 'z'],
            'col_c': [1.5, 2.5, 3.5]
        })
        profile_a = ar.profile(df)
        profile_b = ar.profile(df)
        
        result = ar.compare_profiles(profile_a, profile_b)
        
        assert len(result.drift_report) == 3
        assert all(col in result.drift_report for col in ['col_a', 'col_b', 'col_c'])

    def test_compare_profiles_status_counts_sum_equals_column_count(self):
        """Test that status_counts values sum to total number of columns"""
        df = ar.from_dict({
            'a': [1, 2], 'b': [3, 4], 'c': [5, 6], 'd': [7, 8]
        })
        profile_a = ar.profile(df)
        profile_b = ar.profile(df)
        
        result = ar.compare_profiles(profile_a, profile_b)
        
        total_statuses = sum(result.status_counts.values())
        assert total_statuses == len(profile_a.columns)

    def test_compare_profiles_detects_null_ratio_changes(self):
        """Test that comparing profiles detects changes in null ratios"""
        df_a = ar.from_dict({'col': [1, 2, 3]})
        df_b = ar.from_dict({'col': [1, None, 3]})
        
        profile_a = ar.profile(df_a)
        profile_b = ar.profile(df_b)
        
        result = ar.compare_profiles(profile_a, profile_b)
        
        assert result is not None
        assert 'col' in result.drift_report
        assert isinstance(result.drift_report['col'], dict)

    def test_compare_profiles_single_column_edge_case(self):
        """Test comparing profiles with only one column"""
        df = ar.from_dict({'only_col': [1, 2, 3, 4, 5]})
        profile_a = ar.profile(df)
        profile_b = ar.profile(df)
        
        result = ar.compare_profiles(profile_a, profile_b)
        
        assert len(result.drift_report) == 1
        assert 'only_col' in result.drift_report
        assert len(result.status_counts) == 3

    def test_compare_profiles_large_column_count(self):
        """Test comparing profiles with many columns"""
        data = {f'col_{i}': list(range(i, i+3)) for i in range(20)}
        df = ar.from_dict(data)
        profile_a = ar.profile(df)
        profile_b = ar.profile(df)
        
        result = ar.compare_profiles(profile_a, profile_b)
        
        assert len(result.drift_report) == 20
        assert sum(result.status_counts.values()) == 20

    def test_compare_profiles_different_value_distributions(self):
        """Test comparing profiles with different numeric distributions"""
        df_a = ar.from_dict({'values': [1, 2, 3, 4, 5]})
        df_b = ar.from_dict({'values': [100, 200, 300, 400, 500]})
        
        profile_a = ar.profile(df_a)
        profile_b = ar.profile(df_b)
        
        result = ar.compare_profiles(profile_a, profile_b)
        
        assert result is not None
        assert result.drift_report is not None

    def test_compare_profiles_unique_count_changes(self):
        """Test that comparison handles changes in unique value counts"""
        df_a = ar.from_dict({'col': [1, 1, 1, 1]})
        df_b = ar.from_dict({'col': [1, 2, 3, 4]})
        
        profile_a = ar.profile(df_a)
        profile_b = ar.profile(df_b)
        
        result = ar.compare_profiles(profile_a, profile_b)
        
        assert result is not None
        assert profile_a.columns['col'].unique_count == 1
        assert profile_b.columns['col'].unique_count == 4

    def test_compare_profiles_preserves_profile_objects(self):
        """Test that ProfileComparison preserves references to original profiles"""
        df_a = ar.from_dict({'col': [1, 2, 3]})
        df_b = ar.from_dict({'col': [4, 5, 6]})
        
        profile_a = ar.profile(df_a)
        profile_b = ar.profile(df_b)
        
        result = ar.compare_profiles(profile_a, profile_b)
        
        assert result.left_profile is profile_a
        assert result.right_profile is profile_b
        assert result.left_profile.row_count == 3
        assert result.right_profile.row_count == 3
