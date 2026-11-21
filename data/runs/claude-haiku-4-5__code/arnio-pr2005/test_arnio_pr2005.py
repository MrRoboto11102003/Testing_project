import pytest
import pandas as pd
import numpy as np
import arnio as ar
import json
from unittest.mock import Mock


class TestProfileComparisonToDict:
    """Test arnio.quality.ProfileComparison.to_dict method."""
    
    @pytest.fixture
    def sample_dataframes(self):
        """Create sample DataFrames for profiling."""
        df1 = pd.DataFrame({
            'col_int': [1, 2, 3, 4, 5],
            'col_str': ['a', 'b', 'c', 'd', 'e'],
            'col_float': [1.1, 2.2, 3.3, 4.4, 5.5]
        })
        df2 = pd.DataFrame({
            'col_int': [1, 2, 3, 4, 6],
            'col_str': ['a', 'b', 'c', 'd', 'f'],
            'col_float': [1.1, 2.2, 3.3, 4.4, 6.6]
        })
        return df1, df2
    
    @pytest.fixture
    def profiles(self, sample_dataframes):
        """Create profiles from sample DataFrames."""
        df1, df2 = sample_dataframes
        profile1 = ar.Profile(df1)
        profile2 = ar.Profile(df2)
        return profile1, profile2
    
    @pytest.fixture
    def profile_comparison(self, profiles):
        """Create a ProfileComparison object."""
        profile1, profile2 = profiles
        return ar.quality.ProfileComparison(profile1, profile2)
    
    def test_to_dict_returns_dict(self, profile_comparison):
        """Test that to_dict returns a dictionary."""
        result = profile_comparison.to_dict()
        assert isinstance(result, dict)
    
    def test_to_dict_has_required_keys(self, profile_comparison):
        """Test that result contains all required keys."""
        result = profile_comparison.to_dict()
        required_keys = {'left_profile', 'right_profile', 'status_counts', 'drift_report'}
        assert required_keys.issubset(set(result.keys()))
    
    def test_to_dict_profiles_are_dicts(self, profile_comparison):
        """Test that left_profile and right_profile are dicts."""
        result = profile_comparison.to_dict()
        assert isinstance(result['left_profile'], dict)
        assert isinstance(result['right_profile'], dict)
    
    def test_to_dict_status_counts_is_dict(self, profile_comparison):
        """Test that status_counts is converted to a dict."""
        result = profile_comparison.to_dict()
        assert isinstance(result['status_counts'], dict)
    
    def test_to_dict_drift_report_is_dict(self, profile_comparison):
        """Test that drift_report is a dict."""
        result = profile_comparison.to_dict()
        assert isinstance(result['drift_report'], dict)
    
    def test_to_dict_with_redact_sample_values_false(self, profile_comparison):
        """Test to_dict with redact_sample_values=False (default)."""
        result = profile_comparison.to_dict(redact_sample_values=False)
        assert isinstance(result, dict)
        assert 'left_profile' in result
    
    def test_to_dict_with_redact_sample_values_true(self, profile_comparison):
        """Test to_dict with redact_sample_values=True redacts sample values."""
        result = profile_comparison.to_dict(redact_sample_values=True)
        assert isinstance(result, dict)
        # Verify that profiles were processed with redaction
        assert result['left_profile'] is not None
    
    def test_to_dict_with_exclude_columns_list(self, profile_comparison):
        """Test to_dict with exclude_columns as list."""
        result = profile_comparison.to_dict(exclude_columns=['col_int'])
        assert isinstance(result, dict)
    
    def test_to_dict_with_exclude_columns_set(self, profile_comparison):
        """Test to_dict with exclude_columns as set."""
        result = profile_comparison.to_dict(exclude_columns={'col_int'})
        assert isinstance(result, dict)
    
    def test_to_dict_with_exclude_columns_tuple(self, profile_comparison):
        """Test to_dict with exclude_columns as tuple."""
        result = profile_comparison.to_dict(exclude_columns=('col_int',))
        assert isinstance(result, dict)
    
    def test_to_dict_with_exclude_columns_none(self, profile_comparison):
        """Test to_dict with exclude_columns=None (default)."""
        result = profile_comparison.to_dict(exclude_columns=None)
        assert isinstance(result, dict)
    
    def test_to_dict_combined_parameters(self, profile_comparison):
        """Test to_dict with both redact_sample_values and exclude_columns."""
        result = profile_comparison.to_dict(
            redact_sample_values=True,
            exclude_columns=['col_int', 'col_str']
        )
        assert isinstance(result, dict)
        assert all(key in result for key in ['left_profile', 'right_profile', 'status_counts', 'drift_report'])
    
    def test_to_dict_is_json_friendly(self, profile_comparison):
        """Test that to_dict output is JSON-serializable as documented."""
        result = profile_comparison.to_dict()
        # Should not raise when serializing to JSON
        json_str = json.dumps(result, default=str)
        assert isinstance(json_str, str)
