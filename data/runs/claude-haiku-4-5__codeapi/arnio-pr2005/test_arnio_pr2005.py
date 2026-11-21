import pytest
import arnio as ar


def create_test_arframe(data):
    """Helper to create ArFrame from dict."""
    return ar.from_dict(data)


class TestProfileComparisonToDict:
    """Tests for ProfileComparison.to_dict method."""
    
    def test_to_dict_returns_dict(self):
        """Test that to_dict returns a dictionary."""
        frame1 = create_test_arframe({'col1': [1, 2, 3]})
        frame2 = create_test_arframe({'col1': [1, 2, 3, 4]})
        
        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)
        comparison = ar.compare_profiles(profile1, profile2)
        
        result = comparison.to_dict()
        assert isinstance(result, dict)
    
    def test_to_dict_has_required_keys(self):
        """Test that to_dict result contains all required keys."""
        frame1 = create_test_arframe({'col1': [1, 2, 3]})
        frame2 = create_test_arframe({'col1': [1, 2, 3, 4]})
        
        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)
        comparison = ar.compare_profiles(profile1, profile2)
        
        result = comparison.to_dict()
        
        assert 'left_profile' in result
        assert 'right_profile' in result
        assert 'status_counts' in result
        assert 'drift_report' in result
    
    def test_to_dict_nested_profiles_are_dicts(self):
        """Test that nested profiles are dictionaries."""
        frame1 = create_test_arframe({'col1': [1, 2, 3]})
        frame2 = create_test_arframe({'col1': [1, 2, 3, 4]})
        
        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)
        comparison = ar.compare_profiles(profile1, profile2)
        
        result = comparison.to_dict()
        
        assert isinstance(result['left_profile'], dict)
        assert isinstance(result['right_profile'], dict)
        assert isinstance(result['drift_report'], dict)
    
    def test_to_dict_status_counts_is_dict_copy(self):
        """Test that status_counts is returned as a dict copy."""
        frame1 = create_test_arframe({'col1': [1, 2, 3]})
        frame2 = create_test_arframe({'col1': [1, 2, 3, 4]})
        
        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)
        comparison = ar.compare_profiles(profile1, profile2)
        
        result = comparison.to_dict()
        
        assert isinstance(result['status_counts'], dict)
        assert result['status_counts'] is not comparison.status_counts
    
    def test_to_dict_with_redact_sample_values_true(self):
        """Test with redact_sample_values=True."""
        frame1 = create_test_arframe({'col1': [1, 2, 3]})
        frame2 = create_test_arframe({'col1': [1, 2, 3, 4]})
        
        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)
        comparison = ar.compare_profiles(profile1, profile2)
        
        result = comparison.to_dict(redact_sample_values=True)
        
        assert isinstance(result, dict)
        assert 'left_profile' in result
    
    def test_to_dict_with_redact_sample_values_false(self):
        """Test with redact_sample_values=False (explicit)."""
        frame1 = create_test_arframe({'col1': [1, 2, 3]})
        frame2 = create_test_arframe({'col1': [1, 2, 3, 4]})
        
        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)
        comparison = ar.compare_profiles(profile1, profile2)
        
        result = comparison.to_dict(redact_sample_values=False)
        
        assert isinstance(result, dict)
    
    def test_to_dict_exclude_columns_with_list(self):
        """Test exclude_columns parameter with list."""
        frame1 = create_test_arframe({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        frame2 = create_test_arframe({'col1': [1, 2, 3, 4], 'col2': ['a', 'b', 'c', 'd']})
        
        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)
        comparison = ar.compare_profiles(profile1, profile2)
        
        result = comparison.to_dict(exclude_columns=['col1'])
        
        assert isinstance(result, dict)
    
    def test_to_dict_exclude_columns_with_set(self):
        """Test exclude_columns parameter with set."""
        frame1 = create_test_arframe({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        frame2 = create_test_arframe({'col1': [1, 2, 3, 4], 'col2': ['a', 'b', 'c', 'd']})
        
        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)
        comparison = ar.compare_profiles(profile1, profile2)
        
        result = comparison.to_dict(exclude_columns={'col2'})
        
        assert isinstance(result, dict)
    
    def test_to_dict_exclude_columns_with_tuple(self):
        """Test exclude_columns parameter with tuple."""
        frame1 = create_test_arframe({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c'],
            'col3': [1.0, 2.0, 3.0]
        })
        frame2 = create_test_arframe({
            'col1': [1, 2, 3, 4],
            'col2': ['a', 'b', 'c', 'd'],
            'col3': [1.0, 2.0, 3.0, 4.0]
        })
        
        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)
        comparison = ar.compare_profiles(profile1, profile2)
        
        result = comparison.to_dict(exclude_columns=('col1', 'col3'))
        
        assert isinstance(result, dict)
    
    def test_to_dict_exclude_columns_none(self):
        """Test exclude_columns=None (default)."""
        frame1 = create_test_arframe({'col1': [1, 2, 3]})
        frame2 = create_test_arframe({'col1': [1, 2, 3, 4]})
        
        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)
        comparison = ar.compare_profiles(profile1, profile2)
        
        result = comparison.to_dict(exclude_columns=None)
        
        assert isinstance(result, dict)
    
    def test_to_dict_both_parameters_combined(self):
        """Test with both redact_sample_values and exclude_columns."""
        frame1 = create_test_arframe({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        frame2 = create_test_arframe({'col1': [1, 2, 3, 4], 'col2': ['a', 'b', 'c', 'd']})
        
        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)
        comparison = ar.compare_profiles(profile1, profile2)
        
        result = comparison.to_dict(
            redact_sample_values=True,
            exclude_columns=['col1']
        )
        
        assert isinstance(result, dict)
        assert all(k in result for k in 
                  ['left_profile', 'right_profile', 'status_counts', 'drift_report'])
    
    def test_to_dict_with_null_values_in_frames(self):
        """Test with frames containing null values."""
        frame1 = create_test_arframe({
            'col1': [1, 2, None],
            'col2': ['a', None, 'c']
        })
        frame2 = create_test_arframe({
            'col1': [1, 2, None, 4],
            'col2': ['a', None, 'c', 'd']
        })
        
        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)
        comparison = ar.compare_profiles(profile1, profile2)
        
        result = comparison.to_dict()
        
        assert isinstance(result, dict)
    
    def test_to_dict_with_multiple_data_types(self):
        """Test with frames having multiple column types."""
        frame1 = create_test_arframe({
            'int_col': [1, 2, 3],
            'str_col': ['a', 'b', 'c'],
            'float_col': [1.1, 2.2, 3.3]
        })
        frame2 = create_test_arframe({
            'int_col': [1, 2, 3, 4],
            'str_col': ['a', 'b', 'c', 'd'],
            'float_col': [1.1, 2.2, 3.3, 4.4]
        })
        
        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)
        comparison = ar.compare_profiles(profile1, profile2)
        
        result = comparison.to_dict()
        
        assert isinstance(result, dict)
    
    def test_to_dict_drift_report_structure(self):
        """Test that drift_report contains expected structure."""
        frame1 = create_test_arframe({'col1': [1, 2, 3]})
        frame2 = create_test_arframe({'col1': [1, 2, 3, 4, 5]})
        
        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)
        comparison = ar.compare_profiles(profile1, profile2)
        
        result = comparison.to_dict()
        drift_report = result['drift_report']
        
        assert isinstance(drift_report, dict)
        for key, value in drift_report.items():
            assert isinstance(value, dict)
    
    def test_to_dict_empty_exclude_columns(self):
        """Test exclude_columns with empty list."""
        frame1 = create_test_arframe({'col1': [1, 2, 3]})
        frame2 = create_test_arframe({'col1': [1, 2, 3, 4]})
        
        profile1 = ar.profile(frame1)
        profile2 = ar.profile(frame2)
        comparison = ar.compare_profiles(profile1, profile2)
        
        result = comparison.to_dict(exclude_columns=[])
        
        assert isinstance(result, dict)
