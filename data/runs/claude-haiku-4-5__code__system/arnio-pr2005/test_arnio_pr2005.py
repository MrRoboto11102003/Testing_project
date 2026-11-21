import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_profile_comparison_to_dict_basic_output_structure(tmp_path):
    """Test basic to_dict returns correct structure with required keys."""
    import arnio as ar
    
    df1 = pd.DataFrame({
        'col1': [1, 2, 3, 4, 5],
        'col2': ['a', 'b', 'c', 'd', 'e']
    })
    df2 = pd.DataFrame({
        'col1': [1, 2, 3, 4, 6],
        'col2': ['a', 'b', 'c', 'd', 'f']
    })
    
    profile1 = ar.profile(df1)
    profile2 = ar.profile(df2)
    comparison = ar.compare_profiles(profile1, profile2)
    
    result = comparison.to_dict()
    
    assert isinstance(result, dict)
    assert 'left_profile' in result
    assert 'right_profile' in result
    assert 'status_counts' in result
    assert 'drift_report' in result
    assert isinstance(result['status_counts'], dict)
    assert isinstance(result['drift_report'], dict)


def test_profile_comparison_to_dict_redact_sample_values_true(tmp_path):
    """Test to_dict with redact_sample_values=True."""
    import arnio as ar
    
    df1 = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': ['sensitive_value_1', 'sensitive_value_2', 'sensitive_value_3']
    })
    df2 = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': ['sensitive_value_1', 'sensitive_value_2', 'sensitive_value_4']
    })
    
    profile1 = ar.profile(df1)
    profile2 = ar.profile(df2)
    comparison = ar.compare_profiles(profile1, profile2)
    
    result = comparison.to_dict(redact_sample_values=True)
    
    assert isinstance(result, dict)
    assert 'left_profile' in result
    assert 'right_profile' in result


def test_profile_comparison_to_dict_redact_sample_values_false(tmp_path):
    """Test to_dict with redact_sample_values=False (default)."""
    import arnio as ar
    
    df1 = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': ['value_a', 'value_b', 'value_c']
    })
    df2 = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': ['value_a', 'value_b', 'value_d']
    })
    
    profile1 = ar.profile(df1)
    profile2 = ar.profile(df2)
    comparison = ar.compare_profiles(profile1, profile2)
    
    result = comparison.to_dict(redact_sample_values=False)
    
    assert isinstance(result, dict)
    assert isinstance(result['left_profile'], dict)
    assert isinstance(result['right_profile'], dict)


def test_profile_comparison_to_dict_exclude_columns_list(tmp_path):
    """Test to_dict with exclude_columns as a list."""
    import arnio as ar
    
    df1 = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': ['a', 'b', 'c'],
        'col3': [True, False, True]
    })
    df2 = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': ['a', 'b', 'd'],
        'col3': [True, False, False]
    })
    
    profile1 = ar.profile(df1)
    profile2 = ar.profile(df2)
    comparison = ar.compare_profiles(profile1, profile2)
    
    result = comparison.to_dict(exclude_columns=['col2'])
    
    assert isinstance(result, dict)
    assert 'left_profile' in result
    assert 'right_profile' in result
    assert isinstance(result['status_counts'], dict)


def test_profile_comparison_to_dict_exclude_columns_set(tmp_path):
    """Test to_dict with exclude_columns as a set."""
    import arnio as ar
    
    df1 = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': ['a', 'b', 'c'],
        'col3': [True, False, True]
    })
    df2 = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': ['a', 'b', 'd'],
        'col3': [True, False, False]
    })
    
    profile1 = ar.profile(df1)
    profile2 = ar.profile(df2)
    comparison = ar.compare_profiles(profile1, profile2)
    
    result = comparison.to_dict(exclude_columns={'col2', 'col3'})
    
    assert isinstance(result, dict)
    assert 'left_profile' in result
    assert 'right_profile' in result


def test_profile_comparison_to_dict_exclude_columns_tuple(tmp_path):
    """Test to_dict with exclude_columns as a tuple."""
    import arnio as ar
    
    df1 = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': ['a', 'b', 'c']
    })
    df2 = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': ['a', 'b', 'd']
    })
    
    profile1 = ar.profile(df1)
    profile2 = ar.profile(df2)
    comparison = ar.compare_profiles(profile1, profile2)
    
    result = comparison.to_dict(exclude_columns=('col1',))
    
    assert isinstance(result, dict)
    assert 'left_profile' in result


def test_profile_comparison_to_dict_exclude_columns_none(tmp_path):
    """Test to_dict with exclude_columns=None (default)."""
    import arnio as ar
    
    df1 = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': ['a', 'b', 'c']
    })
    df2 = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': ['a', 'b', 'd']
    })
    
    profile1 = ar.profile(df1)
    profile2 = ar.profile(df2)
    comparison = ar.compare_profiles(profile1, profile2)
    
    result = comparison.to_dict(exclude_columns=None)
    
    assert isinstance(result, dict)
    assert 'left_profile' in result


def test_profile_comparison_to_dict_both_parameters(tmp_path):
    """Test to_dict with both redact_sample_values and exclude_columns."""
    import arnio as ar
    
    df1 = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': ['secret_a', 'secret_b', 'secret_c'],
        'col3': [10, 20, 30]
    })
    df2 = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': ['secret_a', 'secret_b', 'secret_d'],
        'col3': [10, 20, 35]
    })
    
    profile1 = ar.profile(df1)
    profile2 = ar.profile(df2)
    comparison = ar.compare_profiles(profile1, profile2)
    
    result = comparison.to_dict(
        redact_sample_values=True,
        exclude_columns=['col2']
    )
    
    assert isinstance(result, dict)
    assert 'status_counts' in result
    assert 'drift_report' in result


def test_profile_comparison_to_dict_json_serializable(tmp_path):
    """Test that to_dict output is JSON serializable."""
    import arnio as ar
    
    df1 = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': ['a', 'b', 'c']
    })
    df2 = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': ['a', 'b', 'd']
    })
    
    profile1 = ar.profile(df1)
    profile2 = ar.profile(df2)
    comparison = ar.compare_profiles(profile1, profile2)
    
    result = comparison.to_dict()
    
    # Must be JSON serializable
    json_str = json.dumps(result, default=str)
    assert isinstance(json_str, str)
    assert len(json_str) > 0
    
    # Verify round-trip
    parsed = json.loads(json_str)
    assert isinstance(parsed, dict)


def test_profile_comparison_to_dict_with_nan_and_none(tmp_path):
    """Test to_dict with DataFrames containing NaN and None values."""
    import arnio as ar
    
    df1 = pd.DataFrame({
        'col1': [1, 2, np.nan, 4],
        'col2': ['a', None, 'c', 'd']
    })
    df2 = pd.DataFrame({
        'col1': [1, 2, 3, 4],
        'col2': ['a', 'b', 'c', 'd']
    })
    
    profile1 = ar.profile(df1)
    profile2 = ar.profile(df2)
    comparison = ar.compare_profiles(profile1, profile2)
    
    result = comparison.to_dict()
    
    assert isinstance(result, dict)
    assert 'status_counts' in result
    
    # Should still be JSON serializable
    json_str = json.dumps(result, default=str)
    assert isinstance(json_str, str)


def test_profile_comparison_to_dict_numeric_types(tmp_path):
    """Test to_dict with various numeric data types."""
    import arnio as ar
    
    df1 = pd.DataFrame({
        'int_col': np.array([1, 2, 3], dtype=np.int32),
        'float_col': np.array([1.1, 2.2, 3.3], dtype=np.float32),
        'bool_col': [True, False, True]
    })
    df2 = pd.DataFrame({
        'int_col': np.array([1, 2, 4], dtype=np.int32),
        'float_col': np.array([1.1, 2.2, 3.4], dtype=np.float32),
        'bool_col': [True, False, False]
    })
    
    profile1 = ar.profile(df1)
    profile2 = ar.profile(df2)
    comparison = ar.compare_profiles(profile1, profile2)
    
    result = comparison.to_dict()
    
    assert isinstance(result, dict)
    json_str = json.dumps(result, default=str)
    assert isinstance(json_str, str)


def test_profile_comparison_to_dict_multiple_exclude_columns(tmp_path):
    """Test to_dict excluding multiple columns."""
    import arnio as ar
    
    df1 = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': ['a', 'b', 'c'],
        'col3': [True, False, True],
        'col4': [10, 20, 30]
    })
    df2 = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': ['a', 'b', 'd'],
        'col3': [True, False, False],
        'col4': [10, 20, 35]
    })
    
    profile1 = ar.profile(df1)
    profile2 = ar.profile(df2)
    comparison = ar.compare_profiles(profile1, profile2)
    
    result = comparison.to_dict(exclude_columns=['col2', 'col3', 'col4'])
    
    assert isinstance(result, dict)
    assert 'left_profile' in result
    assert 'right_profile' in result


def test_profile_comparison_to_dict_drift_report_content(tmp_path):
    """Test that drift_report is properly included in output."""
    import arnio as ar
    
    df1 = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': ['a', 'b', 'c']
    })
    df2 = pd.DataFrame({
        'col1': [4, 5, 6],
        'col2': ['x', 'y', 'z']
    })
    
    profile1 = ar.profile(df1)
    profile2 = ar.profile(df2)
    comparison = ar.compare_profiles(profile1, profile2)
    
    result = comparison.to_dict()
    
    assert isinstance(result['drift_report'], dict)


def test_profile_comparison_to_dict_status_counts_content(tmp_path):
    """Test that status_counts are properly included in output."""
    import arnio as ar
    
    df1 = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': ['a', 'b', 'c']
    })
    df2 = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': ['a', 'b', 'd']
    })
    
    profile1 = ar.profile(df1)
    profile2 = ar.profile(df2)
    comparison = ar.compare_profiles(profile1, profile2)
    
    result = comparison.to_dict()
    
    assert isinstance(result['status_counts'], dict)
    assert len(result['status_counts']) >= 0
