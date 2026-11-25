import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestCompareProfiles:
    """Test suite for arnio.quality.compare_profiles function."""
    
    @pytest.fixture
    def profile_simple_a(self):
        """Create a simple quality report for profile A."""
        df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'age': [25, 30, 35, 40, 45]
        })
        return ar.profile(df)
    
    @pytest.fixture
    def profile_simple_b(self):
        """Create a simple quality report for profile B."""
        df = pd.DataFrame({
            'id': [10, 20, 30],
            'name': ['Frank', 'Grace', 'Henry'],
            'age': [28, 32, 38]
        })
        return ar.profile(df)
    
    def test_compare_identical_profiles(self, profile_simple_a):
        """Test comparing a profile with itself."""
        result = ar.compare_profiles(profile_simple_a, profile_simple_a)
        assert result is not None
    
    def test_compare_different_profiles(self, profile_simple_a, profile_simple_b):
        """Test comparing two different profiles."""
        result = ar.compare_profiles(profile_simple_a, profile_simple_b)
        assert result is not None
    
    def test_returns_profile_comparison_type(self, profile_simple_a, profile_simple_b):
        """Test that result is a ProfileComparison object."""
        result = ar.compare_profiles(profile_simple_a, profile_simple_b)
        assert hasattr(result, '__class__')
        assert result.__class__.__name__ == 'ProfileComparison'
    
    def test_compare_profiles_different_columns(self):
        """Test comparing profiles with different column sets."""
        df_a = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        df_b = pd.DataFrame({'col1': [7, 8, 9], 'col3': [10, 11, 12]})
        profile_a = ar.profile(df_a)
        profile_b = ar.profile(df_b)
        result = ar.compare_profiles(profile_a, profile_b)
        assert result is not None
    
    def test_compare_empty_dataframes(self):
        """Test comparing profiles of empty dataframes."""
        df_a = pd.DataFrame({'col1': pd.Series([], dtype=int)})
        df_b = pd.DataFrame({'col1': pd.Series([], dtype=int)})
        profile_a = ar.profile(df_a)
        profile_b = ar.profile(df_b)
        result = ar.compare_profiles(profile_a, profile_b)
        assert result is not None
    
    def test_compare_single_row_dataframes(self):
        """Test comparing profiles of single-row dataframes."""
        df_a = pd.DataFrame({'id': [1], 'value': [100]})
        df_b = pd.DataFrame({'id': [2], 'value': [200]})
        profile_a = ar.profile(df_a)
        profile_b = ar.profile(df_b)
        result = ar.compare_profiles(profile_a, profile_b)
        assert result is not None
    
    def test_compare_with_null_values(self):
        """Test comparing profiles with different null patterns."""
        df_a = pd.DataFrame({'col1': [1, None, 3, 4]})
        df_b = pd.DataFrame({'col1': [1, 2, None, 4]})
        profile_a = ar.profile(df_a)
        profile_b = ar.profile(df_b)
        result = ar.compare_profiles(profile_a, profile_b)
        assert result is not None
    
    def test_compare_all_nulls_column(self):
        """Test comparing profile where a column is entirely null."""
        df_a = pd.DataFrame({'col1': [None, None, None]})
        df_b = pd.DataFrame({'col1': [1, 2, 3]})
        profile_a = ar.profile(df_a)
        profile_b = ar.profile(df_b)
        result = ar.compare_profiles(profile_a, profile_b)
        assert result is not None
    
    def test_compare_numeric_dtypes(self):
        """Test comparing profiles with different numeric types."""
        df_a = pd.DataFrame({'col1': [1, 2, 3]})
        df_b = pd.DataFrame({'col1': [1.5, 2.5, 3.5]})
        profile_a = ar.profile(df_a)
        profile_b = ar.profile(df_b)
        result = ar.compare_profiles(profile_a, profile_b)
        assert result is not None
    
    def test_compare_string_columns(self):
        """Test comparing profiles with string columns."""
        df_a = pd.DataFrame({'text': ['apple', 'banana', 'cherry']})
        df_b = pd.DataFrame({'text': ['dog', 'elephant', 'fox']})
        profile_a = ar.profile(df_a)
        profile_b = ar.profile(df_b)
        result = ar.compare_profiles(profile_a, profile_b)
        assert result is not None
    
    def test_compare_mixed_dtypes(self):
        """Test comparing profiles with mixed data types."""
        df_a = pd.DataFrame({
            'int_col': [1, 2, 3],
            'str_col': ['a', 'b', 'c'],
            'float_col': [1.1, 2.2, 3.3]
        })
        df_b = pd.DataFrame({
            'int_col': [4, 5, 6],
            'str_col': ['x', 'y', 'z'],
            'float_col': [4.4, 5.5, 6.6]
        })
        profile_a = ar.profile(df_a)
        profile_b = ar.profile(df_b)
        result = ar.compare_profiles(profile_a, profile_b)
        assert result is not None
    
    def test_compare_different_row_counts(self):
        """Test comparing profiles with significantly different row counts."""
        df_a = pd.DataFrame({'col1': range(100)})
        df_b = pd.DataFrame({'col1': range(5)})
        profile_a = ar.profile(df_a)
        profile_b = ar.profile(df_b)
        result = ar.compare_profiles(profile_a, profile_b)
        assert result is not None
    
    def test_compare_large_dataframes(self):
        """Test comparing profiles of larger dataframes."""
        df_a = pd.DataFrame({
            'id': range(1000),
            'value': np.random.randn(1000),
            'category': np.random.choice(['A', 'B', 'C'], 1000)
        })
        df_b = pd.DataFrame({
            'id': range(500, 1500),
            'value': np.random.randn(1000),
            'category': np.random.choice(['A', 'B', 'C'], 1000)
        })
        profile_a = ar.profile(df_a)
        profile_b = ar.profile(df_b)
        result = ar.compare_profiles(profile_a, profile_b)
        assert result is not None
    
    def test_compare_categorical_columns(self):
        """Test comparing profiles with categorical data."""
        df_a = pd.DataFrame({'category': pd.Categorical(['A', 'B', 'A', 'C'])})
        df_b = pd.DataFrame({'category': pd.Categorical(['X', 'Y', 'Z', 'X'])})
        profile_a = ar.profile(df_a)
        profile_b = ar.profile(df_b)
        result = ar.compare_profiles(profile_a, profile_b)
        assert result is not None
