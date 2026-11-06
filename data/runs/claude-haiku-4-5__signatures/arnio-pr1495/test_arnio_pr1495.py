import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestQualityProfile:
    """Unit tests for arnio.quality.profile function"""
    
    @pytest.fixture
    def basic_df(self):
        """Basic test dataframe with mixed types"""
        return pd.DataFrame({
            'integers': [1, 2, 3, 4, 5],
            'floats': [1.1, 2.2, 3.3, 4.4, 5.5],
            'strings': ['a', 'b', 'c', 'a', 'b'],
            'all_null': [None, None, None, None, None]
        })
    
    @pytest.fixture
    def df_with_nulls(self):
        """Dataframe with scattered missing values"""
        return pd.DataFrame({
            'col_a': [1, 2, None, 4, 5],
            'col_b': ['x', None, 'y', 'z', None],
            'col_c': [1.1, 2.2, 3.3, None, 5.5]
        })
    
    @pytest.fixture
    def large_df(self):
        """Larger dataframe for sampling and approximation tests"""
        np.random.seed(42)
        return pd.DataFrame({
            'numeric': np.random.randint(0, 100, 1000),
            'category': np.random.choice(['A', 'B', 'C', 'D', 'E'], 1000),
            'continuous': np.random.randn(1000)
        })
    
    def test_profile_basic(self, basic_df):
        """Test basic profile functionality"""
        result = ar.quality.profile(basic_df)
        assert result is not None
    
    def test_profile_returns_data_quality_report(self, basic_df):
        """Test that profile returns DataQualityReport object"""
        result = ar.quality.profile(basic_df)
        assert 'DataQualityReport' in type(result).__name__
    
    def test_profile_exclude_columns_multiple(self, basic_df):
        """Test exclude_columns parameter with multiple columns"""
        result = ar.quality.profile(
            basic_df,
            exclude_columns=['strings', 'all_null']
        )
        assert result is not None
    
    def test_profile_exclude_columns_single(self, basic_df):
        """Test excluding a single column"""
        result = ar.quality.profile(basic_df, exclude_columns=['integers'])
        assert result is not None
    
    def test_profile_exclude_all_columns(self, basic_df):
        """Test excluding all columns from profiling"""
        result = ar.quality.profile(
            basic_df,
            exclude_columns=list(basic_df.columns)
        )
        assert result is not None
    
    def test_profile_exclude_columns_none(self, basic_df):
        """Test with exclude_columns=None (default)"""
        result = ar.quality.profile(basic_df, exclude_columns=None)
        assert result is not None
    
    def test_profile_sample_size_small(self, large_df):
        """Test with small sample_size"""
        result = ar.quality.profile(large_df, sample_size=2)
        assert result is not None
    
    def test_profile_sample_size_large(self, large_df):
        """Test with large sample_size"""
        result = ar.quality.profile(large_df, sample_size=500)
        assert result is not None
    
    def test_profile_approx_top_values_disabled(self, large_df):
        """Test with approx_top_values=False"""
        result = ar.quality.profile(large_df, approx_top_values=False)
        assert result is not None
    
    def test_profile_approx_top_values_enabled(self, large_df):
        """Test with approx_top_values=True"""
        result = ar.quality.profile(large_df, approx_top_values=True)
        assert result is not None
    
    def test_profile_all_approximation_parameters(self, large_df):
        """Test all approximation-related parameters together"""
        result = ar.quality.profile(
            large_df,
            approx_top_values=True,
            approx_top_values_min_unique=800,
            approx_top_values_min_ratio=0.25,
            approx_top_values_sample_size=1500
        )
        assert result is not None
    
    def test_profile_empty_dataframe(self):
        """Test profiling an empty dataframe"""
        df = pd.DataFrame()
        result = ar.quality.profile(df)
        assert result is not None
    
    def test_profile_single_row(self):
        """Test profiling a single-row dataframe"""
        df = pd.DataFrame({'col_a': [1], 'col_b': ['x']})
        result = ar.quality.profile(df)
        assert result is not None
    
    def test_profile_all_null_column(self):
        """Test profiling with entirely null column"""
        df = pd.DataFrame({
            'all_nulls': [None] * 10,
            'some_data': list(range(10))
        })
        result = ar.quality.profile(df)
        assert result is not None
