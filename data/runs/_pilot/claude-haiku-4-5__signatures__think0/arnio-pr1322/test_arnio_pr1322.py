import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestCoalesceColumns:
    """Tests for arnio.cleaning.coalesce_columns"""
    
    def test_coalesce_basic(self):
        """Test basic coalescing of columns with missing values"""
        df = pd.DataFrame({
            'a': [1, np.nan, 3],
            'b': [np.nan, 2, np.nan],
            'c': [10, 20, 30]
        })
        result = ar.cleaning.coalesce_columns(df, subset=['a', 'b', 'c'])
        assert result['coalesced'].tolist() == [1.0, 2.0, 3.0]
    
    def test_coalesce_custom_output_column(self):
        """Test coalescing with custom output column name"""
        df = pd.DataFrame({
            'x': [np.nan, 5, np.nan],
            'y': [1, np.nan, 3]
        })
        result = ar.cleaning.coalesce_columns(
            df, subset=['x', 'y'], output_column='merged'
        )
        assert 'merged' in result.columns
        assert result['merged'].tolist() == [1.0, 5.0, 3.0]
    
    def test_coalesce_single_column(self):
        """Test coalescing with a single column"""
        df = pd.DataFrame({'a': [1, 2, 3]})
        result = ar.cleaning.coalesce_columns(df, subset=['a'])
        assert result['coalesced'].tolist() == [1, 2, 3]
    
    def test_coalesce_all_missing(self):
        """Test coalescing when all values are missing"""
        df = pd.DataFrame({
            'a': [np.nan, np.nan, np.nan],
            'b': [np.nan, np.nan, np.nan]
        })
        result = ar.cleaning.coalesce_columns(df, subset=['a', 'b'])
        assert result['coalesced'].isna().all()
    
    def test_coalesce_preserves_order(self):
        """Test that coalescing takes first non-null value in column order"""
        df = pd.DataFrame({
            'first': [np.nan, 10, np.nan],
            'second': [1, 20, np.nan],
            'third': [100, 200, 300]
        })
        result = ar.cleaning.coalesce_columns(
            df, subset=['first', 'second', 'third']
        )
        assert result['coalesced'].tolist() == [1.0, 10.0, 300.0]
    
    def test_coalesce_string_values(self):
        """Test coalescing string values"""
        df = pd.DataFrame({
            'a': [np.nan, 'hello', np.nan],
            'b': ['foo', np.nan, 'bar']
        })
        result = ar.cleaning.coalesce_columns(df, subset=['a', 'b'])
        assert result['coalesced'].tolist() == ['foo', 'hello', 'bar']
    
    def test_coalesce_mixed_types(self):
        """Test coalescing with mixed numeric and object types"""
        df = pd.DataFrame({
            'nums': [1.0, np.nan, 3.0],
            'strs': [np.nan, 'two', np.nan]
        })
        result = ar.cleaning.coalesce_columns(df, subset=['nums', 'strs'])
        assert result['coalesced'].iloc[0] == 1.0
        assert result['coalesced'].iloc[1] == 'two'
        assert result['coalesced'].iloc[2] == 3.0
    
    def test_coalesce_preserves_other_columns(self):
        """Test that coalescing preserves non-subset columns"""
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'col1': [np.nan, 5, np.nan],
            'col2': [10, np.nan, 20]
        })
        result = ar.cleaning.coalesce_columns(
            df, subset=['col1', 'col2']
        )
        assert 'id' in result.columns
        assert result['id'].tolist() == [1, 2, 3]
    
    def test_coalesce_empty_dataframe(self):
        """Test coalescing on empty dataframe"""
        df = pd.DataFrame({'a': [], 'b': []})
        result = ar.cleaning.coalesce_columns(df, subset=['a', 'b'])
        assert len(result) == 0
        assert 'coalesced' in result.columns


class TestStandardizeMissingTokens:
    """Tests for arnio.cleaning.standardize_missing_tokens"""
    
    def test_standardize_default_tokens(self):
        """Test standardizing with default missing tokens"""
        df = pd.DataFrame({
            'col': ['NA', 'N/A', 'value', 'none', 'None']
        })
        result = ar.cleaning.standardize_missing_tokens(df)
        # All missing token variations should be converted to np.nan
        assert pd.isna(result['col'].iloc[0])
        assert pd.isna(result['col'].iloc[1])
        assert result['col'].iloc[2] == 'value'
        assert pd.isna(result['col'].iloc[3]) or pd.isna(result['col'].iloc[4])
    
    def test_standardize_custom_tokens(self):
        """Test standardizing with custom missing tokens"""
        df = pd.DataFrame({
            'col': ['MISSING', 'value', 'NULL', 'data']
        })
        result = ar.cleaning.standardize_missing_tokens(
            df, tokens=['MISSING', 'NULL']
        )
        assert pd.isna(result['col'].iloc[0])
        assert result['col'].iloc[1] == 'value'
        assert pd.isna(result['col'].iloc[2])
        assert result['col'].iloc[3] == 'data'
    
    def test_standardize_subset_columns(self):
        """Test standardizing only a subset of columns"""
        df = pd.DataFrame({
            'col1': ['NA', 'value'],
            'col2': ['NA', 'data']
        })
        result = ar.cleaning.standardize_missing_tokens(
            df, subset=['col1']
        )
        assert pd.isna(result['col1'].iloc[0])
        assert result['col2'].iloc[0] == 'NA'
    
    def test_standardize_numeric_dataframe(self):
        """Test standardizing numeric dataframe"""
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4.5, 6.7, 8.9]
        })
        result = ar.cleaning.standardize_missing_tokens(df)
        pd.testing.assert_frame_equal(df, result)
    
    def test_standardize_mixed_types(self):
        """Test standardizing dataframe with mixed types"""
        df = pd.DataFrame({
            'str_col': ['NA', 'value'],
            'num_col': [1, 2]
        })
        result = ar.cleaning.standardize_missing_tokens(df)
        assert pd.isna(result['str_col'].iloc[0])
        assert result['num_col'].iloc[0] == 1
    
    def test_standardize_empty_dataframe(self):
        """Test standardizing empty dataframe"""
        df = pd.DataFrame({'a': [], 'b': []})
        result = ar.cleaning.standardize_missing_tokens(df)
        assert len(result) == 0
    
    def test_standardize_case_sensitivity(self):
        """Test that token matching respects case"""
        df = pd.DataFrame({
            'col': ['na', 'NA', 'value']
        })
        result = ar.cleaning.standardize_missing_tokens(
            df, tokens=['NA']
        )
        # 'NA' should be replaced but 'na' should not (case-sensitive)
        assert result['col'].iloc[0] == 'na'
        assert pd.isna(result['col'].iloc[1])
