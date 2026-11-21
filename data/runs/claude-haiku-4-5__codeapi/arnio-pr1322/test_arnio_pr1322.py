import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestCoalesceColumns:
    """Tests for ar.coalesce_columns function"""
    
    def test_basic_coalesce_first_non_null(self):
        """Test basic coalescing selects first non-null value from subset columns"""
        df = pd.DataFrame({
            'a': [1, None, 3],
            'b': [None, 2, None],
            'c': [10, 20, 30]
        })
        result = ar.coalesce_columns(df, subset=['a', 'b', 'c'])
        
        assert result['coalesced'].iloc[0] == 1  # From 'a'
        assert result['coalesced'].iloc[1] == 2  # From 'b'
        assert result['coalesced'].iloc[2] == 3  # From 'a'
    
    def test_coalesce_all_nulls_returns_nan(self):
        """Test that rows with all null values in subset result in NaN"""
        df = pd.DataFrame({
            'x': [None, 1.0],
            'y': [None, None],
            'z': [None, 2.0]
        })
        result = ar.coalesce_columns(df, subset=['x', 'y', 'z'], output_column='out')
        
        assert pd.isna(result['out'].iloc[0])
        assert result['out'].iloc[1] == 1.0
    
    def test_coalesce_respects_column_order(self):
        """Test that column order determines precedence"""
        df = pd.DataFrame({
            'first': [5, None],
            'second': [10, 20]
        })
        result = ar.coalesce_columns(df, subset=['first', 'second'])
        
        assert result['coalesced'].iloc[0] == 5
        assert result['coalesced'].iloc[1] == 20
    
    def test_coalesce_custom_output_column_name(self):
        """Test custom output_column parameter"""
        df = pd.DataFrame({'a': [1], 'b': [2]})
        result = ar.coalesce_columns(df, subset=['a', 'b'], output_column='my_column')
        
        assert 'my_column' in result.columns
        assert result['my_column'].iloc[0] == 1
    
    def test_coalesce_mixed_dtypes(self):
        """Test coalescing works with mixed data types"""
        df = pd.DataFrame({
            'nums': [None, 42],
            'strs': ['text', None],
            'floats': [3.14, None]
        })
        result = ar.coalesce_columns(df, subset=['nums', 'strs', 'floats'])
        
        assert result['coalesced'].iloc[0] == 'text'
        assert result['coalesced'].iloc[1] == 42
    
    def test_coalesce_error_output_column_exists(self):
        """Test ValueError when output_column already exists in frame"""
        df = pd.DataFrame({'a': [1], 'existing': [2]})
        
        with pytest.raises(ValueError, match="already exists"):
            ar.coalesce_columns(df, subset=['a'], output_column='existing')
    
    def test_coalesce_error_empty_output_column(self):
        """Test ValueError for empty output_column name"""
        df = pd.DataFrame({'a': [1]})
        
        with pytest.raises(ValueError, match="non-empty"):
            ar.coalesce_columns(df, subset=['a'], output_column='')
    
    def test_coalesce_error_empty_subset(self):
        """Test ValueError for empty subset list"""
        df = pd.DataFrame({'a': [1]})
        
        with pytest.raises(ValueError, match="at least one"):
            ar.coalesce_columns(df, subset=[], output_column='out')


class TestStandardizeMissingTokens:
    """Tests for ar.standardize_missing_tokens function"""
    
    def test_default_missing_tokens_converted(self):
        """Test default tokens are converted to NaN"""
        df = pd.DataFrame({
            'col': ['valid', 'N/A', 'NA', 'n/a', 'nan', '-', 'none', 'null']
        })
        result = ar.standardize_missing_tokens(df)
        
        assert result['col'].iloc[0] == 'valid'
        for i in range(1, len(result)):
            assert pd.isna(result['col'].iloc[i])
    
    def test_custom_tokens_only(self):
        """Test custom tokens list replaces defaults"""
        df = pd.DataFrame({'col': ['real', 'CUSTOM', 'UNKNOWN', 'real2']})
        result = ar.standardize_missing_tokens(df, tokens=['CUSTOM', 'UNKNOWN'])
        
        assert result['col'].iloc[0] == 'real'
        assert pd.isna(result['col'].iloc[1])
        assert pd.isna(result['col'].iloc[2])
        assert result['col'].iloc[3] == 'real2'
    
    def test_subset_specific_columns(self):
        """Test subset parameter applies standardization only to specified columns"""
        df = pd.DataFrame({
            'untouched': ['N/A', 'value'],
            'changed': ['N/A', 'value']
        })
        result = ar.standardize_missing_tokens(df, subset=['changed'])
        
        assert result['untouched'].iloc[0] == 'N/A'
        assert pd.isna(result['changed'].iloc[0])
    
    def test_case_insensitive_matching(self):
        """Test token matching is case-insensitive"""
        df = pd.DataFrame({
            'col': ['N/A', 'n/a', 'N/a', 'NA', 'na', 'Na']
        })
        result = ar.standardize_missing_tokens(df, tokens=['n/a', 'na'])
        
        for val in result['col']:
            assert pd.isna(val)
    
    def test_whitespace_trimmed_before_matching(self):
        """Test whitespace is trimmed before token matching"""
        df = pd.DataFrame({
            'col': [' N/A ', '\tna\t', '  value  ', '   ']
        })
        result = ar.standardize_missing_tokens(df)
        
        assert pd.isna(result['col'].iloc[0])
        assert pd.isna(result['col'].iloc[1])
        assert result['col'].iloc[2] == '  value  '  # Internal whitespace preserved
        assert pd.isna(result['col'].iloc[3])  # Whitespace-only becomes empty string token
    
    def test_non_string_values_unchanged(self):
        """Test that non-string values are preserved"""
        df = pd.DataFrame({
            'col': [42, 3.14, None, 'N/A', 'text']
        })
        result = ar.standardize_missing_tokens(df)
        
        assert result['col'].iloc[0] == 42
        assert result['col'].iloc[1] == 3.14
        assert pd.isna(result['col'].iloc[2])
        assert pd.isna(result['col'].iloc[3])
        assert result['col'].iloc[4] == 'text'
    
    def test_error_subset_as_string(self):
        """Test TypeError when subset is a string instead of list"""
        df = pd.DataFrame({'col': ['value']})
        
        with pytest.raises(TypeError, match="must be a list"):
            ar.standardize_missing_tokens(df, subset='col')
    
    def test_error_tokens_as_string(self):
        """Test TypeError when tokens is a string instead of list"""
        df = pd.DataFrame({'col': ['value']})
        
        with pytest.raises(TypeError, match="must be a list"):
            ar.standardize_missing_tokens(df, tokens='N/A')
    
    def test_error_tokens_non_string_item(self):
        """Test TypeError when tokens list contains non-string values"""
        df = pd.DataFrame({'col': ['value']})
        
        with pytest.raises(TypeError, match="list of strings"):
            ar.standardize_missing_tokens(df, tokens=['N/A', 123])
