import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestDropColumnsMatching:
    """Tests for arnio.cleaning.drop_columns_matching function."""
    
    def test_drop_single_exact_column_match(self):
        """Drop a single column matching exact pattern."""
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['a', 'b', 'c'],
            'age': [20, 25, 30]
        })
        result = ar.cleaning.drop_columns_matching(df, 'age')
        assert 'age' not in result.columns
        assert list(result.columns) == ['id', 'name']
    
    def test_drop_multiple_columns_with_prefix_pattern(self):
        """Drop multiple columns matching a prefix pattern."""
        df = pd.DataFrame({
            'id_1': [1, 2],
            'id_2': [3, 4],
            'name': ['a', 'b']
        })
        result = ar.cleaning.drop_columns_matching(df, 'id_')
        assert list(result.columns) == ['name']
        assert result.shape == (2, 1)
    
    def test_no_columns_match_pattern(self):
        """When no columns match, return unchanged DataFrame."""
        df = pd.DataFrame({
            'name': ['a', 'b'],
            'age': [20, 25]
        })
        result = ar.cleaning.drop_columns_matching(df, 'nonexistent')
        pd.testing.assert_frame_equal(result, df)
    
    def test_all_columns_match_pattern(self):
        """When all columns match pattern, return empty DataFrame with rows."""
        df = pd.DataFrame({
            'col_1': [1, 2],
            'col_2': [3, 4],
            'col_3': [5, 6]
        })
        result = ar.cleaning.drop_columns_matching(df, 'col_')
        assert result.shape[1] == 0
        assert result.shape[0] == 3
    
    def test_regex_wildcard_pattern(self):
        """Pattern with .* regex wildcard matches variable endings."""
        df = pd.DataFrame({
            'temp_value': [1],
            'temp_flag': [2],
            'value': [3]
        })
        result = ar.cleaning.drop_columns_matching(df, 'temp_.*')
        assert list(result.columns) == ['value']
    
    def test_case_sensitive_matching(self):
        """Pattern matching respects case sensitivity."""
        df = pd.DataFrame({
            'Name': [1],
            'name': [2],
            'NAME': [3]
        })
        result = ar.cleaning.drop_columns_matching(df, 'name')
        assert 'name' not in result.columns
        assert 'Name' in result.columns
        assert 'NAME' in result.columns
    
    def test_empty_dataframe(self):
        """Handles empty DataFrame correctly."""
        df = pd.DataFrame()
        result = ar.cleaning.drop_columns_matching(df, 'pattern')
        assert result.shape == (0, 0)
    
    def test_single_column_matching(self):
        """Works correctly with single column DataFrame."""
        df = pd.DataFrame({'col': [1, 2, 3]})
        result = ar.cleaning.drop_columns_matching(df, 'col')
        assert result.shape == (3, 0)
    
    def test_data_preservation_in_kept_columns(self):
        """Non-matching columns preserve their data values."""
        df = pd.DataFrame({
            'id': [1, 2],
            'value': [10, 20],
            'temporary': [100, 200]
        })
        result = ar.cleaning.drop_columns_matching(df, 'temp')
        pd.testing.assert_series_equal(result['id'], df['id'])
        pd.testing.assert_series_equal(result['value'], df['value'])
    
    def test_regex_start_anchor(self):
        """Pattern with ^ anchor only matches column name start."""
        df = pd.DataFrame({
            'prefix_a': [1],
            'prefix_b': [2],
            'middle_prefix': [3]
        })
        result = ar.cleaning.drop_columns_matching(df, '^prefix')
        assert list(result.columns) == ['middle_prefix']
    
    def test_regex_end_anchor(self):
        """Pattern with $ anchor only matches column name end."""
        df = pd.DataFrame({
            'a_suffix': [1],
            'b_suffix': [2],
            'suffix_c': [3]
        })
        result = ar.cleaning.drop_columns_matching(df, 'suffix$')
        assert list(result.columns) == ['suffix_c']
    
    def test_original_dataframe_not_modified(self):
        """Function does not modify the original DataFrame."""
        df = pd.DataFrame({
            'keep': [1, 2],
            'drop': [3, 4]
        })
        original_cols = list(df.columns)
        _ = ar.cleaning.drop_columns_matching(df, 'drop')
        assert list(df.columns) == original_cols
        assert 'drop' in df.columns
    
    def test_regex_alternation_pattern(self):
        """Pattern with alternation (|) drops multiple column types."""
        df = pd.DataFrame({
            'id': [1],
            'temp': [2],
            'flag': [3],
            'value': [4]
        })
        result = ar.cleaning.drop_columns_matching(df, 'id|temp|flag')
        assert list(result.columns) == ['value']
    
    def test_substring_pattern_match(self):
        """Pattern matches substring within column names."""
        df = pd.DataFrame({
            'user_id': [1],
            'item_id': [2],
            'name': [3]
        })
        result = ar.cleaning.drop_columns_matching(df, '_id')
        assert list(result.columns) == ['name']
    
    def test_pattern_with_special_regex_characters(self):
        """Handles patterns with escaped special regex characters."""
        df = pd.DataFrame({
            'col.1': [1],
            'col.2': [2],
            'col_1': [3]
        })
        result = ar.cleaning.drop_columns_matching(df, r'col\.')
        assert 'col.1' not in result.columns
        assert 'col.2' not in result.columns
        assert 'col_1' in result.columns
