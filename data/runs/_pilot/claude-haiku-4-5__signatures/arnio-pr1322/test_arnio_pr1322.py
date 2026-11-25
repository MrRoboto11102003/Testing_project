import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestCoalesceColumns:
    """Tests for arnio.cleaning.coalesce_columns"""

    def test_coalesce_basic_with_nans(self):
        """Test basic coalesce functionality with NaN values"""
        df = pd.DataFrame({
            'a': [1, np.nan, 3],
            'b': [np.nan, 2, np.nan],
            'c': [4, 5, 6]
        })
        result = ar.cleaning.coalesce_columns(df, subset=['a', 'b', 'c'])
        expected = pd.Series([1.0, 2.0, 3.0], name='coalesced')
        pd.testing.assert_series_equal(result, expected)

    def test_coalesce_custom_output_column(self):
        """Test coalesce with custom output column name"""
        df = pd.DataFrame({
            'x': [1, np.nan],
            'y': [np.nan, 2]
        })
        result = ar.cleaning.coalesce_columns(
            df, subset=['x', 'y'], output_column='merged'
        )
        assert result.name == 'merged'
        assert list(result) == [1.0, 2.0]

    def test_coalesce_all_missing(self):
        """Test coalesce when all values in subset are NaN"""
        df = pd.DataFrame({
            'a': [np.nan, np.nan],
            'b': [np.nan, np.nan]
        })
        result = ar.cleaning.coalesce_columns(df, subset=['a', 'b'])
        assert result.isna().all()

    def test_coalesce_single_column(self):
        """Test coalesce with single column in subset"""
        df = pd.DataFrame({'col': [1, 2, 3]})
        result = ar.cleaning.coalesce_columns(df, subset=['col'])
        pd.testing.assert_series_equal(result, df['col'].rename('coalesced'))

    def test_coalesce_takes_first_non_null(self):
        """Test that first non-null value in column order is selected"""
        df = pd.DataFrame({
            'first': [np.nan, 10, np.nan],
            'second': [99, 99, 99],
            'third': [88, 88, 88]
        })
        result = ar.cleaning.coalesce_columns(
            df, subset=['first', 'second', 'third']
        )
        expected = pd.Series([99.0, 10.0, 99.0], name='coalesced')
        pd.testing.assert_series_equal(result, expected)

    def test_coalesce_with_strings(self):
        """Test coalesce with string data"""
        df = pd.DataFrame({
            'a': ['x', None, 'z'],
            'b': [None, 'y', None]
        })
        result = ar.cleaning.coalesce_columns(df, subset=['a', 'b'])
        expected = pd.Series(['x', 'y', 'z'], name='coalesced')
        pd.testing.assert_series_equal(result, expected)

    def test_coalesce_preserves_dtype_string(self):
        """Test that coalesce preserves string type when possible"""
        df = pd.DataFrame({
            'a': ['hello', None],
            'b': [None, 'world']
        })
        result = ar.cleaning.coalesce_columns(df, subset=['a', 'b'])
        assert result.dtype == 'object'


class TestStandardizeMissingTokens:
    """Tests for arnio.cleaning.standardize_missing_tokens"""

    def test_standardize_common_tokens_default(self):
        """Test standardization of common missing value tokens with defaults"""
        df = pd.DataFrame({
            'col': ['value', 'NA', 'null', 'N/A']
        })
        result = ar.cleaning.standardize_missing_tokens(df)
        assert result.loc[0, 'col'] == 'value'
        assert pd.isna(result.loc[1, 'col'])
        assert pd.isna(result.loc[2, 'col'])
        assert pd.isna(result.loc[3, 'col'])

    def test_standardize_custom_tokens_list(self):
        """Test standardization with custom token list"""
        df = pd.DataFrame({
            'col': ['ok', 'MISSING', 'SKIP', 'NA']
        })
        result = ar.cleaning.standardize_missing_tokens(
            df, tokens=['MISSING', 'SKIP']
        )
        assert result.loc[0, 'col'] == 'ok'
        assert pd.isna(result.loc[1, 'col'])
        assert pd.isna(result.loc[2, 'col'])
        assert result.loc[3, 'col'] == 'NA'

    def test_standardize_subset_columns_only(self):
        """Test standardization applied only to subset of columns"""
        df = pd.DataFrame({
            'clean_me': ['value', 'NA'],
            'leave_me': ['NA', 'value']
        })
        result = ar.cleaning.standardize_missing_tokens(
            df, subset=['clean_me']
        )
        assert pd.isna(result.loc[1, 'clean_me'])
        assert result.loc[0, 'leave_me'] == 'NA'
        assert result.loc[1, 'leave_me'] == 'value'

    def test_standardize_all_columns_when_subset_none(self):
        """Test that all columns are standardized when subset is None"""
        df = pd.DataFrame({
            'a': ['NA', 'data'],
            'b': ['null', 'info']
        })
        result = ar.cleaning.standardize_missing_tokens(df, subset=None)
        assert pd.isna(result.loc[0, 'a'])
        assert pd.isna(result.loc[0, 'b'])
        assert result.loc[1, 'a'] == 'data'

    def test_standardize_preserves_real_values(self):
        """Test that non-token values are preserved"""
        df = pd.DataFrame({
            'col': [1, 'text', 2.5, True]
        })
        result = ar.cleaning.standardize_missing_tokens(df)
        assert result.loc[0, 'col'] == 1
        assert result.loc[1, 'col'] == 'text'
        assert result.loc[2, 'col'] == 2.5

    def test_standardize_empty_dataframe(self):
        """Test standardization on empty DataFrame"""
        df = pd.DataFrame()
        result = ar.cleaning.standardize_missing_tokens(df)
        assert len(result) == 0

    def test_standardize_multiple_columns_subset(self):
        """Test standardization across multiple columns with subset"""
        df = pd.DataFrame({
            'x': ['valid', 'NA', 'data'],
            'y': [1, 2, 3],
            'z': ['good', 'null', 'info']
        })
        result = ar.cleaning.standardize_missing_tokens(df, subset=['x', 'z'])
        assert result.loc[0, 'x'] == 'valid'
        assert pd.isna(result.loc[1, 'x'])
        assert result.loc[2, 'y'] == 3
        assert pd.isna(result.loc[1, 'z'])

    def test_standardize_tokens_none_uses_default(self):
        """Test that tokens=None uses default missing value tokens"""
        df = pd.DataFrame({
            'col': ['data', 'NA', 'custom']
        })
        result = ar.cleaning.standardize_missing_tokens(df, tokens=None)
        assert result.loc[0, 'col'] == 'data'
        assert pd.isna(result.loc[1, 'col'])
        assert result.loc[2, 'col'] == 'custom'
