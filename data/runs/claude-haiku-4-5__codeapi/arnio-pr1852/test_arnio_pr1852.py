import pytest
import pandas as pd
import numpy as np
import arnio as ar
import re


class TestCoalesceColumns:
    def test_coalesce_first_non_null(self):
        """Test coalescing selects first non-null value from subset."""
        df = pd.DataFrame({
            'a': [1, None, 3],
            'b': [None, 2, None],
            'c': [4, 5, 6]
        })
        frame = ar.from_pandas(df)
        result = ar.coalesce_columns(frame, subset=['a', 'b', 'c'], output_column='coalesced')
        result_df = ar.to_pandas(result)
        assert result_df['coalesced'].tolist() == [1.0, 2.0, 3.0]
    
    def test_coalesce_output_column_exists_error(self):
        """Test error when output column already exists in frame."""
        df = pd.DataFrame({'a': [1], 'existing': [2]})
        frame = ar.from_pandas(df)
        with pytest.raises(ValueError, match="already exists"):
            ar.coalesce_columns(frame, subset=['a'], output_column='existing')


class TestCombineColumns:
    def test_combine_with_separator(self):
        """Test combining multiple columns with custom separator."""
        df = pd.DataFrame({
            'first': ['John', 'Jane'],
            'last': ['Doe', 'Smith']
        })
        frame = ar.from_pandas(df)
        result = ar.combine_columns(frame, subset=['first', 'last'], separator=' ', output_column='full')
        result_df = ar.to_pandas(result)
        assert result_df['full'].tolist() == ['John Doe', 'Jane Smith']
    
    def test_combine_all_columns_when_subset_none(self):
        """Test combining all columns when subset is None."""
        df = pd.DataFrame({'a': ['x'], 'b': ['y']})
        frame = ar.from_pandas(df)
        result = ar.combine_columns(frame, subset=None, separator='-', output_column='combined')
        result_df = ar.to_pandas(result)
        assert 'x-y' in result_df['combined'].values


class TestDropColumnsMatching:
    def test_drop_columns_by_regex_pattern(self):
        """Test dropping columns matching regex pattern."""
        df = pd.DataFrame({'temp_a': [1], 'temp_b': [2], 'keep': [3]})
        frame = ar.from_pandas(df)
        result = ar.drop_columns_matching(frame, '^temp_')
        result_df = ar.to_pandas(result)
        assert list(result_df.columns) == ['keep']
    
    def test_invalid_regex_pattern_raises_error(self):
        """Test that invalid regex raises re.error."""
        df = pd.DataFrame({'a': [1]})
        frame = ar.from_pandas(df)
        with pytest.raises(re.error):
            ar.drop_columns_matching(frame, '[invalid')


class TestDropConstantColumns:
    def test_drop_single_unique_value_columns(self):
        """Test dropping columns with exactly one unique value."""
        df = pd.DataFrame({
            'constant': [5, 5, 5],
            'varying': [1, 2, 3]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_constant_columns(frame)
        result_df = ar.to_pandas(result)
        assert 'constant' not in result_df.columns
        assert 'varying' in result_df.columns
    
    def test_all_nulls_column_is_constant(self):
        """Test that column with all nulls is considered constant (one unique value)."""
        df = pd.DataFrame({
            'all_null': [None, None, None],
            'has_value': [1, 2, 3]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_constant_columns(frame)
        result_df = ar.to_pandas(result)
        assert 'all_null' not in result_df.columns
        assert 'has_value' in result_df.columns


class TestDropNulls:
    def test_drop_rows_with_any_null(self):
        """Test dropping rows containing any null/empty values."""
        df = pd.DataFrame({
            'a': [1, 2, None, 4],
            'b': [5, None, 7, 8]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_nulls(frame)
        result_df = ar.to_pandas(result)
        assert len(result_df) == 2
        assert result_df['a'].tolist() == [1.0, 4.0]
    
    def test_drop_nulls_subset_columns_only(self):
        """Test dropping nulls checking only subset of specified columns."""
        df = pd.DataFrame({
            'a': [1, None, 3],
            'b': [None, 5, 6]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_nulls(frame, subset=['a'])
        result_df = ar.to_pandas(result)
        assert len(result_df) == 2
        assert None not in result_df['a'].values


class TestFilterRows:
    def test_filter_rows_greater_than_operator(self):
        """Test filtering rows with > operator."""
        df = pd.DataFrame({'age': [18, 25, 30]})
        frame = ar.from_pandas(df)
        result = ar.filter_rows(frame, 'age', '>', 20)
        result_df = ar.to_pandas(result)
        assert len(result_df) == 2
        assert result_df['age'].tolist() == [25, 30]
    
    def test_filter_rows_invalid_operator_error(self):
        """Test that unsupported operator raises ValueError."""
        df = pd.DataFrame({'age': [18]})
        frame = ar.from_pandas(df)
        with pytest.raises(ValueError, match="Unsupported"):
            ar.filter_rows(frame, 'age', '>>>', 20)


class TestKeepRowsWithNulls:
    def test_keep_rows_with_at_least_one_null(self):
        """Test keeping only rows with at least one null/empty value."""
        df = pd.DataFrame({
            'a': [1, None, 3, 4],
            'b': [4, 5, None, 8]
        })
        frame = ar.from_pandas(df)
        result = ar.keep_rows_with_nulls(frame)
        result_df = ar.to_pandas(result)
        assert len(result_df) == 2
    
    def test_keep_rows_subset_as_string_raises_typeerror(self):
        """Test that passing subset as string raises TypeError."""
        df = pd.DataFrame({'a': [1, None]})
        frame = ar.from_pandas(df)
        with pytest.raises(TypeError, match="must be a list"):
            ar.keep_rows_with_nulls(frame, subset='a')


class TestReplaceValues:
    def test_replace_in_specific_column(self):
        """Test replacing values in a specific column."""
        df = pd.DataFrame({
            'status': ['old', 'old', 'new'],
            'other': [1, 2, 3]
        })
        frame = ar.from_pandas(df)
        result = ar.replace_values(frame, {'old': 'updated'}, column='status')
        result_df = ar.to_pandas(result)
        assert result_df['status'].tolist() == ['updated', 'updated', 'new']
    
    def test_replace_across_all_columns_when_column_none(self):
        """Test replacing values across all columns when column=None."""
        df = pd.DataFrame({
            'a': [1, 2, 1],
            'b': [1, 3, 1]
        })
        frame = ar.from_pandas(df)
        result = ar.replace_values(frame, {1: 99}, column=None)
        result_df = ar.to_pandas(result)
        assert result_df.loc[0, 'a'] == 99
        assert result_df.loc[0, 'b'] == 99


class TestSafeDivideColumns:
    def test_safe_divide_with_zero_denominator_uses_fill_value(self):
        """Test safe division uses fill_value when denominator is zero."""
        df = pd.DataFrame({
            'numerator': [10, 20, 30],
            'denominator': [2, 0, 5]
        })
        frame = ar.from_pandas(df)
        result = ar.safe_divide_columns(frame, 'numerator', 'denominator', 'ratio', fill_value=0.0)
        result_df = ar.to_pandas(result)
        assert result_df['ratio'].tolist() == [5.0, 0.0, 6.0]
    
    def test_safe_divide_null_denominator_uses_fill_value(self):
        """Test safe division uses fill_value when denominator is null."""
        df = pd.DataFrame({
            'numerator': [10.0, 20.0],
            'denominator': [2.0, None]
        })
        frame = ar.from_pandas(df)
        result = ar.safe_divide_columns(frame, 'numerator', 'denominator', 'ratio', fill_value=-1.0)
        result_df = ar.to_pandas(result)
        assert result_df['ratio'].iloc[1] == -1.0


class TestStandardizeMissingTokens:
    def test_standardize_default_missing_tokens(self):
        """Test standardizing with default missing tokens (case-insensitive)."""
        df = pd.DataFrame({
            'value': ['N/A', 'valid', 'NA', 'none', 'NULL', '']
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame)
        result_df = ar.to_pandas(result)
        assert pd.isna(result_df['value'].iloc[0])
        assert result_df['value'].iloc[1] == 'valid'
        assert pd.isna(result_df['value'].iloc[2])
        assert pd.isna(result_df['value'].iloc[3])
        assert pd.isna(result_df['value'].iloc[4])
        assert pd.isna(result_df['value'].iloc[5])
    
    def test_standardize_custom_tokens_only(self):
        """Test standardizing with custom tokens list."""
        df = pd.DataFrame({
            'value': ['MISSING', 'valid', 'OTHER']
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame, tokens=['MISSING', 'OTHER'])
        result_df = ar.to_pandas(result)
        assert pd.isna(result_df['value'].iloc[0])
        assert result_df['value'].iloc[1] == 'valid'
        assert pd.isna(result_df['value'].iloc[2])
    
    def test_standardize_subset_columns_only(self):
        """Test standardizing only specific columns via subset parameter."""
        df = pd.DataFrame({
            'col1': ['N/A', 'value'],
            'col2': ['N/A', 'keep']
        })
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame, subset=['col1'])
        result_df = ar.to_pandas(result)
        assert pd.isna(result_df['col1'].iloc[0])
        assert result_df['col2'].iloc[0] == 'N/A'
