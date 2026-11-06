import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestCoalesceColumns:
    def test_basic_coalesce(self):
        """Test selecting first non-null value from multiple columns."""
        frame = ar.from_pandas(pd.DataFrame({
            'a': [1, None, 3],
            'b': [None, 2, None],
            'c': [10, 20, 30]
        }))
        result = ar.coalesce_columns(frame, subset=['a', 'b', 'c'], output_column='first')
        df = ar.to_pandas(result)
        assert df['first'].tolist() == [1.0, 2.0, 3.0]

    def test_coalesce_column_not_found(self):
        """Test KeyError when column in subset doesn't exist."""
        frame = ar.from_pandas(pd.DataFrame({'a': [1, 2]}))
        with pytest.raises(KeyError):
            ar.coalesce_columns(frame, subset=['a', 'missing'])


class TestCombineColumns:
    def test_combine_with_separator(self):
        """Test combining columns with custom separator."""
        frame = ar.from_pandas(pd.DataFrame({
            'first': ['John', 'Jane'],
            'last': ['Doe', 'Smith']
        }))
        result = ar.combine_columns(frame, subset=['first', 'last'],
                                   separator=' ', output_column='name')
        df = ar.to_pandas(result)
        assert 'name' in df.columns
        assert df['name'].tolist() == ['John Doe', 'Jane Smith']

    def test_combine_output_column_exists_error(self):
        """Test ValueError when output_column already exists."""
        frame = ar.from_pandas(pd.DataFrame({'a': [1], 'b': [2]}))
        with pytest.raises(ValueError):
            ar.combine_columns(frame, subset=['a'], output_column='a')


class TestDropColumnsMatching:
    def test_drop_matching_pattern(self):
        """Test dropping columns matching regex pattern."""
        frame = ar.from_pandas(pd.DataFrame({
            'temp_x': [1, 2],
            'temp_y': [3, 4],
            'keep': [5, 6]
        }))
        result = ar.drop_columns_matching(frame, r'^temp_')
        df = ar.to_pandas(result)
        assert df.columns.tolist() == ['keep']

    def test_drop_invalid_regex_error(self):
        """Test error with invalid regex pattern."""
        frame = ar.from_pandas(pd.DataFrame({'a': [1]}))
        with pytest.raises(Exception):  # re.error
            ar.drop_columns_matching(frame, r'[invalid')


class TestDropConstantColumns:
    def test_drop_constant_keeps_variable(self):
        """Test removing constant columns while preserving variable ones."""
        frame = ar.from_pandas(pd.DataFrame({
            'const_5': [5, 5, 5],
            'const_null': [None, None, None],
            'varied': [1, 2, 3]
        }))
        result = ar.drop_constant_columns(frame)
        df = ar.to_pandas(result)
        assert df.columns.tolist() == ['varied']

    def test_drop_constant_one_value_one_null(self):
        """Test that column with one value and nulls is kept (per docstring)."""
        frame = ar.from_pandas(pd.DataFrame({
            'val_and_null': [1, 1, None],
            'const': [5, 5, 5]
        }))
        result = ar.drop_constant_columns(frame)
        df = ar.to_pandas(result)
        assert 'val_and_null' in df.columns
        assert 'const' not in df.columns


class TestDropNulls:
    def test_drop_nulls_basic(self):
        """Test removing rows with nulls across all columns."""
        frame = ar.from_pandas(pd.DataFrame({
            'a': [1, None, 3],
            'b': [4, 5, 6]
        }))
        result = ar.drop_nulls(frame)
        df = ar.to_pandas(result)
        assert len(df) == 2
        assert df['a'].isna().sum() == 0

    def test_drop_nulls_subset(self):
        """Test drop_nulls only checks specified columns."""
        frame = ar.from_pandas(pd.DataFrame({
            'required': [1, 2, None],
            'optional': [None, 5, 6]
        }))
        result = ar.drop_nulls(frame, subset=['required'])
        df = ar.to_pandas(result)
        assert len(df) == 2
        assert df.loc[0, 'optional'] != df.loc[0, 'optional']  # NaN check


class TestFilterRows:
    def test_filter_rows_operators(self):
        """Test filtering with various comparison operators."""
        frame = ar.from_pandas(pd.DataFrame({'age': [18, 25, 30, 40]}))
        
        result_gt = ar.filter_rows(frame, column='age', op='>', value=25)
        assert len(ar.to_pandas(result_gt)) == 2
        
        result_eq = ar.filter_rows(frame, column='age', op='==', value=25)
        assert len(ar.to_pandas(result_eq)) == 1
        
        result_ne = ar.filter_rows(frame, column='age', op='!=', value=25)
        assert len(ar.to_pandas(result_ne)) == 3

    def test_filter_rows_string_column(self):
        """Test filtering on string columns."""
        frame = ar.from_pandas(pd.DataFrame({'city': ['NYC', 'LA', 'NYC']}))
        result = ar.filter_rows(frame, column='city', op='==', value='NYC')
        df = ar.to_pandas(result)
        assert len(df) == 2


class TestKeepRowsWithNulls:
    def test_keep_rows_with_nulls_basic(self):
        """Test keeping only rows with at least one null."""
        frame = ar.from_pandas(pd.DataFrame({
            'a': [1, None, 3, 4],
            'b': [4, 5, None, 7]
        }))
        result = ar.keep_rows_with_nulls(frame)
        df = ar.to_pandas(result)
        assert len(df) == 2  # Rows 1 and 2 have nulls

    def test_keep_rows_with_nulls_subset_type_error(self):
        """Test TypeError when subset is a string instead of list."""
        frame = ar.from_pandas(pd.DataFrame({'a': [1, None]}))
        with pytest.raises(TypeError):
            ar.keep_rows_with_nulls(frame, subset='a')


class TestReplaceValues:
    def test_replace_single_column(self):
        """Test replacing values in specific column."""
        frame = ar.from_pandas(pd.DataFrame({
            'status': ['active', 'inactive', 'active']
        }))
        result = ar.replace_values(frame, {'active': 'A', 'inactive': 'I'},
                                  column='status')
        df = ar.to_pandas(result)
        assert df['status'].tolist() == ['A', 'I', 'A']

    def test_replace_whole_frame(self):
        """Test replacing values across all columns when column=None."""
        frame = ar.from_pandas(pd.DataFrame({
            'a': ['old', 'old'],
            'b': ['old', 'new']
        }))
        result = ar.replace_values(frame, {'old': 'updated'})
        df = ar.to_pandas(result)
        # All 'old' values should be 'updated'
        assert (df == 'updated').sum().sum() >= 2

    def test_replace_null_handling(self):
        """Test mapping null values with None key."""
        frame = ar.from_pandas(pd.DataFrame({'val': [1, None, 3]}))
        result = ar.replace_values(frame, {None: 999}, column='val')
        df = ar.to_pandas(result)
        # Position 1 should now have 999 instead of NaN
        assert df.loc[1, 'val'] == 999


class TestSafeDivideColumns:
    def test_safe_divide_normal(self):
        """Test normal division of two numeric columns."""
        frame = ar.from_pandas(pd.DataFrame({
            'num': [10.0, 20.0, 30.0],
            'denom': [2.0, 4.0, 5.0]
        }))
        result = ar.safe_divide_columns(frame, numerator='num',
                                       denominator='denom',
                                       output_column='ratio')
        df = ar.to_pandas(result)
        assert np.allclose(df['ratio'], [5.0, 5.0, 6.0])

    def test_safe_divide_zero_denominator(self):
        """Test division by zero uses fill_value instead of NaN/Inf."""
        frame = ar.from_pandas(pd.DataFrame({
            'num': [10.0, 20.0],
            'denom': [2.0, 0.0]
        }))
        result = ar.safe_divide_columns(frame, numerator='num',
                                       denominator='denom',
                                       output_column='ratio',
                                       fill_value=0.0)
        df = ar.to_pandas(result)
        assert df['ratio'].tolist() == [5.0, 0.0]


class TestStandardizeMissingTokens:
    def test_standardize_default_tokens(self):
        """Test standardizing common missing-value tokens to NaN."""
        frame = ar.from_pandas(pd.DataFrame({
            'value': ['1', 'N/A', '3', 'NULL', '5']
        }))
        result = ar.standardize_missing_tokens(frame)
        df = ar.to_pandas(result)
        # Default behavior should catch N/A and NULL
        null_count = df['value'].isna().sum()
        assert null_count >= 2

    def test_standardize_case_insensitive(self):
        """Test case-insensitive token matching per docstring."""
        frame = ar.from_pandas(pd.DataFrame({
            'val': ['data', 'NONE', 'null', 'NaN']
        }))
        result = ar.standardize_missing_tokens(frame)
        df = ar.to_pandas(result)
        # Multiple variations should be caught
        assert df['val'].isna().sum() >= 2
