import pytest
import pandas as pd
import numpy as np
import arnio as ar


def test_coalesce_columns_basic():
    """Test basic coalesce_columns with first non-null values."""
    df = pd.DataFrame({
        'a': [1, None, 3],
        'b': [None, 2, None],
        'c': [10, 20, 30]
    })
    result = ar.coalesce_columns(df, subset=['a', 'b', 'c'], output_column='coalesced')
    assert result['coalesced'].tolist() == [1, 2, 3]


def test_coalesce_columns_output_exists():
    """Test coalesce_columns raises when output_column already exists."""
    df = pd.DataFrame({'a': [1], 'b': [2]})
    with pytest.raises(ValueError, match="already exists"):
        ar.coalesce_columns(df, subset=['a'], output_column='a')


def test_combine_columns_with_separator():
    """Test combine_columns with custom separator."""
    df = pd.DataFrame({
        'first': ['John', 'Jane'],
        'last': ['Doe', 'Smith']
    })
    result = ar.combine_columns(df, subset=['first', 'last'], separator=' ', output_column='full_name')
    assert result['full_name'].tolist() == ['John Doe', 'Jane Smith']


def test_combine_columns_all_nulls():
    """Test combine_columns produces null when all values in row are null."""
    df = pd.DataFrame({
        'a': ['x', None],
        'b': ['y', None]
    })
    result = ar.combine_columns(df, subset=['a', 'b'], separator='-', output_column='combined')
    assert pd.isna(result['combined'].iloc[1])


def test_drop_columns_matching_regex():
    """Test drop_columns_matching with regex pattern."""
    df = pd.DataFrame({
        'temp_a': [1],
        'temp_b': [2],
        'keep_me': [3]
    })
    result = ar.drop_columns_matching(df, r'^temp_')
    assert list(result.columns) == ['keep_me']


def test_drop_columns_matching_all_error():
    """Test drop_columns_matching raises error when pattern matches all columns."""
    df = pd.DataFrame({'a': [1], 'b': [2]})
    with pytest.raises(ValueError, match="matches all columns"):
        ar.drop_columns_matching(df, r'.*')


def test_drop_constant_columns_single_value():
    """Test drop_constant_columns removes columns with single unique value."""
    df = pd.DataFrame({
        'constant': [1, 1, 1],
        'varying': [1, 2, 3]
    })
    result = ar.drop_constant_columns(df)
    assert list(result.columns) == ['varying']


def test_drop_constant_columns_null_counts():
    """Test drop_constant_columns keeps column with nulls as second unique value."""
    df = pd.DataFrame({
        'all_null': [None, None],
        'with_null': [1, None]
    })
    result = ar.drop_constant_columns(df)
    assert list(result.columns) == ['with_null']


def test_drop_nulls_removes_rows():
    """Test drop_nulls removes rows containing any nulls."""
    df = ar.from_pandas(pd.DataFrame({
        'a': [1, 2, None],
        'b': [4, None, 6]
    }))
    result = ar.drop_nulls(df)
    assert len(result) == 1


def test_filter_rows_operator():
    """Test filter_rows with > operator."""
    df = pd.DataFrame({'age': [18, 25, 30]})
    result = ar.filter_rows(df, column='age', op='>', value=25)
    assert len(result) == 1
    assert result['age'].iloc[0] == 30


def test_keep_rows_with_nulls_basic():
    """Test keep_rows_with_nulls keeps only rows with at least one null."""
    df = pd.DataFrame({
        'a': [1, None, 3],
        'b': [4, 5, 6]
    })
    result = ar.keep_rows_with_nulls(df)
    assert len(result) == 1


def test_replace_values_in_column():
    """Test replace_values in a specific column."""
    df = pd.DataFrame({'status': ['old', 'old', 'new']})
    result = ar.replace_values(df, {'old': 'recent'}, column='status')
    assert result['status'].tolist() == ['recent', 'recent', 'new']


def test_replace_values_with_null_key():
    """Test replace_values handles None key in mapping."""
    df = pd.DataFrame({'value': [1.0, None, 3.0]})
    result = ar.replace_values(df, {None: 0.0})
    assert result['value'].iloc[1] == 0.0


def test_safe_divide_basic():
    """Test safe_divide_columns basic division."""
    df = pd.DataFrame({
        'numerator': [10.0, 20.0, 30.0],
        'denominator': [2.0, 4.0, 5.0]
    })
    result = ar.safe_divide_columns(df, numerator='numerator', denominator='denominator', output_column='ratio')
    assert result['ratio'].tolist() == [5.0, 5.0, 6.0]


def test_safe_divide_by_zero():
    """Test safe_divide_columns handles division by zero with fill_value."""
    df = pd.DataFrame({
        'numerator': [10.0, 20.0],
        'denominator': [0.0, 2.0]
    })
    result = ar.safe_divide_columns(df, numerator='numerator', denominator='denominator', output_column='ratio', fill_value=0.0)
    assert result['ratio'].iloc[0] == 0.0
    assert result['ratio'].iloc[1] == 10.0


def test_standardize_missing_tokens_default():
    """Test standardize_missing_tokens with default tokens."""
    df = pd.DataFrame({
        'value': ['N/A', 'valid', 'null', '123']
    })
    result = ar.standardize_missing_tokens(df)
    assert pd.isna(result['value'].iloc[0])
    assert result['value'].iloc[1] == 'valid'
    assert pd.isna(result['value'].iloc[2])
