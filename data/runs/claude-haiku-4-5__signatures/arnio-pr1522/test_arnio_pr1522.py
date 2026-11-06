import pytest
import pandas as pd
import numpy as np
import arnio as ar


def test_normalize_whitespace_basic_strip():
    """Strip leading and trailing whitespace."""
    df = pd.DataFrame({'text': ['  hello  ', ' world']})
    result = ar.cleaning.normalize_whitespace(df)
    assert result['text'].iloc[0] == 'hello'
    assert result['text'].iloc[1] == 'world'


def test_normalize_whitespace_multiple_spaces():
    """Collapse multiple internal spaces into single space."""
    df = pd.DataFrame({'text': ['hello  world', 'foo   bar   baz']})
    result = ar.cleaning.normalize_whitespace(df)
    assert result['text'].iloc[0] == 'hello world'
    assert result['text'].iloc[1] == 'foo bar baz'


def test_normalize_whitespace_tabs_newlines():
    """Normalize tabs and newlines."""
    df = pd.DataFrame({'text': ['hello\t\tworld', 'foo\nbar', 'baz \r\n qux']})
    result = ar.cleaning.normalize_whitespace(df)
    # Tabs/newlines should be normalized, with content preserved
    assert 'hello' in result['text'].iloc[0] and 'world' in result['text'].iloc[0]
    assert 'foo' in result['text'].iloc[1] and 'bar' in result['text'].iloc[1]


def test_normalize_whitespace_nan_preserved():
    """Preserve NaN values."""
    df = pd.DataFrame({'text': ['  hello  ', np.nan, 'world']})
    result = ar.cleaning.normalize_whitespace(df)
    assert result['text'].iloc[0] == 'hello'
    assert pd.isna(result['text'].iloc[1])
    assert result['text'].iloc[2] == 'world'


def test_normalize_whitespace_none_preserved():
    """Preserve None values."""
    df = pd.DataFrame({'text': ['  hello  ', None, 'world']})
    result = ar.cleaning.normalize_whitespace(df)
    assert result['text'].iloc[0] == 'hello'
    assert pd.isna(result['text'].iloc[1])
    assert result['text'].iloc[2] == 'world'


def test_normalize_whitespace_empty_strings():
    """Handle empty and whitespace-only strings."""
    df = pd.DataFrame({'text': ['', '   ', '\t', 'hello']})
    result = ar.cleaning.normalize_whitespace(df)
    assert result['text'].iloc[0] == ''
    assert result['text'].iloc[1] == ''
    assert result['text'].iloc[2] == ''
    assert result['text'].iloc[3] == 'hello'


def test_normalize_whitespace_specific_column():
    """Only normalize specified column."""
    df = pd.DataFrame({
        'col1': ['  hello  '],
        'col2': ['  world  ']
    })
    result = ar.cleaning.normalize_whitespace(df, columns=['col1'])
    assert result['col1'].iloc[0] == 'hello'
    assert result['col2'].iloc[0] == '  world  '


def test_normalize_whitespace_multiple_columns():
    """Normalize multiple specified columns."""
    df = pd.DataFrame({
        'a': ['  x  '],
        'b': ['  y  '],
        'c': ['  z  ']
    })
    result = ar.cleaning.normalize_whitespace(df, columns=['a', 'b'])
    assert result['a'].iloc[0] == 'x'
    assert result['b'].iloc[0] == 'y'
    assert result['c'].iloc[0] == '  z  '


def test_normalize_whitespace_all_columns_default():
    """Without columns arg, normalize all string columns."""
    df = pd.DataFrame({
        'text1': ['  hello  '],
        'text2': ['  world  ']
    })
    result = ar.cleaning.normalize_whitespace(df)
    assert result['text1'].iloc[0] == 'hello'
    assert result['text2'].iloc[0] == 'world'


def test_normalize_whitespace_numeric_columns_untouched():
    """Numeric columns are not affected."""
    df = pd.DataFrame({
        'text': ['  hello  '],
        'number': [42],
        'float': [3.14]
    })
    result = ar.cleaning.normalize_whitespace(df)
    assert result['text'].iloc[0] == 'hello'
    assert result['number'].iloc[0] == 42
    assert result['float'].iloc[0] == 3.14


def test_normalize_whitespace_mixed_whitespace_types():
    """Handle combination of spaces, tabs, and newlines."""
    df = pd.DataFrame({'text': [' \t  hello \n\n world  \t ']})
    result = ar.cleaning.normalize_whitespace(df)
    cleaned = result['text'].iloc[0]
    assert 'hello' in cleaned and 'world' in cleaned
    # Should be normalized to single spaces
    assert cleaned != ' \t  hello \n\n world  \t '


def test_normalize_whitespace_preserve_index():
    """DataFrame index is preserved."""
    df = pd.DataFrame({'text': ['  a  ', '  b  ', '  c  ']}, index=[10, 20, 30])
    result = ar.cleaning.normalize_whitespace(df)
    assert list(result.index) == [10, 20, 30]


def test_normalize_whitespace_preserve_column_order():
    """Column order is preserved."""
    df = pd.DataFrame({
        'z': ['  z  '],
        'a': ['  a  '],
        'm': ['  m  ']
    })
    result = ar.cleaning.normalize_whitespace(df)
    assert list(result.columns) == ['z', 'a', 'm']


def test_normalize_whitespace_empty_dataframe():
    """Empty DataFrame is handled gracefully."""
    df = pd.DataFrame()
    result = ar.cleaning.normalize_whitespace(df)
    assert len(result) == 0
    assert isinstance(result, pd.DataFrame)


def test_normalize_whitespace_returns_dataframe():
    """Function returns a pandas DataFrame."""
    df = pd.DataFrame({'text': ['  hello  ']})
    result = ar.cleaning.normalize_whitespace(df)
    assert isinstance(result, pd.DataFrame)
