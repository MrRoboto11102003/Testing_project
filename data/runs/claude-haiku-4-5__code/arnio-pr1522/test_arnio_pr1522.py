import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestNormalizeWhitespace:
    """Test suite for arnio.cleaning.normalize_whitespace function."""

    def test_collapse_multiple_spaces(self):
        """Test collapsing multiple internal spaces to single space."""
        df = pd.DataFrame({
            'text': ['hello  world', 'foo    bar', 'a  b  c']
        })
        result = ar.cleaning.normalize_whitespace(df)
        assert result.loc[0, 'text'] == 'hello world'
        assert result.loc[1, 'text'] == 'foo bar'
        assert result.loc[2, 'text'] == 'a b c'

    def test_strip_leading_and_trailing(self):
        """Test that leading and trailing whitespace is stripped."""
        df = pd.DataFrame({
            'text': ['  hello world  ', '\t\thello\t\t', '\nhello\n']
        })
        result = ar.cleaning.normalize_whitespace(df)
        assert result.loc[0, 'text'] == 'hello world'
        assert result.loc[1, 'text'] == 'hello'
        assert result.loc[2, 'text'] == 'hello'

    def test_collapse_tabs(self):
        """Test that tabs are collapsed to single space."""
        df = pd.DataFrame({
            'text': ['hello\t\tworld', 'foo\tbar']
        })
        result = ar.cleaning.normalize_whitespace(df)
        assert result.loc[0, 'text'] == 'hello world'
        assert result.loc[1, 'text'] == 'foo bar'

    def test_collapse_newlines(self):
        """Test that newlines are collapsed to single space."""
        df = pd.DataFrame({
            'text': ['hello\n\nworld', 'foo\nbar']
        })
        result = ar.cleaning.normalize_whitespace(df)
        assert result.loc[0, 'text'] == 'hello world'
        assert result.loc[1, 'text'] == 'foo bar'

    def test_mixed_whitespace_types(self):
        """Test collapsing mixed whitespace types (spaces, tabs, newlines)."""
        df = pd.DataFrame({
            'text': ['hello \t\n world', 'a  \tb\n  c']
        })
        result = ar.cleaning.normalize_whitespace(df)
        assert result.loc[0, 'text'] == 'hello world'
        assert result.loc[1, 'text'] == 'a b c'

    def test_empty_string(self):
        """Test that empty strings remain empty."""
        df = pd.DataFrame({
            'text': ['', 'hello', '']
        })
        result = ar.cleaning.normalize_whitespace(df)
        assert result.loc[0, 'text'] == ''
        assert result.loc[1, 'text'] == 'hello'
        assert result.loc[2, 'text'] == ''

    def test_whitespace_only_becomes_empty(self):
        """Test that strings with only whitespace become empty."""
        df = pd.DataFrame({
            'text': ['   ', '\t\t', '\n\n', '  \t\n  ']
        })
        result = ar.cleaning.normalize_whitespace(df)
        assert all(result['text'] == '')

    def test_none_and_nan_preserved(self):
        """Test that None and NaN values are preserved."""
        df = pd.DataFrame({
            'text': ['hello  world', None, np.nan, 'foo  bar']
        })
        result = ar.cleaning.normalize_whitespace(df)
        assert result.loc[0, 'text'] == 'hello world'
        assert pd.isna(result.loc[1, 'text'])
        assert pd.isna(result.loc[2, 'text'])
        assert result.loc[3, 'text'] == 'foo bar'

    def test_numeric_values_in_object_column(self):
        """Test that numeric and boolean values in object columns are preserved."""
        df = pd.DataFrame({
            'mixed': [1, '  hello  ', 2.5, '  world  ', True, False]
        })
        result = ar.cleaning.normalize_whitespace(df)
        assert result.loc[0, 'mixed'] == 1
        assert result.loc[1, 'mixed'] == 'hello'
        assert result.loc[2, 'mixed'] == 2.5
        assert result.loc[3, 'mixed'] == 'world'
        assert result.loc[4, 'mixed'] is True
        assert result.loc[5, 'mixed'] is False

    def test_specific_columns_parameter(self):
        """Test processing only specified columns."""
        df = pd.DataFrame({
            'col1': ['hello  world', 'foo  bar'],
            'col2': ['  test  ', '  data  '],
            'col3': ['untouched  spaces', 'keep  them']
        })
        result = ar.cleaning.normalize_whitespace(df, columns=['col1', 'col2'])
        assert result.loc[0, 'col1'] == 'hello world'
        assert result.loc[0, 'col2'] == 'test'
        assert result.loc[0, 'col3'] == 'untouched  spaces'
        assert result.loc[1, 'col3'] == 'keep  them'

    def test_default_processes_all_string_columns(self):
        """Test that default processes all string columns when columns parameter is None."""
        df = pd.DataFrame({
            'str_col1': ['hello  world'],
            'str_col2': ['  test  '],
            'int_col': [123],
            'float_col': [1.5]
        })
        result = ar.cleaning.normalize_whitespace(df)
        assert result.loc[0, 'str_col1'] == 'hello world'
        assert result.loc[0, 'str_col2'] == 'test'
        assert result.loc[0, 'int_col'] == 123
        assert result.loc[0, 'float_col'] == 1.5

    def test_missing_column_raises_error(self):
        """Test that specifying a missing column raises ValueError."""
        df = pd.DataFrame({
            'col1': ['hello'],
            'col2': ['world']
        })
        with pytest.raises(ValueError, match="Missing columns"):
            ar.cleaning.normalize_whitespace(df, columns=['col1', 'missing_col'])

    def test_non_string_columns_ignored_when_specified(self):
        """Test that non-string columns are ignored when specified in columns parameter."""
        df = pd.DataFrame({
            'int_col': [1, 2],
            'float_col': [1.1, 2.2],
            'str_col': ['  hello  ', '  world  ']
        })
        result = ar.cleaning.normalize_whitespace(df, columns=['int_col', 'float_col', 'str_col'])
        assert result.loc[0, 'int_col'] == 1
        assert result.loc[0, 'float_col'] == 1.1
        assert result.loc[0, 'str_col'] == 'hello'

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame({'text': pd.Series([], dtype='object')})
        result = ar.cleaning.normalize_whitespace(df)
        assert len(result) == 0
        assert 'text' in result.columns

    def test_already_normalized_unchanged(self):
        """Test that already normalized strings remain unchanged."""
        df = pd.DataFrame({
            'text': ['hello world', 'foo bar', 'single']
        })
        result = ar.cleaning.normalize_whitespace(df)
        pd.testing.assert_frame_equal(result, df)
