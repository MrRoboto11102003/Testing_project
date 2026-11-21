import pytest
import pandas as pd
import numpy as np
import arnio as ar
from arnio.convert import from_pandas, to_pandas


class TestNormalizeWhitespace:
    """Test suite for arnio.cleaning.normalize_whitespace"""

    def test_basic_multiple_spaces_collapse(self):
        """Test that multiple spaces collapse to single space"""
        df = pd.DataFrame({'text': ['hello   world', 'foo  bar  baz']})
        frame = from_pandas(df)
        result = ar.cleaning.normalize_whitespace(frame)
        result_df = to_pandas(result)
        assert result_df['text'].iloc[0] == 'hello world'
        assert result_df['text'].iloc[1] == 'foo bar baz'

    def test_strip_leading_trailing_whitespace(self):
        """Test removal of leading and trailing whitespace"""
        df = pd.DataFrame({'text': ['  hello world  ', '\thello\t', '\n  text  \n']})
        frame = from_pandas(df)
        result = ar.cleaning.normalize_whitespace(frame)
        result_df = to_pandas(result)
        assert result_df['text'].iloc[0] == 'hello world'
        assert result_df['text'].iloc[1] == 'hello'
        assert result_df['text'].iloc[2] == 'text'

    def test_mixed_whitespace_types(self):
        """Test normalization of tabs, newlines, and spaces together"""
        df = pd.DataFrame({'text': ['hello\t\t world', 'foo\nbar\nbaz', 'a  \t  b']})
        frame = from_pandas(df)
        result = ar.cleaning.normalize_whitespace(frame)
        result_df = to_pandas(result)
        assert result_df['text'].iloc[0] == 'hello world'
        assert result_df['text'].iloc[1] == 'foo bar baz'
        assert result_df['text'].iloc[2] == 'a b'

    def test_null_values_pass_through(self):
        """Test that null/NaN values are not modified"""
        df = pd.DataFrame({'text': ['hello  world', None, np.nan]})
        frame = from_pandas(df)
        result = ar.cleaning.normalize_whitespace(frame)
        result_df = to_pandas(result)
        assert result_df['text'].iloc[0] == 'hello world'
        assert pd.isna(result_df['text'].iloc[1])
        assert pd.isna(result_df['text'].iloc[2])

    def test_non_string_types_pass_through(self):
        """Test that non-string types (int, float, bool) pass through unchanged"""
        df = pd.DataFrame({'mixed': [42, 3.14, True, False, None, 'hello  world']})
        frame = from_pandas(df)
        result = ar.cleaning.normalize_whitespace(frame)
        result_df = to_pandas(result)
        assert result_df['mixed'].iloc[0] == 42
        assert result_df['mixed'].iloc[1] == 3.14
        assert result_df['mixed'].iloc[2] is True
        assert result_df['mixed'].iloc[3] is False
        assert pd.isna(result_df['mixed'].iloc[4])
        assert result_df['mixed'].iloc[5] == 'hello world'

    def test_empty_strings_remain_empty(self):
        """Test that empty strings and whitespace-only strings become empty"""
        df = pd.DataFrame({'text': ['', '   ', '\t', '\n', '   \t\n   ']})
        frame = from_pandas(df)
        result = ar.cleaning.normalize_whitespace(frame)
        result_df = to_pandas(result)
        assert all(result_df['text'] == '')

    def test_specific_columns_parameter(self):
        """Test that only specified columns are processed"""
        df = pd.DataFrame({
            'col1': ['a   b'],
            'col2': ['c   d'],
            'col3': ['e   f']
        })
        frame = from_pandas(df)
        result = ar.cleaning.normalize_whitespace(frame, columns=['col1', 'col3'])
        result_df = to_pandas(result)
        assert result_df['col1'].iloc[0] == 'a b'
        assert result_df['col2'].iloc[0] == 'c   d'  # unchanged
        assert result_df['col3'].iloc[0] == 'e f'

    def test_nonexistent_column_raises_error(self):
        """Test that specifying non-existent column raises ValueError"""
        df = pd.DataFrame({'text': ['hello world']})
        frame = from_pandas(df)
        with pytest.raises(ValueError, match="Missing columns"):
            ar.cleaning.normalize_whitespace(frame, columns=['nonexistent'])

    def test_numeric_column_in_columns_list_filtered(self):
        """Test that numeric columns specified in columns list are silently filtered"""
        df = pd.DataFrame({
            'text': ['a   b'],
            'number': [42]
        })
        frame = from_pandas(df)
        result = ar.cleaning.normalize_whitespace(frame, columns=['text', 'number'])
        result_df = to_pandas(result)
        assert result_df['text'].iloc[0] == 'a b'
        assert result_df['number'].iloc[0] == 42  # numeric unchanged

    def test_default_processes_all_string_columns(self):
        """Test that without columns parameter, all string columns are processed"""
        df = pd.DataFrame({
            'str1': ['x   y'],
            'str2': ['p   q'],
            'num': [100]
        })
        frame = from_pandas(df)
        result = ar.cleaning.normalize_whitespace(frame)
        result_df = to_pandas(result)
        assert result_df['str1'].iloc[0] == 'x y'
        assert result_df['str2'].iloc[0] == 'p q'
        assert result_df['num'].iloc[0] == 100

    def test_original_frame_not_modified(self):
        """Test that original frame is not modified (returns new frame)"""
        df = pd.DataFrame({'text': ['hello   world']})
        frame = from_pandas(df)
        result = ar.cleaning.normalize_whitespace(frame)
        
        # Original unchanged
        original_df = to_pandas(frame)
        assert original_df['text'].iloc[0] == 'hello   world'
        
        # Result changed
        result_df = to_pandas(result)
        assert result_df['text'].iloc[0] == 'hello world'

    def test_already_normalized_strings_unchanged(self):
        """Test that already normalized strings remain unchanged"""
        df = pd.DataFrame({'text': ['hello world', 'foo bar', 'single']})
        frame = from_pandas(df)
        result = ar.cleaning.normalize_whitespace(frame)
        result_df = to_pandas(result)
        assert result_df['text'].iloc[0] == 'hello world'
        assert result_df['text'].iloc[1] == 'foo bar'
        assert result_df['text'].iloc[2] == 'single'

    def test_very_long_whitespace_runs(self):
        """Test collapsing very long runs of whitespace"""
        df = pd.DataFrame({'text': ['hello' + ' ' * 100 + 'world']})
        frame = from_pandas(df)
        result = ar.cleaning.normalize_whitespace(frame)
        result_df = to_pandas(result)
        assert result_df['text'].iloc[0] == 'hello world'

    def test_multiple_rows_and_columns(self):
        """Test processing multiple rows and columns correctly"""
        df = pd.DataFrame({
            'col1': ['a   b', 'c   d', 'e   f'],
            'col2': ['x  y', 'p  q', 'r  s']
        })
        frame = from_pandas(df)
        result = ar.cleaning.normalize_whitespace(frame)
        result_df = to_pandas(result)
        
        assert len(result_df) == 3
        expected_col1 = ['a b', 'c d', 'e f']
        expected_col2 = ['x y', 'p q', 'r s']
        assert list(result_df['col1']) == expected_col1
        assert list(result_df['col2']) == expected_col2

    def test_single_character_strings(self):
        """Test edge case of single character and strings with minimal whitespace"""
        df = pd.DataFrame({'text': ['a', ' a ', 'a b', 'a  b']})
        frame = from_pandas(df)
        result = ar.cleaning.normalize_whitespace(frame)
        result_df = to_pandas(result)
        assert result_df['text'].iloc[0] == 'a'
        assert result_df['text'].iloc[1] == 'a'
        assert result_df['text'].iloc[2] == 'a b'
        assert result_df['text'].iloc[3] == 'a b'

    def test_various_whitespace_combinations(self):
        """Test various combinations of spaces, tabs, newlines"""
        df = pd.DataFrame({
            'text': [
                '  leading',
                'trailing  ',
                '  both  ',
                'mid  dle',
                'tab\tchar',
                'new\nline'
            ]
        })
        frame = from_pandas(df)
        result = ar.cleaning.normalize_whitespace(frame)
        result_df = to_pandas(result)
        
        assert result_df['text'].iloc[0] == 'leading'
        assert result_df['text'].iloc[1] == 'trailing'
        assert result_df['text'].iloc[2] == 'both'
        assert result_df['text'].iloc[3] == 'mid dle'
        assert result_df['text'].iloc[4] == 'tab char'
        assert result_df['text'].iloc[5] == 'new line'
