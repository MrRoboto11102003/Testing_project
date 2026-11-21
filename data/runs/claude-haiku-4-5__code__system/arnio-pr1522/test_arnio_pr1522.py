import pytest
import pandas as pd
import numpy as np
import sys
import subprocess
from pathlib import Path
import arnio as ar


class TestNormalizeWhitespaceBasic:
    """Basic functionality tests for normalize_whitespace"""

    def test_basic_whitespace_normalization(self, tmp_path):
        """Test basic whitespace normalization with leading, trailing, and internal spaces"""
        csv_file = tmp_path / "test_basic.csv"
        csv_file.write_text("name,description\nJohn,  hello   world  \nJane,  foo  bar  ")
        
        frame = ar.read_csv(str(csv_file))
        clean = ar.normalize_whitespace(frame)
        result = clean.to_pandas() if hasattr(clean, 'to_pandas') else clean
        
        assert result['name'].iloc[0] == "John"
        assert result['description'].iloc[0] == "hello world"
        assert result['description'].iloc[1] == "foo bar"

    def test_normalize_specific_columns_only(self, tmp_path):
        """Test that only specified columns are normalized"""
        csv_file = tmp_path / "test_specific.csv"
        csv_file.write_text("name,description\nJohn  ,  hello world  \nJane  ,  foo  bar  ")
        
        frame = ar.read_csv(str(csv_file))
        clean = ar.normalize_whitespace(frame, columns=['name'])
        result = clean.to_pandas() if hasattr(clean, 'to_pandas') else clean
        
        assert result['name'].iloc[0] == "John"
        assert "  " in result['description'].iloc[0]  # description unchanged

    def test_pandas_dataframe_input(self):
        """Test that pandas DataFrame input returns pandas DataFrame"""
        df = pd.DataFrame({
            'text': ['  hello   world  ', 'foo  bar'],
            'value': [1, 2]
        })
        
        clean = ar.normalize_whitespace(df)
        
        assert isinstance(clean, pd.DataFrame)
        assert clean['text'].iloc[0] == "hello world"
        assert clean['text'].iloc[1] == "foo bar"

    def test_mixed_data_types_preserved(self):
        """Test that non-string types are passed through unchanged"""
        df = pd.DataFrame({
            'text': ['  hello  world  ', None, '  foo  bar  '],
            'integer': [1, 2, 3],
            'floating': [1.5, 2.5, 3.5],
            'boolean': [True, False, True]
        })
        
        clean = ar.normalize_whitespace(df)
        
        assert clean['text'].iloc[0] == "hello world"
        assert pd.isna(clean['text'].iloc[1])
        assert clean['text'].iloc[2] == "foo bar"
        assert clean['integer'].iloc[0] == 1
        assert clean['floating'].iloc[0] == 1.5
        assert clean['boolean'].iloc[0] is True

    def test_all_string_columns_normalized_by_default(self, tmp_path):
        """Test that all string columns are normalized when columns not specified"""
        csv_file = tmp_path / "test_all_string.csv"
        csv_file.write_text("col1,col2,col3\na  b,c  d,e  f\ng  h,i  j,k  l")
        
        frame = ar.read_csv(str(csv_file))
        clean = ar.normalize_whitespace(frame)
        result = clean.to_pandas() if hasattr(clean, 'to_pandas') else clean
        
        assert result['col1'].iloc[0] == "a b"
        assert result['col2'].iloc[0] == "c d"
        assert result['col3'].iloc[0] == "e f"

    def test_empty_and_whitespace_only_strings(self):
        """Test that empty strings and whitespace-only strings are handled correctly"""
        df = pd.DataFrame({
            'text': ['  hello  ', '', '   ', '  world  ']
        })
        
        clean = ar.normalize_whitespace(df)
        
        assert clean['text'].iloc[0] == "hello"
        assert clean['text'].iloc[1] == ""
        assert clean['text'].iloc[2] == ""
        assert clean['text'].iloc[3] == "world"

    def test_tabs_and_newlines_normalized(self):
        """Test that tabs, newlines, and carriage returns are normalized to spaces"""
        df = pd.DataFrame({
            'text': ['hello\tworld', 'foo\nbar', 'baz\r\nqux']
        })
        
        clean = ar.normalize_whitespace(df)
        
        assert clean['text'].iloc[0] == "hello world"
        assert clean['text'].iloc[1] == "foo bar"
        assert clean['text'].iloc[2] == "baz qux"

    def test_no_string_columns(self):
        """Test with dataframe containing no string columns"""
        df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.5, 2.5, 3.5]
        })
        
        clean = ar.normalize_whitespace(df)
        
        pd.testing.assert_frame_equal(clean, df)

    def test_missing_column_raises_error(self):
        """Test that specifying non-existent column raises ValueError"""
        df = pd.DataFrame({'text': ['hello']})
        
        with pytest.raises(ValueError, match="Missing columns"):
            ar.normalize_whitespace(df, columns=['nonexistent'])

    def test_object_dtype_with_mixed_types(self):
        """Test object column with mixed string and non-string values"""
        df = pd.DataFrame({
            'mixed': ['  hello  ', 42, None, 3.14, '  world  ']
        })
        
        clean = ar.normalize_whitespace(df)
        
        assert clean['mixed'].iloc[0] == "hello"
        assert clean['mixed'].iloc[1] == 42
        assert pd.isna(clean['mixed'].iloc[2])
        assert clean['mixed'].iloc[3] == 3.14
        assert clean['mixed'].iloc[4] == "world"

    def test_internal_single_spaces_preserved(self):
        """Test that single spaces between words are preserved"""
        df = pd.DataFrame({
            'text': ['hello world', 'foo bar baz']
        })
        
        clean = ar.normalize_whitespace(df)
        
        assert clean['text'].iloc[0] == "hello world"
        assert clean['text'].iloc[1] == "foo bar baz"

    def test_large_whitespace_runs_collapsed(self):
        """Test that large runs of mixed whitespace are collapsed to single space"""
        df = pd.DataFrame({
            'text': ['hello          world', 'foo\t\t\t\tbar\n\n\n\nbaz']
        })
        
        clean = ar.normalize_whitespace(df)
        
        assert clean['text'].iloc[0] == "hello world"
        assert clean['text'].iloc[1] == "foo bar baz"

    def test_pipeline_integration(self, tmp_path):
        """Test normalize_whitespace as part of a pipeline"""
        csv_file = tmp_path / "test_pipeline.csv"
        csv_file.write_text("name,value\n  John  ,100\n  Jane  ,200")
        
        frame = ar.read_csv(str(csv_file))
        clean = ar.pipeline(frame, [("normalize_whitespace", {})])
        result = clean.to_pandas() if hasattr(clean, 'to_pandas') else clean
        
        assert result['name'].iloc[0] == "John"
        assert result['name'].iloc[1] == "Jane"

    def test_string_dtype_column(self):
        """Test with pandas StringDtype column"""
        df = pd.DataFrame({
            'text': pd.Series(['  hello  world  ', '  foo  bar  '], dtype='string')
        })
        
        clean = ar.normalize_whitespace(df)
        
        assert clean['text'].iloc[0] == "hello world"
        assert clean['text'].iloc[1] == "foo bar"

    def test_empty_dataframe(self):
        """Test with empty dataframe"""
        df = pd.DataFrame({'text': pd.Series([], dtype='object')})
        
        clean = ar.normalize_whitespace(df)
        
        assert len(clean) == 0
        assert 'text' in clean.columns

    def test_cli_command(self, tmp_path):
        """Test normalize_whitespace via command line interface"""
        csv_file = tmp_path / "input.csv"
        csv_file.write_text("text\n  hello   world  \n  foo  bar  ")
        output_file = tmp_path / "output.csv"
        
        result = subprocess.run(
            [sys.executable, "-m", "arnio.cli", 
             "normalize_whitespace", 
             str(csv_file),
             "--output", str(output_file)],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert output_file.exists()
        output_df = pd.read_csv(str(output_file))
        assert output_df['text'].iloc[0] == "hello world"
        assert output_df['text'].iloc[1] == "foo bar"

    def test_unicode_whitespace_characters(self):
        """Test that various unicode whitespace characters are normalized"""
        df = pd.DataFrame({
            'text': ['hello\u00a0world', 'foo\u2003bar']  # non-breaking space, em space
        })
        
        clean = ar.normalize_whitespace(df)
        
        assert 'hello' in clean['text'].iloc[0] and 'world' in clean['text'].iloc[0]
        assert 'foo' in clean['text'].iloc[1] and 'bar' in clean['text'].iloc[1]
