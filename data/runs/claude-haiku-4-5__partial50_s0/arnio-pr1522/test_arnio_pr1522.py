import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestNormalizeWhitespace:
    """Test suite for arnio.cleaning.normalize_whitespace function"""

    def test_multiple_spaces_normalized_to_single(self):
        """Test that multiple consecutive spaces are normalized to single space"""
        df = pd.DataFrame({
            "text": ["hello    world", "foo  bar  baz", "a     b     c"]
        })
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        result_df = ar.to_pandas(result)
        
        assert result_df["text"].iloc[0] == "hello world"
        assert result_df["text"].iloc[1] == "foo bar baz"
        assert result_df["text"].iloc[2] == "a b c"

    def test_tabs_normalized_to_spaces(self):
        """Test that tabs are normalized"""
        df = pd.DataFrame({
            "text": ["hello\tworld", "foo\t\tbar"]
        })
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        result_df = ar.to_pandas(result)
        
        # Tabs should be handled/normalized
        assert "\t" not in result_df["text"].iloc[0]
        assert "\t" not in result_df["text"].iloc[1]

    def test_newlines_normalized(self):
        """Test that newlines are normalized"""
        df = pd.DataFrame({
            "text": ["hello\nworld", "foo\n\nbar"]
        })
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        result_df = ar.to_pandas(result)
        
        # Newlines should be handled/normalized
        assert "\n" not in result_df["text"].iloc[0]
        assert "\n" not in result_df["text"].iloc[1]

    def test_mixed_whitespace_types(self):
        """Test normalization with mixed whitespace characters"""
        df = pd.DataFrame({
            "text": ["hello  \t  world\n  test"]
        })
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        result_df = ar.to_pandas(result)
        
        # Mixed whitespace should be normalized
        assert "\t" not in result_df["text"].iloc[0]
        assert "\n" not in result_df["text"].iloc[0]

    def test_subset_columns_specified(self):
        """Test that only specified columns in subset are normalized"""
        df = pd.DataFrame({
            "col1": ["hello    world"],
            "col2": ["foo  bar"],
            "col3": ["baz    qux"]
        })
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame, columns=["col1", "col3"])
        result_df = ar.to_pandas(result)
        
        # col1 and col3 should be processed
        assert "    " not in result_df["col1"].iloc[0]
        assert "    " not in result_df["col3"].iloc[0]
        # col2 should remain unchanged
        assert result_df["col2"].iloc[0] == "foo  bar"

    def test_all_string_columns_when_no_subset(self):
        """Test that all text columns are normalized when subset is None"""
        df = pd.DataFrame({
            "text1": ["hello    world"],
            "text2": ["foo  bar"],
            "text3": ["baz    qux"]
        })
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        result_df = ar.to_pandas(result)
        
        # All string columns should be normalized
        assert result_df["text1"].iloc[0] == "hello world"
        assert result_df["text2"].iloc[0] == "foo bar"
        assert result_df["text3"].iloc[0] == "baz qux"

    def test_numeric_columns_not_affected(self):
        """Test that numeric columns are ignored by the function"""
        df = pd.DataFrame({
            "text": ["hello    world"],
            "number": [12345],
            "float_val": [123.45]
        })
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        result_df = ar.to_pandas(result)
        
        # Numeric values should be unchanged
        assert result_df["number"].iloc[0] == 12345
        assert result_df["float_val"].iloc[0] == 123.45

    def test_null_values_preserved(self):
        """Test that null/NA values are preserved during normalization"""
        df = pd.DataFrame({
            "text": ["hello    world", None, "foo  bar"]
        })
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        result_df = ar.to_pandas(result)
        
        # Null should remain null
        assert pd.isna(result_df["text"].iloc[1])
        # Other values should be normalized
        assert result_df["text"].iloc[0] == "hello world"
        assert result_df["text"].iloc[2] == "foo bar"

    def test_empty_string_preserved(self):
        """Test that empty strings remain empty"""
        df = pd.DataFrame({
            "text": ["hello    world", "", "foo  bar"]
        })
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        result_df = ar.to_pandas(result)
        
        # Empty string should remain empty
        assert result_df["text"].iloc[1] == ""

    def test_no_change_for_clean_strings(self):
        """Test that strings without whitespace issues remain unchanged"""
        df = pd.DataFrame({
            "text": ["hello", "world", "test"]
        })
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        result_df = ar.to_pandas(result)
        
        # Clean strings should be unchanged
        assert result_df["text"].iloc[0] == "hello"
        assert result_df["text"].iloc[1] == "world"
        assert result_df["text"].iloc[2] == "test"

    def test_leading_trailing_whitespace_handling(self):
        """Test behavior with leading and trailing whitespace"""
        df = pd.DataFrame({
            "text": ["  hello    world  ", "  foo  bar  "]
        })
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        result_df = ar.to_pandas(result)
        
        # Internal whitespace should be normalized
        text = result_df["text"].iloc[0]
        # Should not have multiple consecutive internal spaces
        assert "    " not in text

    def test_frame_shape_unchanged(self):
        """Test that frame dimensions are preserved"""
        df = pd.DataFrame({
            "col1": ["a    b", "c  d", "e    f"],
            "col2": ["g    h", "i  j", "k    l"]
        })
        frame = ar.from_pandas(df)
        original_shape = frame.shape
        result = ar.normalize_whitespace(frame)
        
        assert result.shape == original_shape

    def test_column_names_preserved(self):
        """Test that column names and order are maintained"""
        df = pd.DataFrame({
            "z_col": ["a    b"],
            "y_col": ["c  d"],
            "x_col": ["e    f"]
        })
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        
        assert result.columns == ["z_col", "y_col", "x_col"]

    def test_works_in_pipeline(self):
        """Test that normalize_whitespace works as a pipeline step"""
        df = pd.DataFrame({
            "text": ["hello    world", "foo  bar"]
        })
        frame = ar.from_pandas(df)
        
        clean = ar.pipeline(frame, [
            ("normalize_whitespace",),
        ])
        result_df = ar.to_pandas(clean)
        
        assert result_df["text"].iloc[0] == "hello world"
        assert result_df["text"].iloc[1] == "foo bar"

    def test_pipeline_with_subset_parameter(self):
        """Test normalize_whitespace in pipeline with column subset"""
        df = pd.DataFrame({
            "col1": ["hello    world"],
            "col2": ["foo  bar"]
        })
        frame = ar.from_pandas(df)
        
        clean = ar.pipeline(frame, [
            ("normalize_whitespace", {"columns": ["col1"]}),
        ])
        result_df = ar.to_pandas(clean)
        
        # col1 normalized, col2 unchanged
        assert result_df["col1"].iloc[0] == "hello world"
        assert result_df["col2"].iloc[0] == "foo  bar"
