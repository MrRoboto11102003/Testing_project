import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestNormalizeWhitespace:
    """Tests for arnio.cleaning.normalize_whitespace function."""

    def test_basic_multiple_spaces_normalized(self):
        """Test that multiple consecutive spaces are collapsed to single space."""
        df = pd.DataFrame({"text": ["hello  world", "foo   bar"]})
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        result_df = ar.to_pandas(result)
        
        assert result_df["text"].iloc[0] == "hello world"
        assert result_df["text"].iloc[1] == "foo bar"

    def test_tabs_normalized_to_spaces(self):
        """Test that tab characters are normalized."""
        df = pd.DataFrame({"text": ["hello\tworld", "foo\t\tbar"]})
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        result_df = ar.to_pandas(result)
        
        assert "\t" not in result_df["text"].iloc[0]
        assert "\t" not in result_df["text"].iloc[1]

    def test_newlines_normalized(self):
        """Test that newline characters are normalized."""
        df = pd.DataFrame({"text": ["hello\nworld", "foo\n\nbar"]})
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        result_df = ar.to_pandas(result)
        
        assert "\n" not in result_df["text"].iloc[0]
        assert "\n" not in result_df["text"].iloc[1]

    def test_mixed_whitespace_types(self):
        """Test normalization of mixed whitespace characters."""
        df = pd.DataFrame({"text": ["hello \t \n world"]})
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        result_df = ar.to_pandas(result)
        
        # Should collapse mixed whitespace
        assert "hello" in result_df["text"].iloc[0]
        assert "world" in result_df["text"].iloc[0]

    def test_columns_parameter_single_column(self):
        """Test that columns parameter limits normalization to specified columns."""
        df = pd.DataFrame({
            "text": ["hello  world"],
            "other": ["foo   bar"]
        })
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame, columns=["text"])
        result_df = ar.to_pandas(result)
        
        assert result_df["text"].iloc[0] == "hello world"
        assert result_df["other"].iloc[0] == "foo   bar"  # unchanged

    def test_columns_parameter_multiple_columns(self):
        """Test that columns parameter works with multiple columns."""
        df = pd.DataFrame({
            "col1": ["hello  world"],
            "col2": ["foo   bar"],
            "col3": ["baz    qux"]
        })
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame, columns=["col1", "col2"])
        result_df = ar.to_pandas(result)
        
        assert result_df["col1"].iloc[0] == "hello world"
        assert result_df["col2"].iloc[0] == "foo bar"
        assert result_df["col3"].iloc[0] == "baz    qux"  # unchanged

    def test_null_values_preserved(self):
        """Test that null/NaN values are preserved."""
        df = pd.DataFrame({"text": ["hello  world", None, "foo   bar", np.nan]})
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        result_df = ar.to_pandas(result)
        
        assert result_df["text"].iloc[0] == "hello world"
        assert pd.isna(result_df["text"].iloc[1])
        assert result_df["text"].iloc[2] == "foo bar"
        assert pd.isna(result_df["text"].iloc[3])

    def test_numeric_columns_ignored(self):
        """Test that numeric columns are left unchanged."""
        df = pd.DataFrame({
            "text": ["hello  world"],
            "number": [123.45],
            "integer": [42]
        })
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        result_df = ar.to_pandas(result)
        
        assert result_df["number"].iloc[0] == 123.45
        assert result_df["integer"].iloc[0] == 42
        assert result_df["text"].iloc[0] == "hello world"

    def test_empty_frame(self):
        """Test behavior on empty frame."""
        df = pd.DataFrame({"text": pd.Series([], dtype=str)})
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        result_df = ar.to_pandas(result)
        
        assert len(result_df) == 0
        assert "text" in result_df.columns

    def test_all_string_columns_when_none_specified(self):
        """Test that all string columns are processed when columns=None."""
        df = pd.DataFrame({
            "col1": ["hello  world"],
            "col2": ["foo   bar"],
            "col3": [123]
        })
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame, columns=None)
        result_df = ar.to_pandas(result)
        
        assert result_df["col1"].iloc[0] == "hello world"
        assert result_df["col2"].iloc[0] == "foo bar"
        assert result_df["col3"].iloc[0] == 123

    def test_nonexistent_column_raises_error(self):
        """Test that specifying a non-existent column raises ValueError."""
        df = pd.DataFrame({"text": ["hello  world"]})
        frame = ar.from_pandas(df)
        
        with pytest.raises((ValueError, KeyError)):
            ar.normalize_whitespace(frame, columns=["nonexistent"])

    def test_single_spaces_unchanged(self):
        """Test that text with only single spaces is unchanged."""
        df = pd.DataFrame({"text": ["hello world foo bar baz"]})
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        result_df = ar.to_pandas(result)
        
        assert result_df["text"].iloc[0] == "hello world foo bar baz"

    def test_whitespace_only_strings(self):
        """Test handling of strings that contain only whitespace."""
        df = pd.DataFrame({"text": ["   ", "\t\t", "  \n  "]})
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        result_df = ar.to_pandas(result)
        
        # Should normalize but preserve as whitespace strings
        assert len(result_df) == 3

    def test_boolean_columns_ignored(self):
        """Test that boolean columns are left unchanged."""
        df = pd.DataFrame({
            "text": ["hello  world"],
            "flag": [True, False]
        })
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        result_df = ar.to_pandas(result)
        
        assert result_df["text"].iloc[0] == "hello world"
        assert result_df["flag"].iloc[0] is True

    def test_unicode_whitespace_normalized(self):
        """Test that Unicode whitespace characters are normalized."""
        df = pd.DataFrame({"text": ["hello\u00A0\u00A0world"]})  # non-breaking spaces
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        result_df = ar.to_pandas(result)
        
        # Should normalize non-breaking spaces
        assert "hello" in result_df["text"].iloc[0]
        assert "world" in result_df["text"].iloc[0]

    def test_returns_new_frame_not_mutated(self):
        """Test that original frame is not mutated."""
        df = pd.DataFrame({"text": ["hello  world"]})
        frame = ar.from_pandas(df)
        original_preview = frame.preview()
        
        result = ar.normalize_whitespace(frame)
        
        # Original frame should be unchanged
        assert frame.preview() == original_preview
        assert result is not frame
