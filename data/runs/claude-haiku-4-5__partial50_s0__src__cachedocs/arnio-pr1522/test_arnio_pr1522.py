import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestNormalizeWhitespace:
    """Test suite for arnio.cleaning.normalize_whitespace"""

    def test_basic_whitespace_normalization(self):
        """Test collapsing multiple spaces to single space"""
        df = pd.DataFrame({
            "text": ["hello    world", "foo  bar  baz", "test     string"]
        })
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        result_df = ar.to_pandas(result)
        
        assert result_df["text"].iloc[0] == "hello world"
        assert result_df["text"].iloc[1] == "foo bar baz"
        assert result_df["text"].iloc[2] == "test string"

    def test_leading_trailing_whitespace_removal(self):
        """Test removal of leading and trailing whitespace"""
        df = pd.DataFrame({
            "text": ["  hello", "world  ", "  both sides  "]
        })
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        result_df = ar.to_pandas(result)
        
        assert result_df["text"].iloc[0] == "hello"
        assert result_df["text"].iloc[1] == "world"
        assert result_df["text"].iloc[2] == "both sides"

    def test_combined_whitespace_handling(self):
        """Test handling of combined leading, internal, and trailing whitespace"""
        df = pd.DataFrame({
            "text": ["  hello   world  ", "   multiple    spaces    here   "]
        })
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        result_df = ar.to_pandas(result)
        
        assert result_df["text"].iloc[0] == "hello world"
        assert result_df["text"].iloc[1] == "multiple spaces here"

    def test_column_subset_selection(self):
        """Test that only specified columns are processed"""
        df = pd.DataFrame({
            "text1": ["  hello   world  "],
            "text2": ["  foo   bar  "],
        })
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame, columns=["text1"])
        result_df = ar.to_pandas(result)
        
        assert result_df["text1"].iloc[0] == "hello world"
        assert result_df["text2"].iloc[0] == "  foo   bar  "

    def test_default_all_string_columns(self):
        """Test that all string columns are processed when columns=None"""
        df = pd.DataFrame({
            "col1": ["  a   b  "],
            "col2": ["  x   y  "],
            "col3": [42]
        })
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        result_df = ar.to_pandas(result)
        
        assert result_df["col1"].iloc[0] == "a b"
        assert result_df["col2"].iloc[0] == "x y"
        assert result_df["col3"].iloc[0] == 42

    def test_non_string_values_passthrough(self):
        """Test that non-string values pass through unchanged"""
        df = pd.DataFrame({
            "text": ["  hello  ", "world"],
            "num": [42, 3.14],
            "bool": [True, False],
        })
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        result_df = ar.to_pandas(result)
        
        assert result_df["num"].iloc[0] == 42
        assert result_df["num"].iloc[1] == 3.14
        assert result_df["bool"].iloc[0] is True
        assert result_df["bool"].iloc[1] is False

    def test_tab_and_newline_whitespace(self):
        """Test handling of tabs, newlines, and other whitespace"""
        df = pd.DataFrame({
            "text": ["hello\t\tworld", "foo\n\nbar", "mixed \t \n space"]
        })
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        result_df = ar.to_pandas(result)
        
        assert result_df["text"].iloc[0] == "hello world"
        assert result_df["text"].iloc[1] == "foo bar"
        assert result_df["text"].iloc[2] == "mixed space"

    def test_already_normalized_strings(self):
        """Test that already normalized strings remain unchanged"""
        df = pd.DataFrame({
            "text": ["hello world", "foo bar baz"]
        })
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        result_df = ar.to_pandas(result)
        
        assert result_df["text"].iloc[0] == "hello world"
        assert result_df["text"].iloc[1] == "foo bar baz"

    def test_empty_string_handling(self):
        """Test handling of empty and whitespace-only strings"""
        df = pd.DataFrame({
            "text": ["", "  ", "   \t  ", "hello"]
        })
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        result_df = ar.to_pandas(result)
        
        assert result_df["text"].iloc[0] == ""
        assert result_df["text"].iloc[1] == ""
        assert result_df["text"].iloc[2] == ""
        assert result_df["text"].iloc[3] == "hello"

    def test_invalid_column_names_raises_error(self):
        """Test that requesting non-existent columns raises ValueError"""
        df = pd.DataFrame({"text": ["hello world"]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(ValueError, match="Missing columns"):
            ar.normalize_whitespace(frame, columns=["nonexistent"])

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame"""
        df = pd.DataFrame({"text": pd.Series([], dtype="object")})
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        result_df = ar.to_pandas(result)
        
        assert len(result_df) == 0
        assert "text" in result_df.columns

    def test_returns_arframe_for_arframe_input(self):
        """Test that ArFrame input returns ArFrame output"""
        df = pd.DataFrame({"text": ["  hello  "]})
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        
        assert isinstance(result, ar.ArFrame)

    def test_returns_dataframe_for_pandas_input(self):
        """Test that pandas DataFrame input returns pandas DataFrame output"""
        df = pd.DataFrame({"text": ["  hello  "]})
        result = ar.normalize_whitespace(df)
        
        assert isinstance(result, pd.DataFrame)

    def test_preserves_legitimate_internal_structure(self):
        """Test that legitimate single-spaced content is preserved"""
        df = pd.DataFrame({
            "text": ["word1 word2 word3", "a b c d e"]
        })
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        result_df = ar.to_pandas(result)
        
        assert result_df["text"].iloc[0] == "word1 word2 word3"
        assert result_df["text"].iloc[1] == "a b c d e"

    def test_null_values_passthrough(self):
        """Test that None and NaN values pass through unchanged"""
        df = pd.DataFrame({
            "text": ["  hello  ", None, "world"],
        })
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        result_df = ar.to_pandas(result)
        
        assert result_df["text"].iloc[0] == "hello"
        assert pd.isna(result_df["text"].iloc[1])
        assert result_df["text"].iloc[2] == "world"

    def test_mixed_string_dtypes(self):
        """Test handling of both object and string dtype columns"""
        df = pd.DataFrame({
            "obj": pd.Series(["  hello  "], dtype="object"),
            "str": pd.Series(["  world  "], dtype="string")
        })
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        result_df = ar.to_pandas(result)
        
        assert result_df["obj"].iloc[0] == "hello"
        assert result_df["str"].iloc[0] == "world"
