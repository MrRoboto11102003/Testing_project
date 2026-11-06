import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestNormalizeWhitespace:
    """Test suite for ar.normalize_whitespace function."""

    def test_strip_leading_and_trailing_whitespace(self):
        """Test basic stripping of leading and trailing whitespace."""
        df = pd.DataFrame({"text": ["  hello  ", "  world  "]})
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        result_df = ar.to_pandas(result)
        assert result_df["text"].tolist() == ["hello", "world"]

    def test_collapse_internal_spaces(self):
        """Test collapsing multiple internal spaces to single space."""
        df = pd.DataFrame({"text": ["hello   world", "foo    bar"]})
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        result_df = ar.to_pandas(result)
        assert result_df["text"].tolist() == ["hello world", "foo bar"]

    def test_collapse_tabs_and_newlines(self):
        """Test that tabs, newlines, and spaces all collapse to single space."""
        df = pd.DataFrame({"text": ["hello\t\tworld", "foo\nbar", "a  \t  \n  b"]})
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        result_df = ar.to_pandas(result)
        assert result_df["text"].tolist() == ["hello world", "foo bar", "a b"]

    def test_whitespace_only_string_becomes_empty(self):
        """Test that strings with only whitespace become empty."""
        df = pd.DataFrame({"text": ["   ", "\t\t", "\n\n"]})
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        result_df = ar.to_pandas(result)
        assert result_df["text"].tolist() == ["", "", ""]

    def test_empty_string_unchanged(self):
        """Test that empty strings remain empty."""
        df = pd.DataFrame({"text": ["", "hello", ""]})
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        result_df = ar.to_pandas(result)
        assert result_df["text"].tolist() == ["", "hello", ""]

    def test_null_values_preserved(self):
        """Test that null/None values are preserved."""
        df = pd.DataFrame({"text": ["hello  world", None, "foo  bar", pd.NA]})
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        result_df = ar.to_pandas(result)
        assert result_df["text"][0] == "hello world"
        assert pd.isna(result_df["text"][1])
        assert result_df["text"][2] == "foo bar"
        assert pd.isna(result_df["text"][3])

    def test_specific_columns_only(self):
        """Test that only specified columns are normalized."""
        df = pd.DataFrame({
            "col_a": ["  alice  ", "  bob  "],
            "col_b": ["  x  y  ", "  a  b  "]
        })
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame, columns=["col_a"])
        result_df = ar.to_pandas(result)
        assert result_df["col_a"].tolist() == ["alice", "bob"]
        # col_b should remain unchanged
        assert result_df["col_b"].tolist() == ["  x  y  ", "  a  b  "]

    def test_all_string_columns_by_default(self):
        """Test that all string columns are processed when columns=None."""
        df = pd.DataFrame({
            "name": ["  alice  "],
            "city": ["  paris  "],
            "age": [25]  # numeric
        })
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        result_df = ar.to_pandas(result)
        assert result_df["name"][0] == "alice"
        assert result_df["city"][0] == "paris"
        assert result_df["age"][0] == 25  # unchanged

    def test_returns_arframe_when_input_arframe(self):
        """Test that function returns ArFrame when given ArFrame."""
        df = pd.DataFrame({"text": ["  hello  "]})
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        assert isinstance(result, ar.ArFrame)

    def test_returns_dataframe_when_input_dataframe(self):
        """Test that function returns DataFrame when given DataFrame."""
        df = pd.DataFrame({"text": ["  hello  "]})
        result = ar.normalize_whitespace(df)
        assert isinstance(result, pd.DataFrame)
        assert result["text"][0] == "hello"

    def test_non_string_columns_ignored(self):
        """Test that non-string columns are not processed."""
        df = pd.DataFrame({
            "text": ["  hello  "],
            "number": [42],
            "flag": [True]
        })
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        result_df = ar.to_pandas(result)
        assert result_df["text"][0] == "hello"
        assert result_df["number"][0] == 42
        assert result_df["flag"][0] is True

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame({"text": pd.Series([], dtype="object")})
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        result_df = ar.to_pandas(result)
        assert len(result_df) == 0
        assert list(result_df.columns) == ["text"]

    def test_missing_column_raises_error(self):
        """Test that specifying non-existent column raises ValueError."""
        df = pd.DataFrame({"name": ["alice"]})
        frame = ar.from_pandas(df)
        with pytest.raises(ValueError, match="Missing columns"):
            ar.normalize_whitespace(frame, columns=["nonexistent"])

    def test_frame_without_string_columns(self):
        """Test frame with no string columns."""
        df = pd.DataFrame({"a": [1, 2], "b": [3.5, 4.5]})
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        result_df = ar.to_pandas(result)
        pd.testing.assert_frame_equal(result_df, df)
