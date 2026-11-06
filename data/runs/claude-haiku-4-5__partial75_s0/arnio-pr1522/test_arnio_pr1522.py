import pytest
import pandas as pd
import arnio as ar


class TestNormalizeWhitespace:
    """Test suite for arnio.cleaning.normalize_whitespace function."""

    def test_collapse_multiple_spaces(self):
        """Multiple consecutive spaces are collapsed to single space."""
        df = pd.DataFrame({"text": ["hello    world"]})
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        result_df = ar.to_pandas(result)
        assert result_df.loc[0, "text"] == "hello world"

    def test_strip_leading_trailing_whitespace(self):
        """Leading and trailing whitespace is removed."""
        df = pd.DataFrame({"text": ["   hello world   "]})
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        result_df = ar.to_pandas(result)
        assert result_df.loc[0, "text"] == "hello world"

    def test_normalize_tabs_and_newlines(self):
        """Tabs and newlines are normalized to spaces."""
        df = pd.DataFrame({"text": ["hello\t\tworld\ntest"]})
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        result_df = ar.to_pandas(result)
        assert result_df.loc[0, "text"] == "hello world test"

    def test_preserve_clean_strings(self):
        """Properly formatted strings are unchanged."""
        df = pd.DataFrame({"text": ["hello world"]})
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        result_df = ar.to_pandas(result)
        assert result_df.loc[0, "text"] == "hello world"

    def test_whitespace_only_becomes_empty(self):
        """Strings containing only whitespace become empty."""
        df = pd.DataFrame({"text": ["   \t\n  "]})
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        result_df = ar.to_pandas(result)
        assert result_df.loc[0, "text"] == ""

    def test_null_values_preserved(self):
        """Null/None values are preserved."""
        df = pd.DataFrame({"text": ["hello  world", None, "foo   bar"]})
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        result_df = ar.to_pandas(result)
        assert result_df.loc[0, "text"] == "hello world"
        assert pd.isna(result_df.loc[1, "text"])
        assert result_df.loc[2, "text"] == "foo bar"

    def test_subset_single_column(self):
        """Only specified columns are normalized when columns parameter provided."""
        df = pd.DataFrame({
            "col1": ["hello    world"],
            "col2": ["foo    bar"]
        })
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame, columns=["col1"])
        result_df = ar.to_pandas(result)
        assert result_df.loc[0, "col1"] == "hello world"
        assert result_df.loc[0, "col2"] == "foo    bar"

    def test_subset_multiple_columns(self):
        """Multiple columns can be normalized via columns parameter."""
        df = pd.DataFrame({
            "a": ["x    y"],
            "b": ["p    q"],
            "c": ["m    n"]
        })
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame, columns=["a", "c"])
        result_df = ar.to_pandas(result)
        assert result_df.loc[0, "a"] == "x y"
        assert result_df.loc[0, "b"] == "p    q"
        assert result_df.loc[0, "c"] == "m n"

    def test_all_columns_normalized_by_default(self):
        """All string columns normalized when columns parameter not specified."""
        df = pd.DataFrame({
            "name": ["John    Doe"],
            "city": ["New    York"]
        })
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        result_df = ar.to_pandas(result)
        assert result_df.loc[0, "name"] == "John Doe"
        assert result_df.loc[0, "city"] == "New York"

    def test_numeric_columns_unaffected(self):
        """Non-string columns are not modified."""
        df = pd.DataFrame({
            "text": ["a    b"],
            "number": [123],
            "price": [45.67]
        })
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        result_df = ar.to_pandas(result)
        assert result_df.loc[0, "number"] == 123
        assert result_df.loc[0, "price"] == 45.67

    def test_returns_arframe(self):
        """Function returns an ArFrame object."""
        df = pd.DataFrame({"text": ["hello    world"]})
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        assert isinstance(result, ar.ArFrame)

    def test_preserves_shape(self):
        """Frame shape (row and column count) is preserved."""
        df = pd.DataFrame({
            "a": ["x    y", "p    q"],
            "b": ["m    n", "k    l"]
        })
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        assert result.shape == (2, 2)

    def test_empty_string_preserved(self):
        """Empty strings remain empty."""
        df = pd.DataFrame({"text": [""]})
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        result_df = ar.to_pandas(result)
        assert result_df.loc[0, "text"] == ""

    def test_mixed_whitespace_types(self):
        """All types of whitespace are normalized together."""
        df = pd.DataFrame({"text": ["  a\t\tb\n\nc  "]})
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        result_df = ar.to_pandas(result)
        assert result_df.loc[0, "text"] == "a b c"
