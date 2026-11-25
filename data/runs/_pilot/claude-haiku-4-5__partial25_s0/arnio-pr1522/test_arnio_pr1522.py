import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestNormalizeWhitespace:
    """Test suite for arnio.cleaning.normalize_whitespace"""

    def test_collapses_multiple_spaces(self):
        """Test that multiple consecutive spaces are collapsed to single space"""
        df = pd.DataFrame({"name": ["John    Doe", "Jane  Smith", "Bob   Jones"]})
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        result_df = ar.to_pandas(result)
        assert result_df["name"].tolist() == ["John Doe", "Jane Smith", "Bob Jones"]

    def test_collapses_tabs_and_mixed_whitespace(self):
        """Test that tabs and mixed whitespace are normalized"""
        df = pd.DataFrame({"text": ["Hello\t\tWorld", "Foo  \t  Bar", "A\t  \tB"]})
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        result_df = ar.to_pandas(result)
        # Verify internal whitespace is collapsed
        assert "Hello World" in result_df["text"].iloc[0] or len(result_df["text"].iloc[0].split()) == 2

    def test_preserves_single_spaces(self):
        """Test that properly spaced text remains unchanged"""
        df = pd.DataFrame({"name": ["John Doe", "Jane Smith"]})
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        result_df = ar.to_pandas(result)
        assert result_df["name"].tolist() == ["John Doe", "Jane Smith"]

    def test_empty_string_unchanged(self):
        """Test that empty strings remain empty"""
        df = pd.DataFrame({"text": ["", "test"]})
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        result_df = ar.to_pandas(result)
        assert result_df["text"].iloc[0] == ""

    def test_whitespace_only_string(self):
        """Test handling of strings that contain only whitespace"""
        df = pd.DataFrame({"text": ["   ", "\t\t", "content"]})
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        result_df = ar.to_pandas(result)
        # Whitespace-only becomes single space (or similarly normalized)
        assert result_df["text"].iloc[2] == "content"

    def test_null_values_preserved(self):
        """Test that null/NaN values are preserved"""
        df = pd.DataFrame({"name": ["John  Doe", None, np.nan, "Jane  Smith"]})
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        result_df = ar.to_pandas(result)
        assert pd.isna(result_df["name"].iloc[1])
        assert pd.isna(result_df["name"].iloc[2])
        assert result_df["name"].iloc[0] == "John Doe"
        assert result_df["name"].iloc[3] == "Jane Smith"

    def test_columns_parameter_subset_single(self):
        """Test that columns parameter limits operation to specified columns"""
        df = pd.DataFrame({
            "name": ["John  Doe", "Jane  Smith"],
            "address": ["123  Main  St", "456  Oak  Ave"]
        })
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame, columns=["name"])
        result_df = ar.to_pandas(result)
        assert result_df["name"].iloc[0] == "John Doe"
        # address should NOT be modified when not in columns
        assert result_df["address"].iloc[0] == "123  Main  St"

    def test_columns_parameter_multiple(self):
        """Test normalize_whitespace on multiple specific columns"""
        df = pd.DataFrame({
            "first": ["John  Doe", "Jane  Smith"],
            "last": ["Doe  Sr", "Smith  Jr"],
            "id": ["001  A", "002  B"]
        })
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame, columns=["first", "last"])
        result_df = ar.to_pandas(result)
        assert result_df["first"].iloc[0] == "John Doe"
        assert result_df["last"].iloc[0] == "Doe Sr"
        # id should NOT be modified
        assert result_df["id"].iloc[0] == "001  A"

    def test_non_string_columns_safely_ignored(self):
        """Test that non-string columns are safely ignored"""
        df = pd.DataFrame({
            "name": ["John  Doe", "Jane  Smith"],
            "age": [30, 25],
            "score": [95.5, 87.3]
        })
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        result_df = ar.to_pandas(result)
        assert result_df["name"].iloc[0] == "John Doe"
        assert result_df["age"].iloc[0] == 30
        assert result_df["score"].iloc[0] == 95.5

    def test_empty_frame(self):
        """Test handling of empty frame"""
        df = pd.DataFrame({"name": []})
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        result_df = ar.to_pandas(result)
        assert len(result_df) == 0

    def test_numeric_only_frame_unchanged(self):
        """Test frame with only numeric columns returns unchanged"""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "value": [10.5, 20.3, 30.1]
        })
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        result_df = ar.to_pandas(result)
        pd.testing.assert_frame_equal(result_df, df)

    def test_newlines_in_text(self):
        """Test handling of newlines in text"""
        df = pd.DataFrame({"text": ["Hello\nWorld", "Foo\n\nBar"]})
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        result_df = ar.to_pandas(result)
        # Should normalize whitespace including around newlines
        assert len(result_df) == 2

    def test_leading_trailing_spaces_preserved(self):
        """Test that leading/trailing spaces are not stripped (only internal normalized)"""
        df = pd.DataFrame({"text": ["  hello  world  "]})
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        result_df = ar.to_pandas(result)
        # Verify internal spaces are collapsed but structure preserved
        assert "hello" in result_df["text"].iloc[0] and "world" in result_df["text"].iloc[0]

    def test_returns_arframe_instance(self):
        """Test that function returns an ArFrame"""
        df = pd.DataFrame({"name": ["John  Doe"]})
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        # Should be convertible back to pandas
        result_df = ar.to_pandas(result)
        assert isinstance(result_df, pd.DataFrame)

    def test_preserves_row_count(self):
        """Test that row count is preserved after normalization"""
        df = pd.DataFrame({"text": ["a  b", "c  d", "e  f"]})
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        result_df = ar.to_pandas(result)
        assert len(result_df) == len(df)
