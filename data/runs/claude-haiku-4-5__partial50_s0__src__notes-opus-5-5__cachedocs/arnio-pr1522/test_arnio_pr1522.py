import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestNormalizeWhitespace:
    """Test suite for arnio.cleaning.normalize_whitespace"""

    def test_basic_whitespace_normalization_dataframe(self):
        """Test basic whitespace normalization with DataFrame input"""
        df = pd.DataFrame({"text": ["  hello   world  "]})
        result = ar.normalize_whitespace(df)
        assert isinstance(result, pd.DataFrame)
        assert result is not df
        assert result["text"].iloc[0] == "hello world"

    def test_basic_whitespace_normalization_arframe(self):
        """Test basic whitespace normalization with ArFrame input"""
        df = pd.DataFrame({"text": ["  hello   world  "]})
        frame = ar.from_pandas(df)
        result = ar.normalize_whitespace(frame)
        assert isinstance(result, ar.ArFrame)
        assert ar.to_pandas(result)["text"].iloc[0] == "hello world"

    def test_multiple_spaces_tabs_newlines_collapse(self):
        """Test that multiple spaces, tabs, and newlines collapse to single space"""
        df = pd.DataFrame({"s": ["a    b", "c\t\td", "e\n\nf"]})
        result = ar.normalize_whitespace(df)
        assert result["s"].tolist() == ["a b", "c d", "e f"]

    def test_leading_trailing_whitespace_stripped(self):
        """Test that leading and trailing whitespace is removed"""
        df = pd.DataFrame({"s": ["  text  ", "\t\nmore\n\t"]})
        result = ar.normalize_whitespace(df)
        assert result["s"].tolist() == ["text", "more"]

    def test_whitespace_only_becomes_empty_string_not_null(self):
        """Test that whitespace-only strings become empty string, not null"""
        df = pd.DataFrame({"s": ["   ", "\t\t", "\n\n"]})
        result = ar.normalize_whitespace(df)
        assert result["s"].tolist() == ["", "", ""]
        assert not result["s"].isna().any()

    def test_non_string_values_pass_through_unchanged(self):
        """Test that int, float, bool, None pass through object columns"""
        df = pd.DataFrame(
            {"mixed": [" text ", 42, 3.14, True, None, np.nan]},
            dtype=object
        )
        result = ar.normalize_whitespace(df)
        result_list = result["mixed"].tolist()
        assert result_list[0] == "text"
        assert result_list[1] == 42
        assert result_list[2] == 3.14
        assert result_list[3] is True
        assert result_list[4] is None
        assert pd.isna(result_list[5])

    def test_pd_na_preserved_in_string_dtype(self):
        """Test that pd.NA is preserved in string dtype columns"""
        df = pd.DataFrame({"s": pd.array([" text ", pd.NA], dtype="string")})
        result = ar.normalize_whitespace(df)
        result_list = result["s"].tolist()
        assert result_list[0] == "text"
        assert pd.isna(result_list[1])

    def test_columns_none_processes_string_columns_only(self):
        """Test that columns=None processes only string/object columns"""
        df = pd.DataFrame({
            "s1": ["  a  b  "],
            "s2": ["  c  d  "],
            "n": [42]
        })
        result = ar.normalize_whitespace(df)
        assert result["s1"].iloc[0] == "a b"
        assert result["s2"].iloc[0] == "c d"
        assert result["n"].iloc[0] == 42

    def test_explicit_columns_list_processes_only_specified(self):
        """Test that explicit columns list processes only those columns"""
        df = pd.DataFrame({
            "s1": ["  a  b  "],
            "s2": ["  c  d  "],
        })
        result = ar.normalize_whitespace(df, columns=["s1"])
        assert result["s1"].iloc[0] == "a b"
        assert result["s2"].iloc[0] == "  c  d  "

    def test_non_string_columns_in_list_skipped_without_error(self):
        """Test that non-string columns in columns list are skipped"""
        df = pd.DataFrame({
            "s": ["  text  "],
            "n": [42]
        })
        result = ar.normalize_whitespace(df, columns=["s", "n"])
        assert result["s"].iloc[0] == "text"
        assert result["n"].iloc[0] == 42

    def test_error_invalid_frame_type(self):
        """Test that invalid frame type raises TypeError"""
        with pytest.raises(TypeError, match="frame must be an ArFrame or a pandas DataFrame"):
            ar.normalize_whitespace({"a": [1]})

    def test_error_columns_as_string(self):
        """Test that columns as string raises TypeError"""
        df = pd.DataFrame({"s": ["text"]})
        with pytest.raises(TypeError, match="columns must be a sequence of column names, not a string"):
            ar.normalize_whitespace(df, columns="s")

    def test_error_columns_as_non_sequence(self):
        """Test that columns as non-sequence (set) raises TypeError"""
        df = pd.DataFrame({"s": ["text"]})
        with pytest.raises(TypeError, match="columns must be a sequence of column names"):
            ar.normalize_whitespace(df, columns={"s"})

    def test_error_missing_column(self):
        """Test that missing columns raises ValueError"""
        df = pd.DataFrame({"a": ["text"]})
        with pytest.raises(ValueError, match="Missing columns for normalize_whitespace"):
            ar.normalize_whitespace(df, columns=["missing"])

    def test_returns_new_object_always(self):
        """Test that a new DataFrame object is always returned"""
        df = pd.DataFrame({"s": ["text"]})
        result = ar.normalize_whitespace(df)
        assert result is not df
        assert isinstance(result, pd.DataFrame)
