import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestNormalizeWhitespace:
    """Tests for arnio.cleaning.normalize_whitespace function."""
    
    # Basic functionality
    def test_collapse_multiple_spaces(self):
        """Multiple spaces are collapsed to one and edges stripped."""
        df = pd.DataFrame({"s": ["  x    y  "]})
        result = ar.cleaning.normalize_whitespace(df)
        assert result["s"].tolist() == ["x y"]
    
    def test_collapse_mixed_whitespace(self):
        """Tabs, newlines, and spaces are collapsed to single space."""
        df = pd.DataFrame({"s": ["a\t\n\t b", "  c\n  d  "]})
        result = ar.cleaning.normalize_whitespace(df)
        assert result["s"].tolist() == ["a b", "c d"]
    
    def test_whitespace_only_to_empty_string(self):
        """Whitespace-only strings become empty string, not null."""
        df = pd.DataFrame({"s": ["   ", "\t\n  ", ""]})
        result = ar.cleaning.normalize_whitespace(df)
        assert result["s"].tolist() == ["", "", ""]
    
    # Column selection
    def test_columns_none_selects_string_columns(self):
        """columns=None processes all string/object columns, skips numeric."""
        df = pd.DataFrame({
            "s1": ["  a  "],
            "s2": ["  b  "],
            "n": [1]
        })
        result = ar.cleaning.normalize_whitespace(df)
        assert result["s1"].tolist() == ["a"]
        assert result["s2"].tolist() == ["b"]
        assert result["n"].tolist() == [1]
    
    def test_explicit_columns(self):
        """Explicit columns parameter processes only listed string columns."""
        df = pd.DataFrame({
            "s1": ["  a  "],
            "s2": ["  b  "],
            "n": [1]
        })
        result = ar.cleaning.normalize_whitespace(df, columns=["s1", "n"])
        assert result["s1"].tolist() == ["a"]
        assert result["s2"].tolist() == ["  b  "]  # unchanged
        assert result["n"].tolist() == [1]  # numeric column skipped silently
    
    # Non-string values
    def test_non_string_values_pass_through(self):
        """Non-string values (int, float, None) pass through unchanged."""
        df = pd.DataFrame({"s": [" x ", 3, 1.5, None, " y "]}, dtype=object)
        result = ar.cleaning.normalize_whitespace(df)
        assert result["s"].tolist()[0] == "x"
        assert result["s"].tolist()[1] == 3
        assert result["s"].tolist()[2] == 1.5
        assert pd.isna(result["s"].tolist()[3])
        assert result["s"].tolist()[4] == "y"
    
    def test_pandas_na_preserved(self):
        """pd.NA values in string dtype columns are preserved."""
        df = pd.DataFrame({"s": [" a ", pd.NA, " b "]}, dtype="string")
        result = ar.cleaning.normalize_whitespace(df)
        assert result["s"].tolist()[0] == "a"
        assert pd.isna(result["s"].tolist()[1])
        assert result["s"].tolist()[2] == "b"
    
    # Frame types
    def test_dataframe_input_output(self):
        """DataFrame input returns new DataFrame (input not mutated)."""
        df = pd.DataFrame({"s": [" x  y "]})
        result = ar.cleaning.normalize_whitespace(df)
        assert isinstance(result, pd.DataFrame)
        assert result is not df
        assert df["s"].tolist()[0] == " x  y "  # input not mutated
    
    def test_arframe_input_output(self):
        """ArFrame input returns ArFrame output."""
        df = pd.DataFrame({"s": [" x  y "]})
        frame = ar.from_pandas(df)
        result = ar.cleaning.normalize_whitespace(frame)
        assert isinstance(result, ar.ArFrame)
    
    # Error handling
    def test_invalid_frame_type_error(self):
        """Invalid frame type raises TypeError."""
        with pytest.raises(TypeError, match="must be an ArFrame or a pandas DataFrame"):
            ar.cleaning.normalize_whitespace({"a": [1]})
    
    def test_columns_string_error(self):
        """columns as string raises TypeError."""
        df = pd.DataFrame({"s": ["  a  "]})
        with pytest.raises(TypeError, match="must be a sequence"):
            ar.cleaning.normalize_whitespace(df, columns="s")
    
    def test_columns_set_error(self):
        """columns as set raises TypeError."""
        df = pd.DataFrame({"s": ["  a  "]})
        with pytest.raises(TypeError, match="must be a sequence"):
            ar.cleaning.normalize_whitespace(df, columns={"s"})
    
    def test_columns_non_string_item_error(self):
        """columns with non-string items raises TypeError."""
        df = pd.DataFrame({"s": ["  a  "]})
        with pytest.raises(TypeError, match="must contain only string"):
            ar.cleaning.normalize_whitespace(df, columns=["s", 1])
    
    def test_missing_column_error(self):
        """Missing column raises ValueError with specific message format."""
        df = pd.DataFrame({"a": [1], "b": [2]})
        with pytest.raises(ValueError, match=r"Missing columns for normalize_whitespace.*\['zz'\]"):
            ar.cleaning.normalize_whitespace(df, columns=["zz"])
    
    # Edge cases
    def test_empty_dataframe(self):
        """Zero-row DataFrame processes without error and preserves columns."""
        df = pd.DataFrame({"s": pd.Series([], dtype=object)})
        result = ar.cleaning.normalize_whitespace(df)
        assert len(result) == 0
        assert "s" in result.columns
    
    def test_unicode_whitespace(self):
        """Unicode whitespace (non-breaking space U+00A0) is normalized."""
        df = pd.DataFrame({"s": ["a\u00a0\u00a0b"]})
        result = ar.cleaning.normalize_whitespace(df)
        assert result["s"].tolist()[0] == "a b"
