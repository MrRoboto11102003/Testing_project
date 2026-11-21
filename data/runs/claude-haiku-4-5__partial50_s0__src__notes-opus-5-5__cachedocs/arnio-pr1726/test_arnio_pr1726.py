import pytest
import pandas as pd
import numpy as np
import arnio as ar
import re


class TestStandardizeMissingTokens:
    """Tests for arnio.cleaning.standardize_missing_tokens."""

    def test_default_tokens_basic(self):
        """Default tokens (NA, null, N/A, etc.) are replaced with NaN."""
        df = pd.DataFrame({
            "value": ["NA", "null", "N/A", "none", "-", "?", "valid", ""]
        })
        result = ar.standardize_missing_tokens(df)
        
        assert pd.isna(result["value"][0])  # "NA"
        assert pd.isna(result["value"][1])  # "null"
        assert pd.isna(result["value"][2])  # "N/A"
        assert pd.isna(result["value"][3])  # "none"
        assert pd.isna(result["value"][4])  # "-"
        assert pd.isna(result["value"][5])  # "?"
        assert result["value"][6] == "valid"
        assert pd.isna(result["value"][7])  # ""

    def test_case_insensitive_matching(self):
        """Token matching is case-insensitive."""
        df = pd.DataFrame({
            "value": ["NULL", "Null", "null", "NaN", "NONE", "none"]
        })
        result = ar.standardize_missing_tokens(df)
        
        # All should be NaN
        for i in range(len(result)):
            assert pd.isna(result["value"][i])

    def test_whitespace_stripped_matching_preserved_non_matching(self):
        """Whitespace stripped for matching but preserved for non-matching strings."""
        df = pd.DataFrame({
            "value": ["  NULL  ", "\tNaN\t", "   ", " kept ", "x"]
        })
        result = ar.standardize_missing_tokens(df)
        
        assert pd.isna(result["value"][0])  # "  NULL  " -> NaN
        assert pd.isna(result["value"][1])  # "\tNaN\t" -> NaN
        assert pd.isna(result["value"][2])  # "   " (whitespace-only -> "") -> NaN
        assert result["value"][3] == " kept "  # whitespace preserved
        assert result["value"][4] == "x"

    def test_custom_tokens_replace_defaults(self):
        """Custom tokens replace defaults, not add to them."""
        df = pd.DataFrame({
            "value": ["NA", "MISSING", "valid", "null"]
        })
        result = ar.standardize_missing_tokens(df, tokens=["MISSING"])
        
        assert result["value"][0] == "NA"  # Not in custom list
        assert pd.isna(result["value"][1])  # In custom list
        assert result["value"][2] == "valid"
        assert result["value"][3] == "null"  # Defaults not used

    def test_custom_tokens_case_insensitive(self):
        """Custom tokens are matched case-insensitively."""
        df = pd.DataFrame({
            "value": ["missing", "Missing", "MISSING", "valid"]
        })
        result = ar.standardize_missing_tokens(df, tokens=["MISSING"])
        
        for i in range(3):
            assert pd.isna(result["value"][i])
        assert result["value"][3] == "valid"

    def test_empty_tokens_list_no_replacement(self):
        """Empty tokens list means nothing is replaced."""
        df = pd.DataFrame({
            "value": ["NA", "null", "-", "?", "valid"]
        })
        result = ar.standardize_missing_tokens(df, tokens=[])
        
        for i in range(4):
            assert result["value"][i] == df["value"][i]

    def test_subset_single_column(self):
        """Only specified column is processed when subset provided."""
        df = pd.DataFrame({
            "a": ["NA", "null", "valid"],
            "b": ["NA", "null", "valid"]
        })
        result = ar.standardize_missing_tokens(df, subset=["a"])
        
        assert pd.isna(result["a"][0])
        assert pd.isna(result["a"][1])
        assert result["a"][2] == "valid"
        
        # Column b unchanged
        assert result["b"][0] == "NA"
        assert result["b"][1] == "null"
        assert result["b"][2] == "valid"

    def test_subset_multiple_columns(self):
        """subset can specify multiple columns."""
        df = pd.DataFrame({
            "x": ["NA", "valid"],
            "y": ["null", "valid"],
            "z": ["NA", "valid"]
        })
        result = ar.standardize_missing_tokens(df, subset=["x", "y"])
        
        assert pd.isna(result["x"][0])
        assert pd.isna(result["y"][0])
        assert result["z"][0] == "NA"  # z not in subset

    def test_subset_empty_list(self):
        """Empty subset means no columns are processed."""
        df = pd.DataFrame({"value": ["NA", "null", "valid"]})
        result = ar.standardize_missing_tokens(df, subset=[])
        
        pd.testing.assert_frame_equal(result, df)

    def test_non_string_values_pass_through(self):
        """Non-string values (int, float, None, pd.NA, bool) pass through unchanged."""
        df = pd.DataFrame({
            "mixed": [1, 2.5, None, pd.NA, True, False, "NA", "valid"]
        }, dtype=object)
        result = ar.standardize_missing_tokens(df)
        
        assert result["mixed"][0] == 1
        assert result["mixed"][1] == 2.5
        assert result["mixed"][2] is None
        assert pd.isna(result["mixed"][3])  # pd.NA passes through
        assert result["mixed"][4] is True
        assert result["mixed"][5] is False
        assert pd.isna(result["mixed"][6])  # "NA" replaced
        assert result["mixed"][7] == "valid"

    def test_input_dataframe_not_mutated(self):
        """Input DataFrame is not mutated."""
        df = pd.DataFrame({"value": ["NA", "null", "valid"]})
        df_original = df.copy()
        result = ar.standardize_missing_tokens(df)
        
        pd.testing.assert_frame_equal(df, df_original)
        assert pd.isna(result["value"][0])

    def test_arframe_in_arframe_out(self):
        """ArFrame input returns ArFrame output."""
        df = pd.DataFrame({"value": ["NA", "valid"]})
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame)
        
        assert isinstance(result, ar.ArFrame)
        result_df = ar.to_pandas(result)
        assert pd.isna(result_df["value"][0])
        assert result_df["value"][1] == "valid"

    def test_dataframe_in_dataframe_out_different_object(self):
        """DataFrame input returns new DataFrame (not same object)."""
        df = pd.DataFrame({"value": ["NA", "valid"]})
        result = ar.standardize_missing_tokens(df)
        
        assert isinstance(result, pd.DataFrame)
        assert result is not df

    def test_error_subset_string_helpful_message(self):
        """subset as string raises helpful TypeError with suggestion."""
        df = pd.DataFrame({"mycol": ["NA"]})
        with pytest.raises(
            TypeError,
            match=re.escape("subset must be a list of column names, not a string. Did you mean subset=['mycol']?")
        ):
            ar.standardize_missing_tokens(df, subset="mycol")

    def test_error_tokens_string_helpful_message(self):
        """tokens as string raises helpful TypeError with suggestion."""
        df = pd.DataFrame({"value": ["NA"]})
        with pytest.raises(
            TypeError,
            match=re.escape("tokens must be a list of strings, not a bare string. Did you mean tokens=['NULL']?")
        ):
            ar.standardize_missing_tokens(df, tokens="NULL")

    def test_error_tokens_not_list_type(self):
        """tokens must be a list, not tuple or other iterable."""
        df = pd.DataFrame({"value": ["NA"]})
        with pytest.raises(TypeError, match=re.escape("tokens must be a list of strings, got 'tuple'")):
            ar.standardize_missing_tokens(df, tokens=("NA",))

    def test_error_tokens_item_not_string(self):
        """Non-string token item raises TypeError."""
        df = pd.DataFrame({"value": ["NA"]})
        with pytest.raises(TypeError, match=re.escape("tokens must be a list of strings, got 'int' item")):
            ar.standardize_missing_tokens(df, tokens=["NA", 42])

    def test_error_missing_column_in_subset(self):
        """Column missing in DataFrame raises ValueError with column name."""
        df = pd.DataFrame({"a": ["NA"]})
        with pytest.raises(ValueError, match="Unknown columns in subset:"):
            ar.standardize_missing_tokens(df, subset=["missing_col"])
