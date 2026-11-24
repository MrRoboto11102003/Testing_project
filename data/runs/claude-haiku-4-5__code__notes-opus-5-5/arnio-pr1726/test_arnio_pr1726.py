import pytest
import pandas as pd
import numpy as np
import re
import arnio as ar


class TestStandardizeMissingTokens:
    """Tests for arnio.cleaning.standardize_missing_tokens function."""
    
    def test_default_tokens_replaced(self):
        """Test that default missing tokens are replaced with NaN."""
        df = pd.DataFrame({"value": ["N/A", "NA", "n/a", "nan", "-", "none", "nil", "null", "", "?"]})
        result = ar.standardize_missing_tokens(df)
        assert all(pd.isna(result["value"]))
    
    def test_case_insensitive_whitespace_trimming(self):
        """Test case-insensitive matching and whitespace trimming."""
        df = pd.DataFrame({"value": [" N/A ", "\tNaN\t", "NONE ", "kept "]})
        result = ar.standardize_missing_tokens(df)
        assert pd.isna(result["value"].iloc[0])
        assert pd.isna(result["value"].iloc[1])
        assert pd.isna(result["value"].iloc[2])
        assert result["value"].iloc[3] == "kept "
    
    def test_non_string_values_pass_through(self):
        """Test that non-string values remain unchanged."""
        df = pd.DataFrame({"value": [1, 2.5, None, np.nan, True]})
        result = ar.standardize_missing_tokens(df)
        assert result["value"].iloc[0] == 1
        assert result["value"].iloc[1] == 2.5
        assert result["value"].iloc[2] is None
        assert pd.isna(result["value"].iloc[3])
        assert result["value"].iloc[4] is True
    
    def test_custom_tokens_replace_defaults(self):
        """Test that custom tokens completely replace defaults."""
        df = pd.DataFrame({"value": ["N/A", "CUSTOM"]})
        result = ar.standardize_missing_tokens(df, tokens=["CUSTOM"])
        assert result["value"].iloc[0] == "N/A"  # Not replaced
        assert pd.isna(result["value"].iloc[1])  # Replaced
    
    def test_empty_tokens_list(self):
        """Test that empty tokens list replaces nothing."""
        df = pd.DataFrame({"value": ["N/A", "NA", "-"]})
        result = ar.standardize_missing_tokens(df, tokens=[])
        assert all(result["value"] == df["value"])
    
    def test_subset_single_column(self):
        """Test subset parameter with single column."""
        df = pd.DataFrame({"a": ["N/A"], "b": ["N/A"]})
        result = ar.standardize_missing_tokens(df, subset=["a"])
        assert pd.isna(result["a"].iloc[0])
        assert result["b"].iloc[0] == "N/A"  # Unchanged
    
    def test_subset_empty_list(self):
        """Test that empty subset list doesn't modify anything."""
        df = pd.DataFrame({"a": ["N/A"]})
        result = ar.standardize_missing_tokens(df, subset=[])
        assert result["a"].iloc[0] == "N/A"
    
    def test_input_not_mutated(self):
        """Test that input DataFrame is not modified."""
        df = pd.DataFrame({"value": ["N/A"]})
        original = df.loc[0, "value"]
        ar.standardize_missing_tokens(df)
        assert df.loc[0, "value"] == original
    
    def test_error_subset_is_string(self):
        """Test that subset as string raises TypeError with helpful message."""
        df = pd.DataFrame({"col": ["N/A"]})
        with pytest.raises(TypeError, match=re.escape("subset must be a list of column names, not a string. Did you mean subset=['col']?")):
            ar.standardize_missing_tokens(df, subset="col")
    
    def test_error_tokens_is_string(self):
        """Test that tokens as string raises TypeError with helpful message."""
        df = pd.DataFrame({"col": ["N/A"]})
        with pytest.raises(TypeError, match=re.escape("tokens must be a list of strings, not a bare string. Did you mean tokens=['N/A']?")):
            ar.standardize_missing_tokens(df, tokens="N/A")
    
    def test_error_tokens_not_list(self):
        """Test that tokens as non-list raises TypeError."""
        df = pd.DataFrame({"col": ["N/A"]})
        with pytest.raises(TypeError, match=re.escape("tokens must be a list of strings, got 'tuple'")):
            ar.standardize_missing_tokens(df, tokens=("N/A",))
    
    def test_error_non_string_token(self):
        """Test that non-string token item raises TypeError."""
        df = pd.DataFrame({"col": ["N/A"]})
        with pytest.raises(TypeError, match=re.escape("tokens must be a list of strings, got 'int' item")):
            ar.standardize_missing_tokens(df, tokens=["N/A", 1])
    
    def test_error_missing_column_in_subset(self):
        """Test that missing column in subset raises ValueError."""
        df = pd.DataFrame({"a": ["N/A"]})
        with pytest.raises(ValueError, match=re.escape("Unknown columns in subset: ['missing']")):
            ar.standardize_missing_tokens(df, subset=["missing"])
    
    def test_arframe_in_arframe_out(self):
        """Test that ArFrame input returns ArFrame output."""
        df = pd.DataFrame({"value": ["N/A", "valid"]})
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame)
        assert isinstance(result, ar.ArFrame)
        result_df = ar.to_pandas(result)
        assert pd.isna(result_df["value"].iloc[0])
        assert result_df["value"].iloc[1] == "valid"
    
    def test_whitespace_only_strings(self):
        """Test that whitespace-only strings become NaN (empty string is a default token)."""
        df = pd.DataFrame({"value": ["   ", "\t", "\n"]})
        result = ar.standardize_missing_tokens(df)
        assert all(pd.isna(result["value"]))
