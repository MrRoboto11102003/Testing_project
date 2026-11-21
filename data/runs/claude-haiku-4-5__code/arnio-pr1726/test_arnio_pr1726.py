import pytest
import pandas as pd
import arnio as ar


class TestStandardizeMissingTokens:
    
    def test_default_tokens_na_variant(self):
        """Test that default N/A token is converted to NaN."""
        df = pd.DataFrame({"value": [1, 2, "N/A"]})
        result = ar.standardize_missing_tokens(df)
        assert pd.isna(result.loc[2, "value"])
        assert result.loc[0, "value"] == 1
        assert result.loc[1, "value"] == 2
    
    def test_default_tokens_all_variants(self):
        """Test all default tokens are converted to NaN."""
        tokens_to_test = ["N/A", "NA", "n/a", "na", "nan", "-", "none", "nil", "null", "", "?"]
        df = pd.DataFrame({"value": tokens_to_test})
        result = ar.standardize_missing_tokens(df)
        assert result["value"].isna().all()
    
    def test_case_insensitive_matching(self):
        """Test that token matching is case-insensitive."""
        df = pd.DataFrame({"value": ["NA", "Na", "nA", "N/a", "n/A"]})
        result = ar.standardize_missing_tokens(df)
        assert result["value"].isna().all()
    
    def test_whitespace_trimming(self):
        """Test that whitespace is trimmed before checking tokens."""
        df = pd.DataFrame({"value": [" NULL ", "\tNaN\t", "  NA  ", "\nnil\n"]})
        result = ar.standardize_missing_tokens(df)
        assert result["value"].isna().all()
    
    def test_preserve_non_matching_whitespace(self):
        """Test that non-matching values preserve original whitespace."""
        df = pd.DataFrame({"value": [" kept ", "  text  ", "\t preserved\t"]})
        result = ar.standardize_missing_tokens(df)
        assert result.loc[0, "value"] == " kept "
        assert result.loc[1, "value"] == "  text  "
        assert result.loc[2, "value"] == "\t preserved\t"
    
    def test_custom_tokens(self):
        """Test custom tokens list replaces default tokens."""
        df = pd.DataFrame({"value": ["MISSING", "999", "data", "NA"]})
        result = ar.standardize_missing_tokens(df, tokens=["MISSING", "999"])
        assert pd.isna(result.loc[0, "value"])
        assert pd.isna(result.loc[1, "value"])
        assert result.loc[2, "value"] == "data"
        assert result.loc[3, "value"] == "NA"  # Not in custom tokens
    
    def test_subset_single_column(self):
        """Test applying to subset of columns."""
        df = pd.DataFrame({
            "col1": ["NA", "value"],
            "col2": ["NA", "value"]
        })
        result = ar.standardize_missing_tokens(df, subset=["col1"])
        assert pd.isna(result.loc[0, "col1"])
        assert result.loc[0, "col2"] == "NA"
    
    def test_subset_multiple_columns(self):
        """Test applying to multiple columns in subset."""
        df = pd.DataFrame({
            "col1": ["NA", "value"],
            "col2": ["NA", "value"],
            "col3": ["NA", "value"]
        })
        result = ar.standardize_missing_tokens(df, subset=["col1", "col2"])
        assert pd.isna(result.loc[0, "col1"])
        assert pd.isna(result.loc[0, "col2"])
        assert result.loc[0, "col3"] == "NA"
    
    def test_numeric_columns_unchanged(self):
        """Test that numeric values and columns are not affected."""
        df = pd.DataFrame({
            "numeric": [1, 2, 3],
            "string": ["NA", "value", "NA"]
        })
        result = ar.standardize_missing_tokens(df)
        assert list(result["numeric"]) == [1, 2, 3]
        assert pd.isna(result.loc[0, "string"])
    
    def test_mixed_types_in_column(self):
        """Test columns with mixed numeric and string types."""
        df = pd.DataFrame({"value": [1, "NA", None, "text"]})
        result = ar.standardize_missing_tokens(df)
        assert result.loc[0, "value"] == 1
        assert pd.isna(result.loc[1, "value"])
        assert pd.isna(result.loc[2, "value"])
        assert result.loc[3, "value"] == "text"
    
    def test_empty_string_token(self):
        """Test that empty string is treated as a token by default."""
        df = pd.DataFrame({"value": ["", "NA", "text"]})
        result = ar.standardize_missing_tokens(df)
        assert pd.isna(result.loc[0, "value"])
        assert pd.isna(result.loc[1, "value"])
        assert result.loc[2, "value"] == "text"
    
    def test_tokens_as_string_raises_error(self):
        """Test that passing tokens as string raises TypeError."""
        df = pd.DataFrame({"value": ["NA", "text"]})
        with pytest.raises(TypeError, match="tokens must be a list of strings"):
            ar.standardize_missing_tokens(df, tokens="NA")
    
    def test_subset_as_string_raises_error(self):
        """Test that passing subset as string raises TypeError."""
        df = pd.DataFrame({"value": ["NA", "text"]})
        with pytest.raises(TypeError, match="subset must be a list of column names"):
            ar.standardize_missing_tokens(df, subset="value")
    
    def test_tokens_not_list_raises_error(self):
        """Test that passing tokens as non-list raises TypeError."""
        df = pd.DataFrame({"value": ["NA", "text"]})
        with pytest.raises(TypeError, match="tokens must be a list of strings"):
            ar.standardize_missing_tokens(df, tokens=("NA",))
    
    def test_tokens_with_non_string_item_raises_error(self):
        """Test that tokens list with non-string items raises TypeError."""
        df = pd.DataFrame({"value": ["NA", "text"]})
        with pytest.raises(TypeError, match="tokens must be a list of strings"):
            ar.standardize_missing_tokens(df, tokens=["NA", 123])
    
    def test_unknown_column_in_subset_raises_error(self):
        """Test that unknown columns in subset raise ValueError."""
        df = pd.DataFrame({"value": ["NA", "text"]})
        with pytest.raises(ValueError, match="Unknown columns in subset"):
            ar.standardize_missing_tokens(df, subset=["nonexistent"])
