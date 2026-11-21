import pytest
import pandas as pd
import numpy as np
import arnio as ar
import re


class TestParseBoolStrings:
    """Unit tests for ar.parse_bool_strings"""

    def test_default_true_values_converted(self):
        """Test that default true values (true, yes, y, 1) are converted."""
        df = pd.DataFrame({"flag": ["true", "yes", "y", "1"]})
        frame = ar.from_pandas(df)
        result = ar.parse_bool_strings(frame)
        result_df = ar.to_pandas(result)
        assert result_df["flag"].tolist() == [True, True, True, True]

    def test_default_false_values_converted(self):
        """Test that default false values (false, no, n, 0) are converted."""
        df = pd.DataFrame({"flag": ["false", "no", "n", "0"]})
        frame = ar.from_pandas(df)
        result = ar.parse_bool_strings(frame)
        result_df = ar.to_pandas(result)
        assert result_df["flag"].tolist() == [False, False, False, False]

    def test_case_insensitive_matching(self):
        """Test that matching is case-insensitive."""
        df = pd.DataFrame({"flag": ["TRUE", "Yes", "FALSE", "NO"]})
        frame = ar.from_pandas(df)
        result = ar.parse_bool_strings(frame)
        result_df = ar.to_pandas(result)
        assert result_df["flag"].tolist() == [True, True, False, False]

    def test_whitespace_stripped_during_matching(self):
        """Test that whitespace is stripped before matching."""
        df = pd.DataFrame({"flag": ["  true  ", " yes ", " no ", "  false  "]})
        frame = ar.from_pandas(df)
        result = ar.parse_bool_strings(frame)
        result_df = ar.to_pandas(result)
        assert result_df["flag"].tolist() == [True, True, False, False]

    def test_unsupported_values_preserved(self):
        """Test that unsupported string values are preserved unchanged."""
        df = pd.DataFrame({"flag": ["yes", "maybe", "no", "unknown"]})
        frame = ar.from_pandas(df)
        result = ar.parse_bool_strings(frame)
        result_df = ar.to_pandas(result)
        assert result_df["flag"][0] == True
        assert result_df["flag"][1] == "maybe"
        assert result_df["flag"][2] == False
        assert result_df["flag"][3] == "unknown"

    def test_null_values_preserved(self):
        """Test that null/NaN values are preserved."""
        df = pd.DataFrame({"flag": ["yes", None, "no", np.nan]})
        frame = ar.from_pandas(df)
        result = ar.parse_bool_strings(frame)
        result_df = ar.to_pandas(result)
        assert result_df["flag"][0] == True
        assert pd.isna(result_df["flag"][1])
        assert result_df["flag"][2] == False
        assert pd.isna(result_df["flag"][3])

    def test_custom_true_values_replace_defaults(self):
        """Test that custom true_values replace (not merge with) defaults."""
        df = pd.DataFrame({"flag": ["si", "oui", "yes", "no"]})
        frame = ar.from_pandas(df)
        result = ar.parse_bool_strings(frame, true_values={"si", "oui"})
        result_df = ar.to_pandas(result)
        # "yes" should not match because custom true_values replaces defaults
        assert result_df["flag"][0] == True
        assert result_df["flag"][1] == True
        assert result_df["flag"][2] == "yes"
        assert result_df["flag"][3] == False  # false default still applies

    def test_custom_false_values_replace_defaults(self):
        """Test that custom false_values replace defaults."""
        df = pd.DataFrame({"flag": ["yes", "no", "nein"]})
        frame = ar.from_pandas(df)
        result = ar.parse_bool_strings(frame, false_values={"no", "nein"})
        result_df = ar.to_pandas(result)
        assert result_df["flag"][0] == True  # yes from default true_values
        assert result_df["flag"][1] == False
        assert result_df["flag"][2] == False

    def test_subset_affects_only_named_columns(self):
        """Test that subset parameter only affects specified columns."""
        df = pd.DataFrame({
            "converted": ["yes", "no"],
            "untouched": ["yes", "no"]
        })
        frame = ar.from_pandas(df)
        result = ar.parse_bool_strings(frame, subset=["converted"])
        result_df = ar.to_pandas(result)
        assert result_df["converted"].tolist() == [True, False]
        assert result_df["untouched"].tolist() == ["yes", "no"]

    def test_error_true_values_bare_string(self):
        """Test that bare string for true_values raises TypeError."""
        df = pd.DataFrame({"flag": ["yes"]})
        frame = ar.from_pandas(df)
        with pytest.raises(TypeError, match="true_values must be a set/list/tuple of strings, not a bare string"):
            ar.parse_bool_strings(frame, true_values="yes")

    def test_error_false_values_bare_string(self):
        """Test that bare string for false_values raises TypeError."""
        df = pd.DataFrame({"flag": ["no"]})
        frame = ar.from_pandas(df)
        with pytest.raises(TypeError, match="false_values must be a set/list/tuple of strings, not a bare string"):
            ar.parse_bool_strings(frame, false_values="no")

    def test_error_true_values_non_string_element(self):
        """Test that non-string elements in true_values raise TypeError."""
        df = pd.DataFrame({"flag": ["yes"]})
        frame = ar.from_pandas(df)
        with pytest.raises(TypeError, match="true_values must contain only strings, got int"):
            ar.parse_bool_strings(frame, true_values=["yes", 1])

    def test_error_overlap_after_normalization(self):
        """Test that overlap between true_values and false_values raises ValueError."""
        df = pd.DataFrame({"flag": ["yes"]})
        frame = ar.from_pandas(df)
        with pytest.raises(ValueError, match=re.escape("true_values and false_values overlap after normalization: {'yes'}")):
            ar.parse_bool_strings(frame, true_values={"yes"}, false_values={"yes"})

    def test_error_subset_nonexistent_column(self):
        """Test that nonexistent column in subset raises ValueError."""
        df = pd.DataFrame({"flag": ["yes"]})
        frame = ar.from_pandas(df)
        with pytest.raises(ValueError, match="Columns not found in frame"):
            ar.parse_bool_strings(frame, subset=["nonexistent"])

    def test_empty_true_values_collection(self):
        """Test that empty true_values collection is allowed."""
        df = pd.DataFrame({"flag": ["yes", "true", "no"]})
        frame = ar.from_pandas(df)
        result = ar.parse_bool_strings(frame, true_values=set())
        result_df = ar.to_pandas(result)
        # Nothing matches true_values, only false defaults
        assert result_df["flag"][0] == "yes"
        assert result_df["flag"][1] == "true"
        assert result_df["flag"][2] == False
