import pytest
import pandas as pd
import numpy as np
import re
import arnio as ar


class TestCoalesceColumns:
    """Tests for arnio.cleaning.coalesce_columns"""

    def test_basic_first_non_null_selection(self):
        """First non-null value from subset should be selected."""
        df = pd.DataFrame({
            "a": [1.0, np.nan, np.nan],
            "b": [10.0, 2.0, np.nan]
        })
        result = ar.coalesce_columns(df, subset=["a", "b"], output_column="c")
        assert list(result.columns) == ["a", "b", "c"]
        assert result["c"].iloc[0] == 1.0
        assert result["c"].iloc[1] == 2.0
        assert pd.isna(result["c"].iloc[2])

    def test_subset_order_determines_priority(self):
        """Subset order matters: first column in order has highest priority."""
        df = pd.DataFrame({"a": [1.0], "b": [9.0]})
        r1 = ar.coalesce_columns(df, subset=["a", "b"], output_column="c")
        r2 = ar.coalesce_columns(df, subset=["b", "a"], output_column="c")
        assert r1["c"].iloc[0] == 1.0
        assert r2["c"].iloc[0] == 9.0

    def test_coalesce_strings_and_multiple_fallthrough(self):
        """Should handle strings and fallthrough across three columns."""
        df = pd.DataFrame({
            "a": [np.nan, None, "z"],
            "b": [np.nan, 5.0, "z2"],
            "c": ["x", 6.0, "z3"]
        })
        result = ar.coalesce_columns(df, subset=["a", "b", "c"], output_column="out")
        assert result["out"].iloc[0] == "x"
        assert result["out"].iloc[1] == 5.0
        assert result["out"].iloc[2] == "z"

    def test_zero_rows_and_input_not_mutated(self):
        """Empty frame should preserve row count; input should not be modified."""
        df = pd.DataFrame({"a": [], "b": []})
        result = ar.coalesce_columns(df, subset=["a", "b"], output_column="c")
        assert len(result) == 0 and list(result.columns) == ["a", "b", "c"]
        assert list(df.columns) == ["a", "b"]

    def test_error_bad_frame_type(self):
        """Non-DataFrame/ArFrame input should raise TypeError."""
        with pytest.raises(TypeError, match="ArFrame or a pandas DataFrame"):
            ar.coalesce_columns({"a": [1]}, subset=["a"])

    def test_error_output_column_validation(self):
        """Empty/whitespace output_column should raise ValueError before subset checks."""
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="output_column must be a non-empty string"):
            ar.coalesce_columns(df, subset="a", output_column="")
        with pytest.raises(ValueError, match="output_column must be a non-empty string"):
            ar.coalesce_columns(df, subset=["a"], output_column="   ")

    def test_error_subset_sequence_and_items(self):
        """subset must be a sequence of strings with no duplicates."""
        df = pd.DataFrame({"a": [1], "b": [2]})
        with pytest.raises(TypeError, match="subset must be a sequence"):
            ar.coalesce_columns(df, subset="a")
        with pytest.raises(TypeError, match="subset must be a sequence"):
            ar.coalesce_columns(df, subset={"a", "b"})
        with pytest.raises(TypeError, match="subset must contain only string"):
            ar.coalesce_columns(df, subset=["a", 1])
        with pytest.raises(ValueError, match="duplicate"):
            ar.coalesce_columns(df, subset=["a", "a"])

    def test_error_subset_empty_missing_or_existing_output(self):
        """subset must be non-empty with valid columns; output_column must not exist."""
        df = pd.DataFrame({"a": [1], "b": [2]})
        with pytest.raises(ValueError, match="at least one column"):
            ar.coalesce_columns(df, subset=[])
        with pytest.raises(KeyError, match=re.escape("Missing columns")):
            ar.coalesce_columns(df, subset=["a", "zz"])
        with pytest.raises(ValueError, match="already exists"):
            ar.coalesce_columns(df, subset=["a"], output_column="b")


class TestStandardizeMissingTokens:
    """Tests for arnio.cleaning.standardize_missing_tokens"""

    def test_default_tokens_replaced_with_nan(self):
        """Default tokens should become NaN; matching is case-insensitive and trims whitespace."""
        df = pd.DataFrame({
            "v": [" NULL ", "\tNaN\t", "N/A", "NA", "-", "none", "   ", "kept"]
        })
        result = ar.standardize_missing_tokens(df)
        for i in range(7):
            assert pd.isna(result["v"][i]), f"Index {i} should be NaN"
        assert result["v"][7] == "kept"

    def test_custom_tokens_override_defaults(self):
        """Custom tokens list replaces defaults completely."""
        df = pd.DataFrame({"v": ["MISSING", "kept", "N/A"]})
        result = ar.standardize_missing_tokens(df, tokens=["MISSING"])
        assert pd.isna(result["v"][0])
        assert result["v"][1] == "kept"
        assert result["v"][2] == "N/A"

    def test_subset_restricts_to_specific_columns(self):
        """subset parameter applies replacement only to named columns."""
        df = pd.DataFrame({"a": ["NA", "x"], "b": ["NA", "y"]})
        result = ar.standardize_missing_tokens(df, subset=["a"])
        assert pd.isna(result["a"][0])
        assert result["b"][0] == "NA"

    def test_non_string_values_unchanged(self):
        """Non-string values (int, float, None, bool) pass through unchanged."""
        df = pd.DataFrame({"v": [1, 2.0, "NA", None, True]})
        result = ar.standardize_missing_tokens(df)
        assert result["v"][0] == 1
        assert result["v"][1] == 2.0
        assert pd.isna(result["v"][2])
        assert result["v"][3] is None
        assert result["v"][4] is True

    def test_empty_tokens_and_empty_subset(self):
        """Empty tokens list and empty subset should not replace anything."""
        df = pd.DataFrame({"a": ["NA"], "b": ["NA"]})
        r1 = ar.standardize_missing_tokens(df, tokens=[])
        assert r1["a"][0] == "NA"
        r2 = ar.standardize_missing_tokens(df, subset=[])
        assert r2["a"][0] == "NA"

    def test_zero_rows_and_input_not_mutated(self):
        """Empty frame should work; input DataFrame should not be modified."""
        df_empty = pd.DataFrame({"v": pd.Series([], dtype=object)})
        result = ar.standardize_missing_tokens(df_empty)
        assert len(result) == 0 and "v" in result.columns
        
        df = pd.DataFrame({"v": [" NULL "]})
        ar.standardize_missing_tokens(df)
        assert df["v"][0] == " NULL "

    def test_error_bad_frame_type(self):
        """Non-DataFrame/ArFrame input should raise TypeError."""
        with pytest.raises(TypeError, match="ArFrame or a pandas DataFrame"):
            ar.standardize_missing_tokens([1, 2])

    def test_error_subset_string_or_non_sequence(self):
        """subset as string or non-sequence (set, None) should raise TypeError."""
        df = pd.DataFrame({"a": ["NA"]})
        with pytest.raises(TypeError, match=re.escape("Did you mean subset=['a']?")):
            ar.standardize_missing_tokens(df, subset="a")
        with pytest.raises(TypeError, match="subset must be a sequence"):
            ar.standardize_missing_tokens(df, subset={"a"})

    def test_error_tokens_string_or_non_list(self):
        """tokens as string or non-list type should raise TypeError."""
        df = pd.DataFrame({"v": ["NA"]})
        with pytest.raises(TypeError, match=re.escape("Did you mean tokens=['NA']?")):
            ar.standardize_missing_tokens(df, tokens="NA")
        with pytest.raises(TypeError, match="tokens must be a list"):
            ar.standardize_missing_tokens(df, tokens=("NA",))
        with pytest.raises(TypeError, match="tokens must be a list of strings"):
            ar.standardize_missing_tokens(df, tokens=["NA", 1])

    def test_error_missing_column_in_subset(self):
        """Non-existent column in subset should raise ValueError."""
        df = pd.DataFrame({"a": ["NA"]})
        with pytest.raises(ValueError, match="Unknown columns"):
            ar.standardize_missing_tokens(df, subset=["zz"])

    def test_error_subset_checked_before_tokens(self):
        """subset-is-string error should be raised before tokens validation."""
        df = pd.DataFrame({"a": ["NA"]})
        with pytest.raises(TypeError, match=re.escape("Did you mean subset=['a']?")):
            ar.standardize_missing_tokens(df, subset="a", tokens="BAD")
