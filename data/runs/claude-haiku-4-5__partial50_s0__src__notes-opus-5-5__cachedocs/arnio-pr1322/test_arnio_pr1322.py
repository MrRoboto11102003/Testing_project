import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestCoalesceColumns:
    """Tests for ar.coalesce_columns()"""

    def test_basic_first_non_null(self):
        """First non-null value is selected in subset order."""
        df = pd.DataFrame({
            "a": [1.0, np.nan, np.nan],
            "b": [10.0, 2.0, np.nan],
        })
        result = ar.coalesce_columns(df, subset=["a", "b"], output_column="first")
        assert result["first"].tolist()[0] == 1.0
        assert result["first"].tolist()[1] == 2.0
        assert pd.isna(result["first"].tolist()[2])

    def test_subset_order_priority(self):
        """Subset order determines priority, not frame column order."""
        df = pd.DataFrame({"a": [1.0], "b": [10.0]})
        result_ab = ar.coalesce_columns(df, subset=["a", "b"], output_column="out")
        result_ba = ar.coalesce_columns(df, subset=["b", "a"], output_column="out")
        assert result_ab["out"].iloc[0] == 1.0
        assert result_ba["out"].iloc[0] == 10.0

    def test_output_column_added_at_end_input_unchanged(self):
        """Output column added at end, input dataframe not mutated."""
        df = pd.DataFrame({"a": [1], "b": [2]})
        result = ar.coalesce_columns(df, subset=["a"], output_column="out")
        assert list(result.columns) == ["a", "b", "out"]
        assert list(df.columns) == ["a", "b"]

    def test_error_validation(self):
        """Test error cases: empty output_column, bad subset."""
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="output_column must be a non-empty string"):
            ar.coalesce_columns(df, subset=["a"], output_column="")
        with pytest.raises(TypeError, match="sequence of column names, not a string"):
            ar.coalesce_columns(df, subset="a")
        with pytest.raises(ValueError, match="duplicate column names"):
            ar.coalesce_columns(df, subset=["a", "a"])

    def test_error_missing_columns_output_exists(self):
        """Test missing columns and existing output column errors."""
        df = pd.DataFrame({"a": [1], "b": [2]})
        with pytest.raises(KeyError):
            ar.coalesce_columns(df, subset=["a", "zz"])
        with pytest.raises(ValueError, match="Output column 'b' already exists"):
            ar.coalesce_columns(df, subset=["a"], output_column="b")

    def test_arframe_support(self):
        """ArFrame input returns ArFrame output."""
        df = pd.DataFrame({"a": [1.0, np.nan], "b": [2.0, 3.0]})
        frame = ar.from_pandas(df)
        result = ar.coalesce_columns(frame, subset=["a", "b"], output_column="out")
        assert isinstance(result, ar.ArFrame)


class TestStandardizeMissingTokens:
    """Tests for ar.standardize_missing_tokens()"""

    def test_default_tokens_replaced_case_insensitive(self):
        """Default tokens replaced with NaN, case-insensitive."""
        df = pd.DataFrame({"v": [" NULL ", "\tNaN\t", "NA", "kept"]})
        result = ar.standardize_missing_tokens(df)
        assert pd.isna(result["v"].iloc[0])
        assert pd.isna(result["v"].iloc[1])
        assert pd.isna(result["v"].iloc[2])
        assert result["v"].iloc[3] == "kept"

    def test_whitespace_handling(self):
        """Whitespace trimmed for matching but preserved for non-matching."""
        df = pd.DataFrame({"v": [" kept ", "\tmissing\t"]})
        result = ar.standardize_missing_tokens(df)
        assert result["v"].iloc[0] == " kept "
        assert result["v"].iloc[1] == "\tmissing\t"

    def test_custom_tokens_subset(self):
        """Custom tokens replace defaults, subset limits columns."""
        df = pd.DataFrame({"a": ["NA", "MISSING", "x"], "b": ["NA", "y"]})
        result = ar.standardize_missing_tokens(df, tokens=["MISSING"], subset=["a"])
        assert result["a"].iloc[0] == "NA"
        assert pd.isna(result["a"].iloc[1])
        assert result["b"].iloc[0] == "NA"

    def test_non_string_and_non_matching_preserved(self):
        """Non-string values and non-matching strings preserved."""
        df = pd.DataFrame({"v": [1, 2.0, None, True, "x"]}, dtype=object)
        result = ar.standardize_missing_tokens(df)
        assert result["v"].iloc[0] == 1
        assert result["v"].iloc[3] is True
        assert result["v"].iloc[4] == "x"

    def test_input_not_mutated(self):
        """Input dataframe not mutated."""
        df = pd.DataFrame({"v": [" NULL "]})
        ar.standardize_missing_tokens(df)
        assert df["v"].iloc[0] == " NULL "

    def test_error_validation(self):
        """Test error cases for subset and tokens parameters."""
        df = pd.DataFrame({"a": ["x"]})
        with pytest.raises(TypeError, match="subset must be a list of column names, not a string"):
            ar.standardize_missing_tokens(df, subset="a")
        with pytest.raises(TypeError, match="tokens must be a list of strings, not a bare string"):
            ar.standardize_missing_tokens(df, tokens="NA")
        with pytest.raises(TypeError, match="got 'tuple'"):
            ar.standardize_missing_tokens(df, tokens=("NA",))

    def test_error_missing_columns_non_string_tokens(self):
        """Test missing columns and non-string token item errors."""
        df = pd.DataFrame({"a": ["NA"]})
        with pytest.raises(ValueError, match="Unknown columns in subset"):
            ar.standardize_missing_tokens(df, subset=["zz"])
        with pytest.raises(TypeError, match="got 'int' item"):
            ar.standardize_missing_tokens(df, tokens=["NA", 1])

    def test_arframe_support(self):
        """ArFrame input returns ArFrame output."""
        df = pd.DataFrame({"v": ["NA", "x"]})
        frame = ar.from_pandas(df)
        result = ar.standardize_missing_tokens(frame)
        assert isinstance(result, ar.ArFrame)
