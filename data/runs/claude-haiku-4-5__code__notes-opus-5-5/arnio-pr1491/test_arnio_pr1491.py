import pytest
import pandas as pd
import re
import arnio as ar


class TestDropColumnsMatching:
    """Test arnio.cleaning.drop_columns_matching function."""

    def test_dataframe_partial_match_prefix(self):
        """Test dropping columns with prefix pattern."""
        df = pd.DataFrame({
            "temp_a": [1, 2],
            "b": ["x", "y"],
            "temp_c": [1.5, 2.5]
        })
        result = ar.cleaning.drop_columns_matching(df, "^temp_")
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["b"]

    def test_dataframe_no_match(self):
        """Test no-match returns all columns in new object."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        result = ar.cleaning.drop_columns_matching(df, "zzz")
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["a", "b", "c"]
        assert result is not df

    def test_dataframe_substring_match(self):
        """Test that re.search matches substrings."""
        df = pd.DataFrame({"my_temp": [1], "temp_x": [2], "keep": [3]})
        result = ar.cleaning.drop_columns_matching(df, "temp")
        assert list(result.columns) == ["keep"]

    def test_dataframe_case_sensitive(self):
        """Test that matching is case-sensitive."""
        df = pd.DataFrame({"TEMP_a": [1], "temp_b": [2], "c": [3]})
        result = ar.cleaning.drop_columns_matching(df, "^temp")
        assert list(result.columns) == ["TEMP_a", "c"]

    def test_dataframe_preserves_column_order(self):
        """Test that column order is preserved."""
        df = pd.DataFrame({"a": [1], "mid": [2], "z": [3]})
        result = ar.cleaning.drop_columns_matching(df, "^mid$")
        assert list(result.columns) == ["a", "z"]

    def test_dataframe_non_string_column_labels(self):
        """Test that non-string column labels are converted to string."""
        df = pd.DataFrame({0: [1], 1: [2], "x": [3]})
        result = ar.cleaning.drop_columns_matching(df, r"^\d$")
        assert list(result.columns) == ["x"]

    def test_dataframe_zero_columns_returns_input(self):
        """Test that zero-column frame returns input object without error."""
        df = pd.DataFrame(index=range(3))
        result = ar.cleaning.drop_columns_matching(df, "")
        assert result is df

    def test_dataframe_zero_rows(self):
        """Test that zero-row frame works normally."""
        df = pd.DataFrame({"temp_a": [], "b": [], "temp_c": []})
        result = ar.cleaning.drop_columns_matching(df, "^temp_")
        assert list(result.columns) == ["b"]
        assert len(result) == 0

    def test_dataframe_input_not_mutated(self):
        """Test that input DataFrame is not mutated."""
        df = pd.DataFrame({"temp_a": [1, 2], "b": [3, 4]})
        df_copy = df.copy()
        ar.cleaning.drop_columns_matching(df, "^temp_")
        pd.testing.assert_frame_equal(df, df_copy)

    def test_arframe_returns_arframe_type(self):
        """Test that ArFrame input returns ArFrame output."""
        df = pd.DataFrame({"temp_x": [1, 2], "keep": [3, 4]})
        frame = ar.from_pandas(df)
        result = ar.cleaning.drop_columns_matching(frame, "^temp")
        assert isinstance(result, ar.ArFrame)
        assert list(result.columns) == ["keep"]

    def test_error_invalid_frame_type(self):
        """Test TypeError when frame is not ArFrame or DataFrame."""
        with pytest.raises(TypeError, match="frame must be an ArFrame or a pandas DataFrame"):
            ar.cleaning.drop_columns_matching([1, 2], "x")

    def test_error_non_string_pattern(self):
        """Test TypeError when pattern is not a string."""
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(TypeError, match="pattern must be a string, got int"):
            ar.cleaning.drop_columns_matching(df, 5)

    def test_error_invalid_regex_pattern(self):
        """Test re.error when pattern is invalid regex."""
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(re.error, match=re.escape("Invalid regex pattern: '['")):
            ar.cleaning.drop_columns_matching(df, "[")

    def test_error_all_columns_match_empty(self):
        """Test ValueError when pattern matches all columns (empty pattern)."""
        df = pd.DataFrame({"a": [1], "b": [2]})
        with pytest.raises(ValueError, match=re.escape("Pattern matches all columns. At least one column must remain.")):
            ar.cleaning.drop_columns_matching(df, "")

    def test_error_all_columns_match_dot(self):
        """Test ValueError when pattern matches all columns (dot pattern)."""
        df = pd.DataFrame({"a": [1], "b": [2]})
        with pytest.raises(ValueError, match=re.escape("Pattern matches all columns. At least one column must remain.")):
            ar.cleaning.drop_columns_matching(df, ".")
