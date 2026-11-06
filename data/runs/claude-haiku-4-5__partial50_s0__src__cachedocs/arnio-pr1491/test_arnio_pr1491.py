import pytest
import pandas as pd
import re
import arnio as ar


class TestDropColumnsMatching:
    """Test suite for arnio.drop_columns_matching function."""

    def test_basic_drop_matching_columns(self):
        """Test dropping columns matching a simple pattern."""
        df = pd.DataFrame({
            "temp_1": [1, 2],
            "temp_2": [3, 4],
            "data": [5, 6]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_columns_matching(frame, "^temp_")
        result_df = ar.to_pandas(result)
        assert list(result_df.columns) == ["data"]

    def test_returns_arframe_for_arframe_input(self):
        """Test that input ArFrame returns ArFrame."""
        df = pd.DataFrame({
            "temp_1": [1, 2],
            "data": [3, 4]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_columns_matching(frame, "^temp_")
        assert isinstance(result, ar.ArFrame)

    def test_returns_dataframe_for_pandas_input(self):
        """Test that input DataFrame returns DataFrame."""
        df = pd.DataFrame({
            "temp_1": [1, 2],
            "data": [3, 4]
        })
        result = ar.drop_columns_matching(df, "^temp_")
        assert isinstance(result, pd.DataFrame)

    def test_no_matching_columns(self):
        """Test when pattern matches no columns."""
        df = pd.DataFrame({
            "id": [1, 2],
            "name": ["a", "b"]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_columns_matching(frame, "^temp_")
        result_df = ar.to_pandas(result)
        assert list(result_df.columns) == ["id", "name"]
        assert result_df.shape == df.shape

    def test_multiple_matching_columns(self):
        """Test dropping multiple columns matching pattern."""
        df = pd.DataFrame({
            "temp_1": [1, 2],
            "temp_2": [3, 4],
            "temp_x": [5, 6],
            "data": [7, 8]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_columns_matching(frame, "^temp_")
        result_df = ar.to_pandas(result)
        assert list(result_df.columns) == ["data"]

    def test_invalid_regex_pattern_raises_error(self):
        """Test that invalid regex pattern raises re.error."""
        df = pd.DataFrame({"col": [1]})
        frame = ar.from_pandas(df)
        with pytest.raises(re.error):
            ar.drop_columns_matching(frame, "[invalid(")

    def test_pattern_not_string_raises_typeerror(self):
        """Test that non-string pattern raises TypeError."""
        df = pd.DataFrame({"col": [1]})
        frame = ar.from_pandas(df)
        with pytest.raises(TypeError):
            ar.drop_columns_matching(frame, 123)

    def test_pattern_as_list_raises_typeerror(self):
        """Test that list pattern raises TypeError."""
        df = pd.DataFrame({"col": [1]})
        frame = ar.from_pandas(df)
        with pytest.raises(TypeError):
            ar.drop_columns_matching(frame, ["temp_"])

    def test_pattern_matches_all_columns_raises_valueerror(self):
        """Test that ValueError is raised when pattern matches all columns."""
        df = pd.DataFrame({
            "col_1": [1, 2],
            "col_2": [3, 4]
        })
        frame = ar.from_pandas(df)
        with pytest.raises(ValueError, match="At least one column must remain"):
            ar.drop_columns_matching(frame, "^col_")

    def test_wildcard_pattern_matches_all_raises_valueerror(self):
        """Test ValueError when .* pattern matches all columns."""
        df = pd.DataFrame({
            "a": [1],
            "b": [2]
        })
        frame = ar.from_pandas(df)
        with pytest.raises(ValueError):
            ar.drop_columns_matching(frame, ".*")

    def test_pattern_with_special_regex_characters(self):
        """Test pattern with special regex characters (literal dot)."""
        df = pd.DataFrame({
            "col.1": [1],
            "col_1": [2],
            "col-1": [3],
            "data": [4]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_columns_matching(frame, r"col\.1")
        result_df = ar.to_pandas(result)
        assert "col.1" not in result_df.columns
        assert list(result_df.columns) == ["col_1", "col-1", "data"]

    def test_case_sensitive_matching(self):
        """Test that pattern matching is case-sensitive."""
        df = pd.DataFrame({
            "Temp_1": [1],
            "temp_1": [2],
            "TEMP_1": [3],
            "data": [4]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_columns_matching(frame, "^temp_")
        result_df = ar.to_pandas(result)
        assert "temp_1" not in result_df.columns
        assert "Temp_1" in result_df.columns
        assert "TEMP_1" in result_df.columns

    def test_pattern_with_end_anchor(self):
        """Test pattern with end-of-string anchor $."""
        df = pd.DataFrame({
            "id_temp": [1],
            "temp": [2],
            "temp_col": [3],
            "data": [4]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_columns_matching(frame, "temp$")
        result_df = ar.to_pandas(result)
        assert "id_temp" not in result_df.columns
        assert "temp" not in result_df.columns
        assert list(result_df.columns) == ["temp_col", "data"]

    def test_single_column_non_matching(self):
        """Test frame with single column that doesn't match pattern."""
        df = pd.DataFrame({"data": [1, 2]})
        frame = ar.from_pandas(df)
        result = ar.drop_columns_matching(frame, "^temp_")
        result_df = ar.to_pandas(result)
        assert list(result_df.columns) == ["data"]

    def test_pattern_matching_middle_of_name(self):
        """Test pattern matching middle of column name."""
        df = pd.DataFrame({
            "col_debug_1": [1],
            "col_debug_2": [2],
            "col_data": [3]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_columns_matching(frame, "debug")
        result_df = ar.to_pandas(result)
        assert list(result_df.columns) == ["col_data"]
