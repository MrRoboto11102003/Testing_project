import pytest
import pandas as pd
import arnio as ar


class TestDropColumnsMatching:
    """Test suite for arnio.cleaning.drop_columns_matching"""

    def test_drop_columns_matching_basic_prefix(self):
        """Test dropping columns matching a prefix pattern."""
        df = pd.DataFrame({
            "temp_col1": [1, 2, 3],
            "temp_col2": [4, 5, 6],
            "keep_col": [7, 8, 9]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_columns_matching(frame, pattern="^temp_")
        result_df = ar.to_pandas(result)
        
        assert list(result_df.columns) == ["keep_col"]
        assert len(result_df) == 3

    def test_drop_columns_matching_suffix_pattern(self):
        """Test dropping columns matching a suffix pattern."""
        df = pd.DataFrame({
            "col_debug": [1, 2, 3],
            "data_debug": [4, 5, 6],
            "keep_col": [7, 8, 9]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_columns_matching(frame, pattern="_debug$")
        result_df = ar.to_pandas(result)
        
        assert list(result_df.columns) == ["keep_col"]

    def test_drop_columns_matching_substring(self):
        """Test dropping columns containing a substring."""
        df = pd.DataFrame({
            "debug_1": [1, 2, 3],
            "col_debug_2": [4, 5, 6],
            "keep": [7, 8, 9]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_columns_matching(frame, pattern="debug")
        result_df = ar.to_pandas(result)
        
        assert list(result_df.columns) == ["keep"]

    def test_drop_columns_matching_no_matches_raises_error(self):
        """Test that no matching columns raises ValueError."""
        df = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": [4, 5, 6]
        })
        frame = ar.from_pandas(df)
        
        with pytest.raises(ValueError):
            ar.drop_columns_matching(frame, pattern="^nomatch_")

    def test_drop_columns_matching_all_columns(self):
        """Test pattern matching all columns."""
        df = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": [4, 5, 6],
            "col3": [7, 8, 9]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_columns_matching(frame, pattern="col")
        result_df = ar.to_pandas(result)
        
        assert len(result_df.columns) == 0

    def test_drop_columns_matching_preserves_order(self):
        """Test that remaining columns preserve original order."""
        df = pd.DataFrame({
            "a_x": [1, 2, 3],
            "keep_b": [4, 5, 6],
            "c_x": [7, 8, 9],
            "keep_d": [10, 11, 12]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_columns_matching(frame, pattern="_x$")
        result_df = ar.to_pandas(result)
        
        assert list(result_df.columns) == ["keep_b", "keep_d"]

    def test_drop_columns_matching_case_sensitive(self):
        """Test that matching is case-sensitive."""
        df = pd.DataFrame({
            "Temp_col": [1, 2, 3],
            "temp_col": [4, 5, 6],
            "keep_col": [7, 8, 9]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_columns_matching(frame, pattern="^temp_")
        result_df = ar.to_pandas(result)
        
        assert "Temp_col" in result_df.columns
        assert "temp_col" not in result_df.columns
        assert list(result_df.columns) == ["Temp_col", "keep_col"]

    def test_drop_columns_matching_regex_special_chars(self):
        """Test pattern with escaped regex special characters."""
        df = pd.DataFrame({
            "col.1": [1, 2, 3],
            "col.2": [4, 5, 6],
            "col_3": [7, 8, 9]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_columns_matching(frame, pattern=r"col\.\d")
        result_df = ar.to_pandas(result)
        
        assert list(result_df.columns) == ["col_3"]

    def test_drop_columns_matching_invalid_regex_raises(self):
        """Test that invalid regex pattern raises an error."""
        df = pd.DataFrame({
            "col1": [1, 2, 3]
        })
        frame = ar.from_pandas(df)
        
        with pytest.raises(Exception):
            ar.drop_columns_matching(frame, pattern="[invalid(regex")

    def test_drop_columns_matching_does_not_mutate_original(self):
        """Test that original frame is not modified."""
        df = pd.DataFrame({
            "temp_col": [1, 2, 3],
            "keep_col": [4, 5, 6]
        })
        frame = ar.from_pandas(df)
        original_columns = frame.columns.copy()
        
        ar.drop_columns_matching(frame, pattern="^temp_")
        
        assert frame.columns == original_columns
        assert "temp_col" in frame.columns

    def test_drop_columns_matching_with_mixed_dtypes(self):
        """Test matching works regardless of column data types."""
        df = pd.DataFrame({
            "temp_int": [1, 2, 3],
            "temp_str": ["a", "b", "c"],
            "temp_float": [1.5, 2.5, 3.5],
            "keep_col": [True, False, True]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_columns_matching(frame, pattern="^temp_")
        result_df = ar.to_pandas(result)
        
        assert list(result_df.columns) == ["keep_col"]

    def test_drop_columns_matching_empty_frame(self):
        """Test with frame containing no rows."""
        df = pd.DataFrame({
            "col1": pd.Series([], dtype='int64'),
            "col2": pd.Series([], dtype='int64')
        })
        frame = ar.from_pandas(df)
        result = ar.drop_columns_matching(frame, pattern="col1")
        result_df = ar.to_pandas(result)
        
        assert list(result_df.columns) == ["col2"]
        assert len(result_df) == 0

    def test_drop_columns_matching_complex_pattern(self):
        """Test with complex regex pattern."""
        df = pd.DataFrame({
            "id_123": [1, 2, 3],
            "id_456": [4, 5, 6],
            "value_789": [7, 8, 9],
            "name": [10, 11, 12]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_columns_matching(frame, pattern="^id_\\d+$")
        result_df = ar.to_pandas(result)
        
        assert list(result_df.columns) == ["value_789", "name"]

    def test_drop_columns_matching_returns_arframe(self):
        """Test that result is an ArFrame, not a pandas DataFrame."""
        df = pd.DataFrame({
            "temp_col": [1, 2, 3],
            "keep_col": [4, 5, 6]
        })
        frame = ar.from_pandas(df)
        result = ar.drop_columns_matching(frame, pattern="^temp_")
        
        assert isinstance(result, ar.ArFrame)
