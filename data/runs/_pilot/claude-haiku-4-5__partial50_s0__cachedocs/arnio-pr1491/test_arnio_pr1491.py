import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestDropColumnsMatching:
    """Test suite for ar.drop_columns_matching()"""

    def test_basic_prefix_pattern(self):
        """Drop columns matching a simple prefix pattern"""
        df = pd.DataFrame({
            "temp_col1": [1, 2],
            "temp_col2": [3, 4],
            "keep_col": [5, 6],
        })
        frame = ar.from_pandas(df)
        result = ar.drop_columns_matching(frame, pattern="^temp_")
        
        result_df = ar.to_pandas(result)
        assert list(result_df.columns) == ["keep_col"]
        assert len(result_df) == 2

    def test_no_matches_returns_all_columns(self):
        """Pattern that matches no columns returns all columns"""
        df = pd.DataFrame({
            "col_a": [1, 2],
            "col_b": [3, 4],
        })
        frame = ar.from_pandas(df)
        result = ar.drop_columns_matching(frame, pattern="^temp_")
        
        result_df = ar.to_pandas(result)
        assert list(result_df.columns) == ["col_a", "col_b"]

    def test_match_all_columns(self):
        """Pattern matching all columns returns empty column set"""
        df = pd.DataFrame({
            "col_a": [1, 2],
            "col_b": [3, 4],
        })
        frame = ar.from_pandas(df)
        result = ar.drop_columns_matching(frame, pattern=".*")
        
        result_df = ar.to_pandas(result)
        assert len(result_df.columns) == 0
        assert len(result_df) == 2

    def test_suffix_pattern(self):
        """Drop columns matching a suffix pattern"""
        df = pd.DataFrame({
            "col_temp": [1, 2],
            "data_temp": [3, 4],
            "keep_col": [5, 6],
        })
        frame = ar.from_pandas(df)
        result = ar.drop_columns_matching(frame, pattern="_temp$")
        
        result_df = ar.to_pandas(result)
        assert list(result_df.columns) == ["keep_col"]

    def test_case_sensitive_matching(self):
        """Regex matching is case-sensitive by default"""
        df = pd.DataFrame({
            "Temp_col": [1, 2],
            "temp_col": [3, 4],
            "TEMP_COL": [5, 6],
            "keep": [7, 8],
        })
        frame = ar.from_pandas(df)
        result = ar.drop_columns_matching(frame, pattern="^temp_")
        
        result_df = ar.to_pandas(result)
        assert list(result_df.columns) == ["Temp_col", "TEMP_COL", "keep"]

    def test_alternation_pattern(self):
        """Pattern with alternation (OR) logic"""
        df = pd.DataFrame({
            "temp_col": [1, 2],
            "debug_col": [3, 4],
            "keep_col": [5, 6],
        })
        frame = ar.from_pandas(df)
        result = ar.drop_columns_matching(frame, pattern="^(temp_|debug_)")
        
        result_df = ar.to_pandas(result)
        assert list(result_df.columns) == ["keep_col"]

    def test_character_class_pattern(self):
        """Pattern with character classes"""
        df = pd.DataFrame({
            "col1": [1, 2],
            "col2": [3, 4],
            "col_a": [5, 6],
            "keep": [7, 8],
        })
        frame = ar.from_pandas(df)
        result = ar.drop_columns_matching(frame, pattern="col[0-9]")
        
        result_df = ar.to_pandas(result)
        assert list(result_df.columns) == ["col_a", "keep"]

    def test_preserves_column_order(self):
        """Remaining columns preserve their original order"""
        df = pd.DataFrame({
            "a": [1],
            "temp_b": [2],
            "c": [3],
            "temp_d": [4],
            "e": [5],
        })
        frame = ar.from_pandas(df)
        result = ar.drop_columns_matching(frame, pattern="^temp_")
        
        result_df = ar.to_pandas(result)
        assert list(result_df.columns) == ["a", "c", "e"]

    def test_returns_new_frame_does_not_mutate(self):
        """Function returns new ArFrame, original unchanged"""
        df = pd.DataFrame({
            "temp_col": [1, 2],
            "keep_col": [3, 4],
        })
        frame = ar.from_pandas(df)
        result = ar.drop_columns_matching(frame, pattern="^temp_")
        
        # Original frame is unchanged
        assert frame.columns == ["temp_col", "keep_col"]
        # Result is a new frame
        assert result.columns == ["keep_col"]

    def test_empty_dataframe_no_rows(self):
        """Handle frame with no rows but matching columns"""
        df = pd.DataFrame({
            "temp_col": [],
            "keep_col": [],
        })
        frame = ar.from_pandas(df)
        result = ar.drop_columns_matching(frame, pattern="^temp_")
        
        result_df = ar.to_pandas(result)
        assert list(result_df.columns) == ["keep_col"]
        assert len(result_df) == 0

    def test_single_matching_column(self):
        """Frame with single column that matches pattern"""
        df = pd.DataFrame({
            "temp_col": [1, 2, 3],
        })
        frame = ar.from_pandas(df)
        result = ar.drop_columns_matching(frame, pattern="^temp_")
        
        result_df = ar.to_pandas(result)
        assert len(result_df.columns) == 0
        assert len(result_df) == 3

    def test_wildcard_middle_pattern(self):
        """Pattern with wildcards in the middle"""
        df = pd.DataFrame({
            "col_x_temp": [1],
            "col_y_temp": [2],
            "col_permanent": [3],
            "keep": [4],
        })
        frame = ar.from_pandas(df)
        result = ar.drop_columns_matching(frame, pattern="col_.*_temp$")
        
        result_df = ar.to_pandas(result)
        assert list(result_df.columns) == ["col_permanent", "keep"]

    def test_whitespace_in_column_names(self):
        """Pattern matching with whitespace in column names"""
        df = pd.DataFrame({
            "temp col": [1],
            "keep col": [2],
            "temporary": [3],
        })
        frame = ar.from_pandas(df)
        result = ar.drop_columns_matching(frame, pattern="^temp ")
        
        result_df = ar.to_pandas(result)
        assert list(result_df.columns) == ["keep col", "temporary"]

    def test_numeric_column_names(self):
        """Pattern matching numeric column names"""
        df = pd.DataFrame({
            "123": [1],
            "456": [2],
            "col789": [3],
        })
        frame = ar.from_pandas(df)
        result = ar.drop_columns_matching(frame, pattern="^[0-9]+$")
        
        result_df = ar.to_pandas(result)
        assert list(result_df.columns) == ["col789"]

    def test_data_integrity_preserved(self):
        """Row data is preserved correctly after column drop"""
        df = pd.DataFrame({
            "temp_x": [10, 20, 30],
            "keep_y": [100, 200, 300],
            "temp_z": [1000, 2000, 3000],
        })
        frame = ar.from_pandas(df)
        result = ar.drop_columns_matching(frame, pattern="^temp_")
        
        result_df = ar.to_pandas(result)
        assert result_df["keep_y"].tolist() == [100, 200, 300]
        assert len(result_df) == 3
