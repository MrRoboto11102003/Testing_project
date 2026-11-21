import pytest
import pandas as pd
import tempfile
import subprocess
import sys
from pathlib import Path
import re

import arnio as ar
from arnio import cleaning


class TestDropColumnsMatching:
    """System tests for arnio.cleaning.drop_columns_matching"""

    def test_basic_drop_matching_columns_pandas(self, tmp_path):
        """Test basic workflow: load CSV, drop matching columns with pandas DataFrame"""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({
            'temp_col1': [1, 2, 3],
            'temp_col2': [4, 5, 6],
            'keep_col': [7, 8, 9]
        })
        df.to_csv(csv_file, index=False)
        
        loaded_df = pd.read_csv(csv_file)
        result = cleaning.drop_columns_matching(loaded_df, "^temp_")
        
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ['keep_col']
        assert len(result) == 3
        assert list(result['keep_col']) == [7, 8, 9]

    def test_drop_columns_with_arframe(self, tmp_path):
        """Test dropping columns from ArFrame workflow"""
        csv_file = tmp_path / "metrics.csv"
        df = pd.DataFrame({
            'metric_a': [1, 2, 3],
            'metric_b': [4, 5, 6],
            'id': [10, 20, 30]
        })
        df.to_csv(csv_file, index=False)
        
        frame = ar.read_csv(str(csv_file))
        result = cleaning.drop_columns_matching(frame, "^metric_")
        
        result_df = pd.DataFrame(result)
        assert 'id' in result_df.columns
        assert 'metric_a' not in result_df.columns
        assert 'metric_b' not in result_df.columns

    def test_end_to_end_load_clean_write(self, tmp_path):
        """Test complete end-to-end workflow: load, clean, write to output"""
        input_file = tmp_path / "input.csv"
        output_file = tmp_path / "cleaned.csv"
        
        df = pd.DataFrame({
            'temp_1': [1, 2],
            'temp_2': [3, 4],
            'keep': [5, 6]
        })
        df.to_csv(input_file, index=False)
        
        loaded = pd.read_csv(input_file)
        cleaned = cleaning.drop_columns_matching(loaded, "^temp_")
        cleaned.to_csv(output_file, index=False)
        
        assert output_file.exists()
        output_df = pd.read_csv(output_file)
        assert list(output_df.columns) == ['keep']
        assert len(output_df) == 2

    def test_invalid_pattern_type_raises_typeerror(self, tmp_path):
        """Test that non-string pattern raises TypeError"""
        df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        
        with pytest.raises(TypeError, match="pattern must be a string"):
            cleaning.drop_columns_matching(df, 123)
        
        with pytest.raises(TypeError, match="pattern must be a string"):
            cleaning.drop_columns_matching(df, ['col1'])
        
        with pytest.raises(TypeError, match="pattern must be a string"):
            cleaning.drop_columns_matching(df, None)

    def test_invalid_regex_pattern_raises_error(self, tmp_path):
        """Test that syntactically invalid regex raises re.error"""
        df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        
        with pytest.raises(re.error):
            cleaning.drop_columns_matching(df, "[unclosed")
        
        with pytest.raises(re.error):
            cleaning.drop_columns_matching(df, "(?P<invalid)")

    def test_pattern_matching_all_columns_raises_valueerror(self, tmp_path):
        """Test that pattern matching all columns raises ValueError"""
        df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        
        with pytest.raises(ValueError, match="At least one column must remain"):
            cleaning.drop_columns_matching(df, ".*")

    def test_pattern_matches_nothing(self, tmp_path):
        """Test that pattern matching no columns returns unchanged frame"""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })
        
        result = cleaning.drop_columns_matching(df, "^nomatch")
        
        assert list(result.columns) == ['col1', 'col2']
        pd.testing.assert_frame_equal(result, df)

    def test_multiple_columns_match_pattern(self, tmp_path):
        """Test dropping multiple columns that match pattern"""
        df = pd.DataFrame({
            'temp_a': [1, 2],
            'temp_b': [3, 4],
            'temp_c': [5, 6],
            'keep_x': [7, 8],
            'keep_y': [9, 10]
        })
        
        result = cleaning.drop_columns_matching(df, "^temp_")
        
        assert list(result.columns) == ['keep_x', 'keep_y']
        assert len(result) == 2

    def test_case_sensitive_matching(self, tmp_path):
        """Test that pattern matching is case-sensitive"""
        df = pd.DataFrame({
            'Temp_Col': [1, 2],
            'temp_col': [3, 4],
            'Other': [5, 6]
        })
        
        result = cleaning.drop_columns_matching(df, "^temp_")
        
        assert 'Temp_Col' in result.columns
        assert 'temp_col' not in result.columns
        assert 'Other' in result.columns

    def test_regex_anchors_start_and_end(self, tmp_path):
        """Test regex anchors ^ and $ work correctly"""
        df = pd.DataFrame({
            'temp_start': [1, 2],
            'middle_temp': [3, 4],
            'end_temp': [5, 6],
            'keep': [7, 8]
        })
        
        result = cleaning.drop_columns_matching(df, "_temp$")
        assert list(result.columns) == ['temp_start', 'keep']
        
        result2 = cleaning.drop_columns_matching(df, "^temp")
        assert list(result2.columns) == ['middle_temp', 'end_temp', 'keep']

    def test_special_regex_characters_in_names(self, tmp_path):
        """Test patterns with special regex characters for column names"""
        df = pd.DataFrame({
            'col-1': [1, 2],
            'col.2': [3, 4],
            'col_3': [5, 6],
            'other': [7, 8]
        })
        
        result = cleaning.drop_columns_matching(df, r"col[\.-]")
        
        assert 'col-1' not in result.columns
        assert 'col.2' not in result.columns
        assert 'col_3' in result.columns
        assert 'other' in result.columns

    def test_data_values_preserved_after_drop(self, tmp_path):
        """Test that remaining data values are unchanged after column drop"""
        df = pd.DataFrame({
            'keep_id': [1, 2, 3],
            'keep_value': [10.5, 20.5, 30.5],
            'drop_col': [100, 200, 300]
        })
        
        result = cleaning.drop_columns_matching(df, "^drop_")
        
        assert result['keep_id'].tolist() == [1, 2, 3]
        assert result['keep_value'].tolist() == [10.5, 20.5, 30.5]

    def test_single_column_no_match(self, tmp_path):
        """Test single-column dataframe where pattern doesn't match"""
        df = pd.DataFrame({'important': [1, 2, 3]})
        
        result = cleaning.drop_columns_matching(df, "^temp_")
        
        assert list(result.columns) == ['important']
        assert len(result) == 3

    def test_numeric_string_column_names(self, tmp_path):
        """Test matching numeric string column names"""
        df = pd.DataFrame([[1, 2, 3]], columns=['0', '1', '2'])
        
        result = cleaning.drop_columns_matching(df, "[01]")
        
        assert list(result.columns) == ['2']
        assert result.iloc[0, 0] == 3

    def test_partial_substring_matching(self, tmp_path):
        """Test matching based on substring pattern"""
        df = pd.DataFrame({
            'my_temp_col': [1, 2],
            'another_temp_data': [3, 4],
            'important_field': [5, 6]
        })
        
        result = cleaning.drop_columns_matching(df, "temp")
        
        assert list(result.columns) == ['important_field']
        assert len(result) == 2
