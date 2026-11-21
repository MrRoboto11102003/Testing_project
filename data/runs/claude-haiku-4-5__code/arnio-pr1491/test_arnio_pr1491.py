import pytest
import pandas as pd
import re
import arnio as ar


class TestDropColumnsMatching:
    """Pytest tests for arnio.cleaning.drop_columns_matching function."""
    
    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame with various column names."""
        return pd.DataFrame({
            'temp_col1': [1, 2, 3],
            'temp_col2': [4, 5, 6],
            'data_col1': [7, 8, 9],
            'data_col2': [10, 11, 12],
            'info': [13, 14, 15]
        })
    
    def test_basic_prefix_pattern_match(self, sample_df):
        """Test dropping columns that match a prefix pattern."""
        result = ar.cleaning.drop_columns_matching(sample_df, "^temp_")
        assert list(result.columns) == ['data_col1', 'data_col2', 'info']
        assert len(result) == 3
    
    def test_no_columns_match_pattern(self, sample_df):
        """Test that all columns are retained when pattern matches none."""
        result = ar.cleaning.drop_columns_matching(sample_df, "^nomatch_")
        pd.testing.assert_frame_equal(result, sample_df)
    
    def test_pattern_matches_all_columns_raises_error(self, sample_df):
        """Test ValueError is raised when pattern matches all columns."""
        with pytest.raises(ValueError, match="Pattern matches all columns"):
            ar.cleaning.drop_columns_matching(sample_df, ".*")
    
    def test_pattern_not_string_raises_type_error(self, sample_df):
        """Test TypeError is raised when pattern is not a string."""
        with pytest.raises(TypeError, match="pattern must be a string"):
            ar.cleaning.drop_columns_matching(sample_df, 123)
    
    def test_pattern_none_raises_type_error(self, sample_df):
        """Test TypeError is raised when pattern is None."""
        with pytest.raises(TypeError, match="pattern must be a string"):
            ar.cleaning.drop_columns_matching(sample_df, None)
    
    def test_invalid_regex_pattern_raises_error(self, sample_df):
        """Test re.error is raised for invalid regex pattern."""
        with pytest.raises(re.error):
            ar.cleaning.drop_columns_matching(sample_df, "[invalid(")
    
    def test_suffix_anchor_pattern(self, sample_df):
        """Test pattern matching with suffix anchor."""
        result = ar.cleaning.drop_columns_matching(sample_df, "1$")
        assert list(result.columns) == ['temp_col2', 'data_col2', 'info']
    
    def test_substring_pattern(self, sample_df):
        """Test pattern matching substring in column names."""
        result = ar.cleaning.drop_columns_matching(sample_df, "_col")
        assert list(result.columns) == ['info']
    
    def test_exact_column_match(self, sample_df):
        """Test exact column name match using anchors."""
        result = ar.cleaning.drop_columns_matching(sample_df, "^info$")
        assert list(result.columns) == ['temp_col1', 'temp_col2', 'data_col1', 'data_col2']
    
    def test_data_integrity_preserved(self, sample_df):
        """Test that data values in remaining columns are unchanged."""
        result = ar.cleaning.drop_columns_matching(sample_df, "^temp_")
        expected = sample_df[['data_col1', 'data_col2', 'info']]
        pd.testing.assert_frame_equal(result, expected)
    
    def test_case_sensitive_matching(self, sample_df):
        """Test that pattern matching is case-sensitive."""
        result = ar.cleaning.drop_columns_matching(sample_df, "^TEMP_")
        pd.testing.assert_frame_equal(result, sample_df)
    
    def test_alternation_pattern_or(self, sample_df):
        """Test regex alternation pattern (OR)."""
        result = ar.cleaning.drop_columns_matching(sample_df, "^temp_|^info$")
        assert list(result.columns) == ['data_col1', 'data_col2']
    
    def test_numeric_column_names(self):
        """Test with numeric column names."""
        df = pd.DataFrame({0: [1, 2], 1: [3, 4], 2: [5, 6]})
        result = ar.cleaning.drop_columns_matching(df, "^[01]$")
        assert list(result.columns) == [2]
    
    def test_special_characters_escaping(self):
        """Test pattern matching with special regex characters in names."""
        df = pd.DataFrame({
            'col-1': [1, 2],
            'col_2': [3, 4],
            'col.3': [5, 6]
        })
        result = ar.cleaning.drop_columns_matching(df, r"col[\-_]")
        assert list(result.columns) == ['col.3']
    
    def test_return_type_is_pandas_dataframe(self, sample_df):
        """Test that return type is pandas DataFrame."""
        result = ar.cleaning.drop_columns_matching(sample_df, "^temp_")
        assert isinstance(result, pd.DataFrame)
