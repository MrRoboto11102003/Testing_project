import pytest
import pandas as pd
import re
import arnio as ar


class TestDropColumnsMatching:
    """Test suite for arnio.cleaning.drop_columns_matching"""
    
    @pytest.fixture
    def sample_df(self):
        """Sample DataFrame with mixed column names"""
        return pd.DataFrame({
            'temp_col1': [1, 2, 3],
            'temp_col2': [4, 5, 6],
            'keep_col1': [7, 8, 9],
            'keep_col2': [10, 11, 12],
            'temp_data': [13, 14, 15],
        })
    
    @pytest.fixture
    def sample_arframe(self, sample_df):
        """Sample ArFrame for testing"""
        return ar.from_pandas(sample_df)
    
    def test_drop_columns_prefix_pattern_pandas(self, sample_df):
        """Test dropping columns with prefix pattern on DataFrame"""
        result = ar.drop_columns_matching(sample_df, "^temp_")
        assert list(result.columns) == ['keep_col1', 'keep_col2']
    
    def test_drop_columns_prefix_pattern_arframe(self, sample_arframe):
        """Test dropping columns with prefix pattern on ArFrame"""
        result = ar.drop_columns_matching(sample_arframe, "^temp_")
        assert result.columns == ['keep_col1', 'keep_col2']
    
    def test_return_type_preserved_pandas(self, sample_df):
        """Verify DataFrame input returns DataFrame"""
        result = ar.drop_columns_matching(sample_df, "^temp_")
        assert isinstance(result, pd.DataFrame)
    
    def test_return_type_preserved_arframe(self, sample_arframe):
        """Verify ArFrame input returns ArFrame"""
        result = ar.drop_columns_matching(sample_arframe, "^temp_")
        assert isinstance(result, ar.ArFrame)
    
    def test_no_columns_match_pattern(self, sample_df):
        """Test when pattern matches no columns"""
        result = ar.drop_columns_matching(sample_df, "^nomatch")
        pd.testing.assert_frame_equal(result, sample_df)
    
    def test_drop_columns_suffix_pattern(self, sample_df):
        """Test dropping columns with suffix pattern"""
        result = ar.drop_columns_matching(sample_df, "_col1$")
        assert list(result.columns) == ['temp_col2', 'keep_col2', 'temp_data']
    
    def test_drop_columns_contains_pattern(self, sample_df):
        """Test dropping columns containing substring"""
        result = ar.drop_columns_matching(sample_df, "keep")
        assert list(result.columns) == ['temp_col1', 'temp_col2', 'temp_data']
    
    def test_invalid_regex_raises_error(self, sample_df):
        """Test that invalid regex pattern raises re.error"""
        with pytest.raises(re.error):
            ar.drop_columns_matching(sample_df, "[invalid(")
    
    def test_non_string_pattern_raises_typeerror(self, sample_df):
        """Test that non-string pattern raises TypeError"""
        with pytest.raises(TypeError):
            ar.drop_columns_matching(sample_df, 123)
    
    def test_pattern_matches_all_columns_raises_error(self, sample_df):
        """Test that pattern matching all columns raises ValueError"""
        with pytest.raises(ValueError):
            ar.drop_columns_matching(sample_df, ".*")
    
    def test_single_column_all_match_raises_error(self):
        """Test ValueError when all columns match"""
        df = pd.DataFrame({'col1': [1, 2, 3]})
        with pytest.raises(ValueError):
            ar.drop_columns_matching(df, "col")
    
    def test_case_sensitive_pattern_matching(self):
        """Test that pattern matching is case sensitive"""
        df = pd.DataFrame({
            'TempCol': [1, 2, 3],
            'tempCol': [4, 5, 6],
            'KeepCol': [7, 8, 9],
        })
        result = ar.drop_columns_matching(df, "^temp")
        assert list(result.columns) == ['TempCol', 'KeepCol']
    
    def test_character_class_pattern(self):
        """Test pattern with character class"""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6],
            'col3': [7, 8, 9],
            'data': [10, 11, 12],
        })
        result = ar.drop_columns_matching(df, "col[12]")
        assert list(result.columns) == ['col3', 'data']
    
    def test_alternation_pattern(self):
        """Test pattern with alternation (|)"""
        df = pd.DataFrame({
            'temp_col': [1, 2, 3],
            'cache_col': [4, 5, 6],
            'keep_col': [7, 8, 9],
        })
        result = ar.drop_columns_matching(df, "(temp|cache)")
        assert list(result.columns) == ['keep_col']
    
    def test_data_integrity_preserved(self, sample_df):
        """Test that data values are preserved in kept columns"""
        result = ar.drop_columns_matching(sample_df, "^temp")
        expected = sample_df[['keep_col1', 'keep_col2']]
        pd.testing.assert_frame_equal(result, expected)
