import pytest
import pandas as pd
import re
import arnio as ar


class TestDropColumnsMatching:
    """Test suite for arnio.cleaning.drop_columns_matching"""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample pandas DataFrame for testing"""
        return pd.DataFrame({
            'temp_col1': [1, 2, 3],
            'temp_col2': [4, 5, 6],
            'data_col': [7, 8, 9],
            'temporary': [10, 11, 12],
            'value': [13, 14, 15]
        })
    
    @pytest.fixture
    def sample_arframe(self):
        """Create a sample ArFrame for testing"""
        return ar.from_dict({
            'temp_col1': [1, 2, 3],
            'temp_col2': [4, 5, 6],
            'data_col': [7, 8, 9],
            'temporary': [10, 11, 12],
            'value': [13, 14, 15]
        })
    
    def test_drop_columns_matching_pandas_dataframe(self, sample_dataframe):
        """Test basic column dropping with pandas DataFrame"""
        result = ar.cleaning.drop_columns_matching(sample_dataframe, "^temp_")
        assert isinstance(result, pd.DataFrame)
        assert 'temp_col1' not in result.columns
        assert 'temp_col2' not in result.columns
        assert 'data_col' in result.columns
        assert len(result) == 3
    
    def test_drop_columns_matching_arframe_return_type(self, sample_arframe):
        """Test that ArFrame input returns ArFrame output"""
        result = ar.cleaning.drop_columns_matching(sample_arframe, "^temp_")
        assert isinstance(result, ar.ArFrame)
        assert 'temp_col1' not in result.columns
    
    def test_drop_columns_matching_multiple_matches(self, sample_dataframe):
        """Test pattern matching multiple columns"""
        result = ar.cleaning.drop_columns_matching(sample_dataframe, "temp")
        assert 'temp_col1' not in result.columns
        assert 'temp_col2' not in result.columns
        assert 'temporary' not in result.columns
        assert result.shape[1] == 2
    
    def test_drop_columns_matching_no_matches(self, sample_dataframe):
        """Test pattern that matches no columns returns data unchanged"""
        result = ar.cleaning.drop_columns_matching(sample_dataframe, "^nonexistent_")
        assert result.shape == sample_dataframe.shape
        pd.testing.assert_frame_equal(result, sample_dataframe)
    
    def test_drop_columns_matching_single_column(self, sample_dataframe):
        """Test pattern matching a single column"""
        result = ar.cleaning.drop_columns_matching(sample_dataframe, "^value$")
        assert 'value' not in result.columns
        assert result.shape[1] == 4
    
    def test_drop_columns_matching_regex_anchors(self, sample_dataframe):
        """Test pattern with regex anchors distinguishes prefix from substring"""
        result = ar.cleaning.drop_columns_matching(sample_dataframe, "^temp_")
        assert 'temporary' in result.columns
        assert 'temp_col1' not in result.columns
    
    def test_drop_columns_matching_special_characters(self):
        """Test pattern with special regex characters"""
        df = pd.DataFrame({
            'col[1]': [1, 2],
            'col[2]': [3, 4],
            'col_1': [5, 6]
        })
        result = ar.cleaning.drop_columns_matching(df, r"col\[\d\]")
        assert 'col[1]' not in result.columns
        assert 'col[2]' not in result.columns
        assert 'col_1' in result.columns
    
    def test_drop_columns_matching_case_sensitive(self, sample_dataframe):
        """Test that pattern matching is case-sensitive"""
        result = ar.cleaning.drop_columns_matching(sample_dataframe, "^TEMP_")
        assert result.shape == sample_dataframe.shape
        assert all(col in result.columns for col in sample_dataframe.columns)
    
    def test_drop_columns_matching_invalid_regex_raises_error(self, sample_dataframe):
        """Test that invalid regex pattern raises re.error"""
        with pytest.raises(re.error):
            ar.cleaning.drop_columns_matching(sample_dataframe, "[invalid")
    
    def test_drop_columns_matching_pattern_not_string_raises_typeerror(self, sample_dataframe):
        """Test that non-string pattern raises TypeError"""
        with pytest.raises(TypeError):
            ar.cleaning.drop_columns_matching(sample_dataframe, 123)
        
        with pytest.raises(TypeError):
            ar.cleaning.drop_columns_matching(sample_dataframe, None)
    
    def test_drop_columns_matching_all_columns_raises_valueerror(self, sample_dataframe):
        """Test that pattern matching all columns raises ValueError"""
        with pytest.raises(ValueError):
            ar.cleaning.drop_columns_matching(sample_dataframe, ".*")
    
    def test_drop_columns_matching_preserves_data_integrity(self, sample_dataframe):
        """Test that remaining data values are preserved correctly"""
        result = ar.cleaning.drop_columns_matching(sample_dataframe, "^temp_")
        pd.testing.assert_series_equal(result['value'], sample_dataframe['value'])
        assert result['value'].tolist() == [13, 14, 15]
    
    def test_drop_columns_matching_empty_dataframe(self):
        """Test with empty DataFrame"""
        df = pd.DataFrame()
        result = ar.cleaning.drop_columns_matching(df, "^temp_")
        assert len(result.columns) == 0
    
    def test_drop_columns_matching_or_pattern(self, sample_dataframe):
        """Test OR pattern to match multiple column prefixes"""
        result = ar.cleaning.drop_columns_matching(sample_dataframe, "^(temp_|value)")
        assert 'temp_col1' not in result.columns
        assert 'temp_col2' not in result.columns
        assert 'value' not in result.columns
        assert result.shape[1] == 2
