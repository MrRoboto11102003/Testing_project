import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestFilterRows:
    """Tests for arnio.cleaning.filter_rows"""
    
    @pytest.fixture
    def sample_df(self):
        """Sample DataFrame for testing"""
        return pd.DataFrame({
            'age': [20, 25, 30, 35, 40],
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'score': [85.5, 90.0, 75.5, 95.0, 80.0]
        })
    
    @pytest.fixture
    def df_with_nan(self):
        """DataFrame with NaN values"""
        return pd.DataFrame({
            'value': [1.0, 2.0, np.nan, 4.0, 5.0]
        })
    
    def test_filter_greater_than(self, sample_df):
        """Test > operator"""
        result = ar.filter_rows(sample_df, column='age', op='>', value=25)
        assert list(result['age']) == [30, 35, 40]
    
    def test_filter_less_than(self, sample_df):
        """Test < operator"""
        result = ar.filter_rows(sample_df, column='age', op='<', value=30)
        assert list(result['age']) == [20, 25]
    
    def test_filter_greater_equal(self, sample_df):
        """Test >= operator"""
        result = ar.filter_rows(sample_df, column='age', op='>=', value=30)
        assert list(result['age']) == [30, 35, 40]
    
    def test_filter_less_equal(self, sample_df):
        """Test <= operator"""
        result = ar.filter_rows(sample_df, column='age', op='<=', value=25)
        assert list(result['age']) == [20, 25]
    
    def test_filter_equal(self, sample_df):
        """Test == operator"""
        result = ar.filter_rows(sample_df, column='age', op='==', value=30)
        assert len(result) == 1 and result['age'].iloc[0] == 30
    
    def test_filter_not_equal(self, sample_df):
        """Test != operator"""
        result = ar.filter_rows(sample_df, column='age', op='!=', value=30)
        assert len(result) == 4 and 30 not in result['age'].values
    
    def test_returns_dataframe(self, sample_df):
        """pd.DataFrame input returns pd.DataFrame"""
        result = ar.filter_rows(sample_df, column='age', op='>', value=25)
        assert isinstance(result, pd.DataFrame) and not isinstance(result, ar.ArFrame)
    
    def test_returns_arframe(self, sample_df):
        """ArFrame input returns ArFrame"""
        arframe = ar.from_pandas(sample_df)
        result = ar.filter_rows(arframe, column='age', op='>', value=25)
        assert isinstance(result, ar.ArFrame)
    
    def test_filter_string_column(self, sample_df):
        """Test filtering on string column"""
        result = ar.filter_rows(sample_df, column='name', op='==', value='Bob')
        assert len(result) == 1 and result['name'].iloc[0] == 'Bob'
    
    def test_filter_no_matches(self, sample_df):
        """Test when no rows match the condition"""
        result = ar.filter_rows(sample_df, column='age', op='>', value=100)
        assert len(result) == 0 and list(result.columns) == list(sample_df.columns)
    
    def test_filter_nan_handling(self, df_with_nan):
        """Test that NaN values are excluded from results"""
        result = ar.filter_rows(df_with_nan, column='value', op='>', value=2.0)
        assert len(result) == 2 and np.isnan(result['value']).sum() == 0
    
    def test_error_invalid_column_type(self, sample_df):
        """Column parameter must be a string"""
        with pytest.raises(TypeError):
            ar.filter_rows(sample_df, column=123, op='>', value=25)
    
    def test_error_empty_column(self, sample_df):
        """Column parameter cannot be empty string"""
        with pytest.raises(TypeError):
            ar.filter_rows(sample_df, column='', op='>', value=25)
    
    def test_error_unsupported_operator(self, sample_df):
        """Unsupported operator raises ValueError"""
        with pytest.raises(ValueError, match="Unsupported operator"):
            ar.filter_rows(sample_df, column='age', op='@@', value=25)
    
    def test_error_column_not_found(self, sample_df):
        """Non-existent column raises ValueError"""
        with pytest.raises(ValueError, match="Unknown column"):
            ar.filter_rows(sample_df, column='missing', op='>', value=25)
