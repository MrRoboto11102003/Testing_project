import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestFilterRows:
    """Test suite for arnio.cleaning.filter_rows"""
    
    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing"""
        return pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'age': [25, 30, 35, 28, 32],
            'salary': [50000, 60000, 75000, 55000, 65000],
            'department': ['Sales', 'IT', 'HR', 'Sales', 'IT']
        })
    
    def test_filter_rows_equality_operator(self, sample_df):
        """Test filtering with == operator on integer column"""
        result = ar.cleaning.filter_rows(sample_df, 'age', '==', 30)
        assert len(result) == 1
        assert result['name'].iloc[0] == 'Bob'
    
    def test_filter_rows_greater_than_operator(self, sample_df):
        """Test filtering with > operator"""
        result = ar.cleaning.filter_rows(sample_df, 'age', '>', 30)
        assert len(result) == 2
        assert all(result['age'] > 30)
    
    def test_filter_rows_less_than_operator(self, sample_df):
        """Test filtering with < operator"""
        result = ar.cleaning.filter_rows(sample_df, 'age', '<', 30)
        assert len(result) == 1
        assert result['age'].iloc[0] == 25
    
    def test_filter_rows_gte_operator(self, sample_df):
        """Test filtering with >= operator"""
        result = ar.cleaning.filter_rows(sample_df, 'age', '>=', 30)
        assert len(result) == 4
        assert all(result['age'] >= 30)
    
    def test_filter_rows_lte_operator(self, sample_df):
        """Test filtering with <= operator"""
        result = ar.cleaning.filter_rows(sample_df, 'age', '<=', 30)
        assert len(result) == 2
        assert all(result['age'] <= 30)
    
    def test_filter_rows_not_equal_operator(self, sample_df):
        """Test filtering with != operator"""
        result = ar.cleaning.filter_rows(sample_df, 'age', '!=', 30)
        assert len(result) == 4
        assert 30 not in result['age'].values
    
    def test_filter_rows_string_column(self, sample_df):
        """Test filtering on string column"""
        result = ar.cleaning.filter_rows(sample_df, 'department', '==', 'IT')
        assert len(result) == 2
        assert all(result['department'] == 'IT')
    
    def test_filter_rows_no_matches(self, sample_df):
        """Test filtering with no matching rows returns empty frame"""
        result = ar.cleaning.filter_rows(sample_df, 'age', '==', 999)
        assert len(result) == 0
        assert isinstance(result, pd.DataFrame)
    
    def test_filter_rows_all_matches(self, sample_df):
        """Test filtering where all rows match"""
        result = ar.cleaning.filter_rows(sample_df, 'age', '>', 20)
        assert len(result) == len(sample_df)
    
    def test_filter_rows_float_values(self):
        """Test filtering with float values"""
        df = pd.DataFrame({'value': [1.5, 2.3, 3.7, 2.1]})
        result = ar.cleaning.filter_rows(df, 'value', '>', 2.5)
        assert len(result) == 1
        assert np.isclose(result['value'].iloc[0], 3.7)
    
    def test_filter_rows_empty_dataframe(self):
        """Test filtering on empty DataFrame"""
        empty_df = pd.DataFrame({'col': pd.Series([], dtype='int64')})
        result = ar.cleaning.filter_rows(empty_df, 'col', '==', 1)
        assert len(result) == 0
    
    def test_filter_rows_preserves_columns(self, sample_df):
        """Test that filtering preserves all columns"""
        result = ar.cleaning.filter_rows(sample_df, 'age', '>', 25)
        assert list(result.columns) == list(sample_df.columns)
    
    def test_filter_rows_preserves_dtypes(self, sample_df):
        """Test that filtering preserves column data types"""
        result = ar.cleaning.filter_rows(sample_df, 'age', '>', 25)
        for col in sample_df.columns:
            assert result[col].dtype == sample_df[col].dtype
    
    def test_filter_rows_preserves_index(self, sample_df):
        """Test that filtering preserves original row indices"""
        result = ar.cleaning.filter_rows(sample_df, 'name', '==', 'Bob')
        assert 1 in result.index
