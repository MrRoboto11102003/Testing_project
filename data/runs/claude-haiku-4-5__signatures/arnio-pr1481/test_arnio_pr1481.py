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
            'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
            'age': [25, 30, 35, 28, 32],
            'salary': [50000, 60000, 75000, 55000, 65000],
            'department': ['HR', 'IT', 'IT', 'HR', 'Finance']
        })
    
    @pytest.fixture
    def sample_arframe(self, sample_df):
        """Create a sample ArFrame for testing"""
        return ar.ArFrame(sample_df)
    
    def test_filter_equal_operator(self, sample_df):
        """Test filtering with equality operator on string column"""
        result = ar.cleaning.filter_rows(sample_df, 'department', '==', 'IT')
        assert len(result) == 2
        assert all(result['department'] == 'IT')
    
    def test_filter_not_equal_operator(self, sample_df):
        """Test filtering with not equal operator"""
        result = ar.cleaning.filter_rows(sample_df, 'department', '!=', 'IT')
        assert len(result) == 3
        assert all(result['department'] != 'IT')
    
    def test_filter_greater_than(self, sample_df):
        """Test filtering with greater than operator on numeric column"""
        result = ar.cleaning.filter_rows(sample_df, 'age', '>', 30)
        assert len(result) == 2
        assert all(result['age'] > 30)
    
    def test_filter_less_than(self, sample_df):
        """Test filtering with less than operator"""
        result = ar.cleaning.filter_rows(sample_df, 'age', '<', 30)
        assert len(result) == 1
        assert result['age'].iloc[0] == 25
    
    def test_filter_greater_equal(self, sample_df):
        """Test filtering with greater than or equal operator"""
        result = ar.cleaning.filter_rows(sample_df, 'age', '>=', 30)
        assert len(result) == 4
        assert all(result['age'] >= 30)
    
    def test_filter_less_equal(self, sample_df):
        """Test filtering with less than or equal operator"""
        result = ar.cleaning.filter_rows(sample_df, 'age', '<=', 30)
        assert len(result) == 2
        assert all(result['age'] <= 30)
    
    def test_filter_in_operator(self, sample_df):
        """Test filtering with 'in' operator"""
        result = ar.cleaning.filter_rows(sample_df, 'department', 'in', ['IT', 'HR'])
        assert len(result) == 4
        assert all(result['department'].isin(['IT', 'HR']))
    
    def test_filter_not_in_operator(self, sample_df):
        """Test filtering with 'not in' operator"""
        result = ar.cleaning.filter_rows(sample_df, 'department', 'not in', ['IT'])
        assert len(result) == 3
        assert all(~result['department'].isin(['IT']))
    
    def test_filter_no_matches(self, sample_df):
        """Test filtering when no rows match criteria"""
        result = ar.cleaning.filter_rows(sample_df, 'age', '>', 100)
        assert len(result) == 0
        assert list(result.columns) == list(sample_df.columns)
    
    def test_filter_all_matches(self, sample_df):
        """Test filtering when all rows match criteria"""
        result = ar.cleaning.filter_rows(sample_df, 'age', '>', 0)
        assert len(result) == len(sample_df)
    
    def test_filter_empty_dataframe(self):
        """Test filtering on empty DataFrame"""
        empty_df = pd.DataFrame({'col': pd.Series([], dtype='object')})
        result = ar.cleaning.filter_rows(empty_df, 'col', '==', 'value')
        assert len(result) == 0
        assert isinstance(result, pd.DataFrame)
    
    def test_filter_preserves_arframe_type(self, sample_arframe):
        """Test that filtering an ArFrame returns an ArFrame"""
        result = ar.cleaning.filter_rows(sample_arframe, 'age', '>', 25)
        assert isinstance(result, ar.ArFrame)
        assert len(result) == 4
    
    def test_filter_preserves_dataframe_type(self, sample_df):
        """Test that filtering a DataFrame returns a DataFrame"""
        result = ar.cleaning.filter_rows(sample_df, 'age', '>', 25)
        assert isinstance(result, pd.DataFrame)
        assert not isinstance(result, ar.ArFrame)
    
    def test_filter_single_row_match(self, sample_df):
        """Test filtering returns exactly one matching row"""
        result = ar.cleaning.filter_rows(sample_df, 'age', '==', 25)
        assert len(result) == 1
        assert result['name'].iloc[0] == 'Alice'
    
    def test_filter_maintains_column_structure(self, sample_df):
        """Test that filtering maintains all columns"""
        result = ar.cleaning.filter_rows(sample_df, 'age', '>', 30)
        assert list(result.columns) == list(sample_df.columns)
        assert len(result.columns) == 4
