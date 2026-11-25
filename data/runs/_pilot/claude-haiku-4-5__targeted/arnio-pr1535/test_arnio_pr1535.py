"""Tests for arnio.cleaning.filter_rows function"""

import pytest
import pandas as pd
import arnio as ar


class TestFilterRows:
    """Test suite for arnio.cleaning.filter_rows"""
    
    @pytest.fixture
    def df(self):
        """Sample DataFrame for testing"""
        return pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
            'age': [25, 30, 35, 28],
            'score': [85.5, 90.0, 78.5, 92.0]
        })
    
    @pytest.fixture
    def arframe(self, df):
        """Sample ArFrame for testing"""
        return ar.from_dict(df.to_dict(orient='list'))
    
    def test_filter_greater_than(self, df):
        """Test > operator filters correctly"""
        result = ar.filter_rows(df, column='age', op='>', value=28)
        assert len(result) == 2
        assert list(result['age'].values) == [30, 35]
    
    def test_filter_less_than(self, df):
        """Test < operator filters correctly"""
        result = ar.filter_rows(df, column='age', op='<', value=30)
        assert len(result) == 2
    
    def test_filter_greater_equal(self, df):
        """Test >= operator filters correctly"""
        result = ar.filter_rows(df, column='age', op='>=', value=30)
        assert len(result) == 2
    
    def test_filter_less_equal(self, df):
        """Test <= operator filters correctly"""
        result = ar.filter_rows(df, column='age', op='<=', value=28)
        assert len(result) == 2
    
    def test_filter_equal(self, df):
        """Test == operator filters correctly"""
        result = ar.filter_rows(df, column='age', op='==', value=30)
        assert len(result) == 1
    
    def test_filter_not_equal(self, df):
        """Test != operator filters correctly"""
        result = ar.filter_rows(df, column='age', op='!=', value=30)
        assert len(result) == 3
    
    def test_filter_string_column(self, df):
        """Test filtering on string column"""
        result = ar.filter_rows(df, column='name', op='==', value='Alice')
        assert len(result) == 1
        assert result['age'].values[0] == 25
    
    def test_filter_float_column(self, df):
        """Test filtering on float column"""
        result = ar.filter_rows(df, column='score', op='>', value=85.0)
        assert len(result) == 3
    
    def test_filter_no_matches(self, df):
        """Test filter returning empty DataFrame"""
        result = ar.filter_rows(df, column='age', op='>', value=100)
        assert len(result) == 0
    
    def test_filter_all_matches(self, df):
        """Test filter matching all rows"""
        result = ar.filter_rows(df, column='age', op='>', value=20)
        assert len(result) == 4
    
    def test_filter_returns_dataframe(self, df):
        """Test that DataFrame input returns DataFrame output"""
        result = ar.filter_rows(df, column='age', op='>', value=25)
        assert isinstance(result, pd.DataFrame)
    
    def test_filter_returns_arframe(self, arframe):
        """Test that ArFrame input returns ArFrame output"""
        result = ar.filter_rows(arframe, column='age', op='>', value=25)
        assert isinstance(result, ar.ArFrame)
    
    def test_filter_preserves_columns(self, df):
        """Test that column order is preserved"""
        result = ar.filter_rows(df, column='age', op='>', value=25)
        assert list(result.columns) == list(df.columns)
    
    def test_filter_with_zero_comparison(self, df):
        """Test filtering with zero as the comparison value"""
        result = ar.filter_rows(df, column='age', op='>=', value=0)
        assert len(result) == 4
