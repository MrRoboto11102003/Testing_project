import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestFilterRows:
    
    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
            'age': [25, 30, 35, 28, 22],
            'salary': [50000.0, 60000.0, 75000.0, 55000.0, 45000.0],
            'active': [True, False, True, True, False]
        })
    
    def test_filter_rows_greater_than(self, sample_df):
        """Test filtering with > operator."""
        result = ar.filter_rows(sample_df, column='age', op='>', value=28)
        assert len(result) == 2
        assert all(result['age'] > 28)
    
    def test_filter_rows_less_than(self, sample_df):
        """Test filtering with < operator."""
        result = ar.filter_rows(sample_df, column='age', op='<', value=28)
        assert len(result) == 2
        assert all(result['age'] < 28)
    
    def test_filter_rows_greater_equal(self, sample_df):
        """Test filtering with >= operator."""
        result = ar.filter_rows(sample_df, column='age', op='>=', value=28)
        assert len(result) == 3
        assert all(result['age'] >= 28)
    
    def test_filter_rows_less_equal(self, sample_df):
        """Test filtering with <= operator."""
        result = ar.filter_rows(sample_df, column='age', op='<=', value=28)
        assert len(result) == 3
        assert all(result['age'] <= 28)
    
    def test_filter_rows_equal(self, sample_df):
        """Test filtering with == operator."""
        result = ar.filter_rows(sample_df, column='age', op='==', value=30)
        assert len(result) == 1
        assert result['age'].values[0] == 30
    
    def test_filter_rows_not_equal(self, sample_df):
        """Test filtering with != operator."""
        result = ar.filter_rows(sample_df, column='age', op='!=', value=30)
        assert len(result) == 4
        assert all(result['age'] != 30)
    
    def test_filter_rows_returns_dataframe(self, sample_df):
        """Test that filtering a DataFrame returns a DataFrame."""
        result = ar.filter_rows(sample_df, column='age', op='>', value=25)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == list(sample_df.columns)
    
    def test_filter_rows_empty_result(self, sample_df):
        """Test filtering that results in empty DataFrame."""
        result = ar.filter_rows(sample_df, column='age', op='>', value=100)
        assert len(result) == 0
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == list(sample_df.columns)
    
    def test_filter_rows_full_result(self, sample_df):
        """Test filtering that matches all rows."""
        result = ar.filter_rows(sample_df, column='age', op='>=', value=0)
        assert len(result) == len(sample_df)
    
    def test_filter_rows_invalid_column_type(self, sample_df):
        """Test that non-string column raises TypeError."""
        with pytest.raises(TypeError, match="column must be a non-empty string"):
            ar.filter_rows(sample_df, column=123, op='>', value=25)
    
    def test_filter_rows_empty_column_string(self, sample_df):
        """Test that empty string column raises TypeError."""
        with pytest.raises(TypeError, match="column must be a non-empty string"):
            ar.filter_rows(sample_df, column='', op='>', value=25)
    
    def test_filter_rows_invalid_op_type(self, sample_df):
        """Test that non-string op raises TypeError."""
        with pytest.raises(TypeError, match="op must be a string"):
            ar.filter_rows(sample_df, column='age', op=1, value=25)
    
    def test_filter_rows_unsupported_operator(self, sample_df):
        """Test that unsupported operator raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported operator"):
            ar.filter_rows(sample_df, column='age', op='~~', value=25)
    
    def test_filter_rows_unknown_column(self, sample_df):
        """Test that unknown column raises ValueError."""
        with pytest.raises(ValueError, match="Unknown column"):
            ar.filter_rows(sample_df, column='nonexistent', op='>', value=25)
    
    def test_filter_rows_non_scalar_value(self, sample_df):
        """Test that non-scalar value raises TypeError."""
        with pytest.raises(TypeError, match="value must be a scalar"):
            ar.filter_rows(sample_df, column='age', op='>', value=[25, 30])
