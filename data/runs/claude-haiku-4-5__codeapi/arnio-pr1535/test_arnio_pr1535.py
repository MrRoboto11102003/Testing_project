import pytest
import pandas as pd
import numpy as np
import arnio as ar
from arnio.cleaning import filter_rows
from arnio.convert import from_pandas, to_pandas


class TestFilterRowsOperators:
    """Test filter_rows with various operators."""
    
    def test_filter_rows_greater_than(self):
        """Test filtering with > operator."""
        df = pd.DataFrame({
            'age': [18, 25, 35, 40],
            'name': ['Alice', 'Bob', 'Charlie', 'David']
        })
        frame = from_pandas(df)
        result = filter_rows(frame, column='age', op='>', value=25)
        result_df = to_pandas(result)
        assert result_df.shape[0] == 2
        assert list(result_df['age']) == [35, 40]
    
    def test_filter_rows_less_than(self):
        """Test filtering with < operator."""
        df = pd.DataFrame({
            'age': [18, 25, 35, 40],
            'name': ['Alice', 'Bob', 'Charlie', 'David']
        })
        frame = from_pandas(df)
        result = filter_rows(frame, column='age', op='<', value=30)
        result_df = to_pandas(result)
        assert result_df.shape[0] == 2
        assert list(result_df['age']) == [18, 25]
    
    def test_filter_rows_greater_equal(self):
        """Test filtering with >= operator."""
        df = pd.DataFrame({
            'age': [18, 25, 35, 40],
        })
        frame = from_pandas(df)
        result = filter_rows(frame, column='age', op='>=', value=25)
        result_df = to_pandas(result)
        assert result_df.shape[0] == 3
        assert list(result_df['age']) == [25, 35, 40]
    
    def test_filter_rows_less_equal(self):
        """Test filtering with <= operator."""
        df = pd.DataFrame({
            'age': [18, 25, 35, 40],
        })
        frame = from_pandas(df)
        result = filter_rows(frame, column='age', op='<=', value=30)
        result_df = to_pandas(result)
        assert result_df.shape[0] == 2
        assert list(result_df['age']) == [18, 25]
    
    def test_filter_rows_equal(self):
        """Test filtering with == operator."""
        df = pd.DataFrame({
            'age': [18, 25, 35, 40, 25],
        })
        frame = from_pandas(df)
        result = filter_rows(frame, column='age', op='==', value=25)
        result_df = to_pandas(result)
        assert result_df.shape[0] == 2
        assert list(result_df['age']) == [25, 25]
    
    def test_filter_rows_not_equal(self):
        """Test filtering with != operator."""
        df = pd.DataFrame({
            'age': [18, 25, 35, 40],
        })
        frame = from_pandas(df)
        result = filter_rows(frame, column='age', op='!=', value=25)
        result_df = to_pandas(result)
        assert result_df.shape[0] == 3
        assert list(result_df['age']) == [18, 35, 40]


class TestFilterRowsFrameTypes:
    """Test filter_rows with different input types."""
    
    def test_filter_rows_returns_arframe(self):
        """Test that filter_rows returns ArFrame when given ArFrame."""
        df = pd.DataFrame({'age': [18, 25, 35], 'name': ['A', 'B', 'C']})
        frame = from_pandas(df)
        result = filter_rows(frame, column='age', op='>', value=20)
        assert isinstance(result, ar.frame.ArFrame)
    
    def test_filter_rows_returns_pandas_dataframe(self):
        """Test that filter_rows returns pd.DataFrame when given pd.DataFrame."""
        df = pd.DataFrame({'age': [18, 25, 35], 'name': ['A', 'B', 'C']})
        result = filter_rows(df, column='age', op='>', value=20)
        assert isinstance(result, pd.DataFrame)


class TestFilterRowsEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_filter_rows_no_matches(self):
        """Test filtering that results in empty frame."""
        df = pd.DataFrame({'age': [18, 25, 35], 'name': ['A', 'B', 'C']})
        frame = from_pandas(df)
        result = filter_rows(frame, column='age', op='>', value=100)
        result_df = to_pandas(result)
        assert result_df.shape[0] == 0
        assert list(result_df.columns) == ['age', 'name']
    
    def test_filter_rows_all_matches(self):
        """Test filtering where all rows match."""
        df = pd.DataFrame({'age': [18, 25, 35]})
        frame = from_pandas(df)
        result = filter_rows(frame, column='age', op='>', value=10)
        result_df = to_pandas(result)
        assert result_df.shape[0] == 3
    
    def test_filter_rows_with_null_values(self):
        """Test filtering with NaN values in column - nulls should not match."""
        df = pd.DataFrame({
            'age': [18.0, 25.0, np.nan, 40.0],
        })
        frame = from_pandas(df)
        result = filter_rows(frame, column='age', op='>', value=20)
        result_df = to_pandas(result)
        assert result_df.shape[0] == 2
        assert list(result_df['age']) == [25.0, 40.0]
    
    def test_filter_rows_string_column(self):
        """Test filtering on string column with == operator."""
        df = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie'],
            'id': [1, 2, 3]
        })
        frame = from_pandas(df)
        result = filter_rows(frame, column='name', op='==', value='Bob')
        result_df = to_pandas(result)
        assert result_df.shape[0] == 1
        assert result_df['name'].iloc[0] == 'Bob'
    
    def test_filter_rows_with_duplicates(self):
        """Test filtering with duplicate values."""
        df = pd.DataFrame({
            'value': [5, 5, 10, 5, 15],
        })
        frame = from_pandas(df)
        result = filter_rows(frame, column='value', op='==', value=5)
        result_df = to_pandas(result)
        assert result_df.shape[0] == 3


class TestFilterRowsErrors:
    """Test error handling and validation."""
    
    def test_filter_rows_invalid_column_name(self):
        """Test with non-existent column."""
        df = pd.DataFrame({'age': [18, 25], 'name': ['A', 'B']})
        frame = from_pandas(df)
        with pytest.raises(ValueError, match="Unknown column"):
            filter_rows(frame, column='nonexistent', op='>', value=20)
    
    def test_filter_rows_invalid_operator(self):
        """Test with unsupported operator."""
        df = pd.DataFrame({'age': [18, 25]})
        frame = from_pandas(df)
        with pytest.raises(ValueError, match="Unsupported operator"):
            filter_rows(frame, column='age', op='~=', value=20)
    
    def test_filter_rows_non_string_column(self):
        """Test with non-string column parameter."""
        df = pd.DataFrame({'age': [18, 25]})
        frame = from_pandas(df)
        with pytest.raises(TypeError, match="column must be a non-empty string"):
            filter_rows(frame, column=123, op='>', value=20)
    
    def test_filter_rows_empty_string_column(self):
        """Test with whitespace-only column parameter."""
        df = pd.DataFrame({'age': [18, 25]})
        frame = from_pandas(df)
        with pytest.raises(TypeError, match="column must be a non-empty string"):
            filter_rows(frame, column='  ', op='>', value=20)
    
    def test_filter_rows_non_string_operator(self):
        """Test with non-string operator parameter."""
        df = pd.DataFrame({'age': [18, 25]})
        frame = from_pandas(df)
        with pytest.raises(TypeError, match="op must be a string"):
            filter_rows(frame, column='age', op=42, value=20)
    
    def test_filter_rows_non_scalar_value(self):
        """Test with non-scalar value parameter."""
        df = pd.DataFrame({'age': [18, 25]})
        frame = from_pandas(df)
        with pytest.raises(TypeError, match="filter_rows value must be a scalar"):
            filter_rows(frame, column='age', op='>', value=[20, 30])
    
    def test_filter_rows_incompatible_types(self):
        """Test filtering with incompatible types (string column, numeric operator)."""
        df = pd.DataFrame({'name': ['Alice', 'Bob']})
        frame = from_pandas(df)
        with pytest.raises(TypeError, match="cannot compare column"):
            filter_rows(frame, column='name', op='>', value=25)
