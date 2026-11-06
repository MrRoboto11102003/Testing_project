import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestFilterRows:
    """Test arnio.cleaning.filter_rows function."""

    @pytest.fixture
    def df_sample(self):
        """Sample DataFrame with mixed data types."""
        return pd.DataFrame({
            'age': [18, 25, 30, 35, 40],
            'name': ['alice', 'bob', 'charlie', 'diana', 'eve'],
            'score': [85.5, 90.0, 88.2, 92.5, 78.3],
            'active': [True, False, True, False, True]
        })

    @pytest.fixture
    def frame_sample(self, df_sample):
        """Sample ArFrame."""
        return ar.from_pandas(df_sample)

    def test_filter_rows_greater_than_arframe(self, frame_sample):
        """Test > operator returns filtered ArFrame."""
        result = ar.filter_rows(frame_sample, column='age', op='>', value=25)
        assert isinstance(result, ar.ArFrame)
        df = ar.to_pandas(result)
        assert len(df) == 3
        assert all(df['age'] > 25)

    def test_filter_rows_greater_than_dataframe(self, df_sample):
        """Test > operator returns filtered DataFrame."""
        result = ar.filter_rows(df_sample, column='age', op='>', value=25)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    def test_filter_rows_all_operators(self, df_sample):
        """Test all supported operators: <, >=, <=, ==, !=."""
        assert len(ar.filter_rows(df_sample, column='age', op='<', value=30)) == 2
        assert len(ar.filter_rows(df_sample, column='age', op='>=', value=30)) == 3
        assert len(ar.filter_rows(df_sample, column='age', op='<=', value=30)) == 3
        assert len(ar.filter_rows(df_sample, column='age', op='==', value=30)) == 1
        assert len(ar.filter_rows(df_sample, column='age', op='!=', value=30)) == 4

    def test_filter_rows_string_column(self, df_sample):
        """Test filtering on string column."""
        result = ar.filter_rows(df_sample, column='name', op='==', value='bob')
        assert len(result) == 1
        assert result.iloc[0]['name'] == 'bob'

    def test_filter_rows_float_column(self, df_sample):
        """Test filtering on float column."""
        result = ar.filter_rows(df_sample, column='score', op='>', value=90.0)
        assert len(result) == 2

    def test_filter_rows_bool_column(self, df_sample):
        """Test filtering on boolean column."""
        result = ar.filter_rows(df_sample, column='active', op='==', value=True)
        assert len(result) == 3

    def test_filter_rows_empty_result(self, df_sample):
        """Test when no rows match returns empty frame."""
        result = ar.filter_rows(df_sample, column='age', op='>', value=100)
        assert len(result) == 0

    def test_filter_rows_all_match(self, df_sample):
        """Test when all rows match returns all rows."""
        result = ar.filter_rows(df_sample, column='age', op='>', value=10)
        assert len(result) == len(df_sample)

    def test_filter_rows_with_nulls(self):
        """Test filtering with null values treats NaN as non-matching."""
        df = pd.DataFrame({
            'value': [10, 20, np.nan, 30, 40],
            'name': ['a', 'b', 'c', 'd', 'e']
        })
        result = ar.filter_rows(df, column='value', op='>', value=15)
        assert len(result) == 3  # NaN doesn't match

    def test_filter_rows_invalid_operator(self, df_sample):
        """Test unsupported operator raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported operator"):
            ar.filter_rows(df_sample, column='age', op='=>', value=25)

    def test_filter_rows_nonexistent_column(self, df_sample):
        """Test non-existent column raises ValueError."""
        with pytest.raises(ValueError, match="Unknown column"):
            ar.filter_rows(df_sample, column='nonexistent', op='>', value=25)

    def test_filter_rows_invalid_column_type(self, df_sample):
        """Test non-string column raises TypeError."""
        with pytest.raises(TypeError, match="column must be a non-empty string"):
            ar.filter_rows(df_sample, column=123, op='>', value=25)

    def test_filter_rows_empty_column_string(self, df_sample):
        """Test empty column string raises TypeError."""
        with pytest.raises(TypeError, match="column must be a non-empty string"):
            ar.filter_rows(df_sample, column='', op='>', value=25)

    def test_filter_rows_invalid_op_type(self, df_sample):
        """Test non-string op raises TypeError."""
        with pytest.raises(TypeError, match="op must be a string"):
            ar.filter_rows(df_sample, column='age', op=123, value=25)

    def test_filter_rows_non_scalar_value(self, df_sample):
        """Test non-scalar value raises TypeError."""
        with pytest.raises(TypeError, match="value must be a scalar"):
            ar.filter_rows(df_sample, column='age', op='>', value=[25, 30])
