import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestFilterRows:
    """Test suite for ar.filter_rows()"""

    @pytest.fixture
    def sample_df(self):
        """Sample DataFrame for testing"""
        return pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "age": [20, 25, 30, 35, 40],
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "score": [85.5, 90.0, 78.3, 92.1, 88.7],
        })

    @pytest.fixture
    def sample_frame(self, sample_df):
        """Sample ArFrame for testing"""
        return ar.from_pandas(sample_df)

    def test_filter_greater_than(self, sample_df):
        """Test > operator with numeric column"""
        result = ar.filter_rows(sample_df, column="age", op=">", value=25)
        assert len(result) == 3
        assert all(result["age"] > 25)
        assert list(result["age"]) == [30, 35, 40]

    def test_filter_less_than(self, sample_df):
        """Test < operator with numeric column"""
        result = ar.filter_rows(sample_df, column="age", op="<", value=30)
        assert len(result) == 2
        assert all(result["age"] < 30)
        assert list(result["age"]) == [20, 25]

    def test_filter_greater_equal(self, sample_df):
        """Test >= operator"""
        result = ar.filter_rows(sample_df, column="age", op=">=", value=30)
        assert len(result) == 3
        assert all(result["age"] >= 30)

    def test_filter_less_equal(self, sample_df):
        """Test <= operator"""
        result = ar.filter_rows(sample_df, column="age", op="<=", value=30)
        assert len(result) == 3
        assert all(result["age"] <= 30)

    def test_filter_equal(self, sample_df):
        """Test == operator"""
        result = ar.filter_rows(sample_df, column="age", op="==", value=30)
        assert len(result) == 1
        assert result["age"].iloc[0] == 30

    def test_filter_not_equal(self, sample_df):
        """Test != operator"""
        result = ar.filter_rows(sample_df, column="age", op="!=", value=30)
        assert len(result) == 4
        assert all(result["age"] != 30)

    def test_filter_string_column_equal(self, sample_df):
        """Test == operator on string column"""
        result = ar.filter_rows(sample_df, column="name", op="==", value="Alice")
        assert len(result) == 1
        assert result["name"].iloc[0] == "Alice"

    def test_filter_string_column_not_equal(self, sample_df):
        """Test != operator on string column"""
        result = ar.filter_rows(sample_df, column="name", op="!=", value="Alice")
        assert len(result) == 4
        assert all(result["name"] != "Alice")

    def test_filter_float_column(self, sample_df):
        """Test filtering on float column"""
        result = ar.filter_rows(sample_df, column="score", op=">", value=90.0)
        assert len(result) == 2
        assert all(result["score"] > 90.0)

    def test_filter_all_rows_match(self, sample_df):
        """Test when all rows match the condition"""
        result = ar.filter_rows(sample_df, column="age", op=">", value=0)
        pd.testing.assert_frame_equal(result, sample_df)

    def test_filter_no_rows_match(self, sample_df):
        """Test when no rows match the condition"""
        result = ar.filter_rows(sample_df, column="age", op=">", value=1000)
        assert len(result) == 0
        assert list(result.columns) == list(sample_df.columns)

    def test_filter_preserves_dataframe_type(self, sample_df):
        """Test that filtering pd.DataFrame returns pd.DataFrame"""
        result = ar.filter_rows(sample_df, column="age", op=">", value=25)
        assert isinstance(result, pd.DataFrame)

    def test_filter_preserves_arframe_type(self, sample_frame):
        """Test that filtering ArFrame returns ArFrame"""
        result = ar.filter_rows(sample_frame, column="age", op=">", value=25)
        assert isinstance(result, ar.ArFrame)

    def test_filter_nonexistent_column_raises_error(self, sample_df):
        """Test that filtering on nonexistent column raises error"""
        with pytest.raises((KeyError, ValueError)):
            ar.filter_rows(sample_df, column="nonexistent", op=">", value=25)

    def test_filter_invalid_operator_raises_error(self, sample_df):
        """Test that invalid operator raises error"""
        with pytest.raises((ValueError, KeyError)):
            ar.filter_rows(sample_df, column="age", op=">>>", value=25)
