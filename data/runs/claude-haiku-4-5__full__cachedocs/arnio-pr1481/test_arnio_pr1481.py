import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestFilterRows:
    """Test suite for ar.filter_rows function"""

    @pytest.fixture
    def numeric_frame(self):
        """ArFrame with numeric data for testing"""
        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "age": [20, 25, 30, 35, 40],
            "score": [85.5, 90.0, 88.2, 92.5, 87.0],
        })
        return ar.from_pandas(df)

    @pytest.fixture
    def numeric_df(self):
        """pandas DataFrame with numeric data for testing"""
        return pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "age": [20, 25, 30, 35, 40],
            "score": [85.5, 90.0, 88.2, 92.5, 87.0],
        })

    @pytest.fixture
    def mixed_frame(self):
        """ArFrame with mixed data types"""
        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "name": ["alice", "bob", "charlie", "david", "eve"],
            "active": [True, False, True, False, True],
            "value": [10.5, 20.5, 30.5, 40.5, 50.5],
        })
        return ar.from_pandas(df)

    @pytest.fixture
    def mixed_df(self):
        """pandas DataFrame with mixed data types"""
        return pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "name": ["alice", "bob", "charlie", "david", "eve"],
            "active": [True, False, True, False, True],
            "value": [10.5, 20.5, 30.5, 40.5, 50.5],
        })

    # Test all operators with ArFrame
    def test_filter_greater_than_arframe(self, numeric_frame):
        """Test > operator with ArFrame returns ArFrame"""
        result = ar.filter_rows(numeric_frame, column="age", op=">", value=25)
        assert isinstance(result, ar.ArFrame)
        df = ar.to_pandas(result)
        assert len(df) == 3
        assert list(df["age"]) == [30, 35, 40]

    def test_filter_less_than_arframe(self, numeric_frame):
        """Test < operator with ArFrame"""
        result = ar.filter_rows(numeric_frame, column="age", op="<", value=30)
        assert isinstance(result, ar.ArFrame)
        df = ar.to_pandas(result)
        assert len(df) == 2
        assert list(df["age"]) == [20, 25]

    def test_filter_greater_equal_arframe(self, numeric_frame):
        """Test >= operator with ArFrame"""
        result = ar.filter_rows(numeric_frame, column="age", op=">=", value=30)
        df = ar.to_pandas(result)
        assert len(df) == 3
        assert list(df["age"]) == [30, 35, 40]

    def test_filter_less_equal_arframe(self, numeric_frame):
        """Test <= operator with ArFrame"""
        result = ar.filter_rows(numeric_frame, column="age", op="<=", value=30)
        df = ar.to_pandas(result)
        assert len(df) == 3
        assert list(df["age"]) == [20, 25, 30]

    def test_filter_equal_arframe(self, numeric_frame):
        """Test == operator with ArFrame"""
        result = ar.filter_rows(numeric_frame, column="age", op="==", value=30)
        df = ar.to_pandas(result)
        assert len(df) == 1
        assert df["age"].iloc[0] == 30

    def test_filter_not_equal_arframe(self, numeric_frame):
        """Test != operator with ArFrame"""
        result = ar.filter_rows(numeric_frame, column="age", op="!=", value=30)
        df = ar.to_pandas(result)
        assert len(df) == 4
        assert list(df["age"]) == [20, 25, 35, 40]

    # Test with pandas DataFrame (return type preservation)
    def test_filter_returns_dataframe_when_given_dataframe(self, numeric_df):
        """Test that pandas DataFrame input returns pandas DataFrame"""
        result = ar.filter_rows(numeric_df, column="age", op=">", value=25)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert list(result["age"]) == [30, 35, 40]

    def test_filter_equal_on_dataframe(self, numeric_df):
        """Test == operator with pandas DataFrame"""
        result = ar.filter_rows(numeric_df, column="age", op="==", value=25)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    # Test with different column types
    def test_filter_string_column(self, mixed_frame):
        """Test filtering on string column with =="""
        result = ar.filter_rows(mixed_frame, column="name", op="==", value="bob")
        df = ar.to_pandas(result)
        assert len(df) == 1
        assert df["name"].iloc[0] == "bob"

    def test_filter_boolean_column_true(self, mixed_frame):
        """Test filtering boolean column for True values"""
        result = ar.filter_rows(mixed_frame, column="active", op="==", value=True)
        df = ar.to_pandas(result)
        assert len(df) == 3
        assert all(df["active"])

    def test_filter_boolean_column_false(self, mixed_frame):
        """Test filtering boolean column for False values"""
        result = ar.filter_rows(mixed_frame, column="active", op="==", value=False)
        df = ar.to_pandas(result)
        assert len(df) == 2
        assert all(~df["active"])

    def test_filter_float_column(self, numeric_frame):
        """Test filtering on float column"""
        result = ar.filter_rows(numeric_frame, column="score", op=">", value=88.5)
        df = ar.to_pandas(result)
        assert len(df) == 3
        assert all(df["score"] > 88.5)

    # Edge cases
    def test_filter_no_matches(self, numeric_frame):
        """Test filter that matches no rows"""
        result = ar.filter_rows(numeric_frame, column="age", op=">", value=100)
        df = ar.to_pandas(result)
        assert len(df) == 0

    def test_filter_all_matches(self, numeric_frame):
        """Test filter that matches all rows"""
        result = ar.filter_rows(numeric_frame, column="age", op=">", value=10)
        df = ar.to_pandas(result)
        assert len(df) == 5
