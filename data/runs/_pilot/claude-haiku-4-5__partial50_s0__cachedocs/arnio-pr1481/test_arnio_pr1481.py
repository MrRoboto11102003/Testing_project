import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestFilterRows:
    """Test suite for ar.filter_rows functionality."""

    @pytest.fixture
    def arframe_sample(self):
        """Create a sample ArFrame for testing."""
        df = pd.DataFrame({
            "age": [20, 25, 30, 35, 40],
            "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
            "score": [85.5, 90.0, 78.5, 95.0, 88.0],
            "active": [True, False, True, False, True],
        })
        return ar.from_pandas(df)

    @pytest.fixture
    def dataframe_sample(self):
        """Create a sample pandas DataFrame for testing."""
        return pd.DataFrame({
            "age": [20, 25, 30, 35, 40],
            "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
            "score": [85.5, 90.0, 78.5, 95.0, 88.0],
            "active": [True, False, True, False, True],
        })

    def test_filter_greater_than_arframe(self, arframe_sample):
        """Test > operator with ArFrame input returns ArFrame."""
        result = ar.filter_rows(arframe_sample, column="age", op=">", value=30)
        assert hasattr(result, 'columns')
        df = ar.to_pandas(result)
        assert len(df) == 2
        assert all(df["age"] > 30)

    def test_filter_greater_than_pandas(self, dataframe_sample):
        """Test > operator with pandas DataFrame returns DataFrame."""
        result = ar.filter_rows(dataframe_sample, column="age", op=">", value=30)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert all(result["age"] > 30)

    def test_filter_less_than(self, arframe_sample):
        """Test < operator."""
        result = ar.filter_rows(arframe_sample, column="age", op="<", value=30)
        df = ar.to_pandas(result)
        assert len(df) == 2
        assert all(df["age"] < 30)

    def test_filter_greater_equal(self, arframe_sample):
        """Test >= operator."""
        result = ar.filter_rows(arframe_sample, column="age", op=">=", value=30)
        df = ar.to_pandas(result)
        assert len(df) == 3

    def test_filter_less_equal(self, arframe_sample):
        """Test <= operator."""
        result = ar.filter_rows(arframe_sample, column="age", op="<=", value=30)
        df = ar.to_pandas(result)
        assert len(df) == 3

    def test_filter_equal(self, arframe_sample):
        """Test == operator."""
        result = ar.filter_rows(arframe_sample, column="age", op="==", value=30)
        df = ar.to_pandas(result)
        assert len(df) == 1
        assert df.iloc[0]["age"] == 30

    def test_filter_not_equal(self, arframe_sample):
        """Test != operator."""
        result = ar.filter_rows(arframe_sample, column="age", op="!=", value=30)
        df = ar.to_pandas(result)
        assert len(df) == 4

    def test_filter_string_column(self, arframe_sample):
        """Test filtering on string column."""
        result = ar.filter_rows(arframe_sample, column="name", op="==", value="Alice")
        df = ar.to_pandas(result)
        assert len(df) == 1
        assert df.iloc[0]["name"] == "Alice"

    def test_filter_float_column(self, arframe_sample):
        """Test filtering on float column."""
        result = ar.filter_rows(arframe_sample, column="score", op=">", value=85.0)
        df = ar.to_pandas(result)
        assert len(df) == 3
        assert all(df["score"] > 85.0)

    def test_filter_boolean_column(self, arframe_sample):
        """Test filtering on boolean column."""
        result = ar.filter_rows(arframe_sample, column="active", op="==", value=True)
        df = ar.to_pandas(result)
        assert len(df) == 3

    def test_filter_empty_result(self, arframe_sample):
        """Test filter returning no rows."""
        result = ar.filter_rows(arframe_sample, column="age", op=">", value=100)
        df = ar.to_pandas(result)
        assert len(df) == 0

    def test_filter_all_match(self, arframe_sample):
        """Test filter where all rows match."""
        result = ar.filter_rows(arframe_sample, column="age", op=">=", value=0)
        df = ar.to_pandas(result)
        assert len(df) == 5

    def test_filter_nonexistent_column_raises(self, arframe_sample):
        """Test that nonexistent column raises error."""
        with pytest.raises((KeyError, ValueError)):
            ar.filter_rows(arframe_sample, column="nonexistent", op=">", value=10)

    def test_filter_invalid_operator_raises(self, arframe_sample):
        """Test that invalid operator raises error."""
        with pytest.raises((ValueError, KeyError)):
            ar.filter_rows(arframe_sample, column="age", op=">>", value=10)

    def test_filter_preserves_column_order(self, arframe_sample):
        """Test that filtering preserves column order."""
        result = ar.filter_rows(arframe_sample, column="age", op=">", value=25)
        df = ar.to_pandas(result)
        assert list(df.columns) == ["age", "name", "score", "active"]
