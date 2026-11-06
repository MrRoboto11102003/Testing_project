import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestFilterRows:
    """Tests for arnio.cleaning.filter_rows function."""

    @pytest.fixture
    def sample_arframe(self):
        """Create a sample ArFrame for testing."""
        data = {
            "id": [1, 2, 3, 4, 5],
            "age": [20, 25, 30, 35, 40],
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "score": [85.5, 90.2, 78.3, 92.1, 88.7],
            "active": [True, False, True, False, True],
        }
        return ar.from_pandas(pd.DataFrame(data))

    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample pandas DataFrame for testing."""
        data = {
            "id": [1, 2, 3, 4, 5],
            "age": [20, 25, 30, 35, 40],
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "score": [85.5, 90.2, 78.3, 92.1, 88.7],
        }
        return pd.DataFrame(data)

    def test_filter_greater_than_operator(self, sample_arframe):
        """Test filtering with > operator."""
        result = ar.filter_rows(sample_arframe, column="age", op=">", value=30)
        df = ar.to_pandas(result)
        assert len(df) == 2
        assert all(df["age"] > 30)

    def test_filter_less_than_operator(self, sample_arframe):
        """Test filtering with < operator."""
        result = ar.filter_rows(sample_arframe, column="age", op="<", value=30)
        df = ar.to_pandas(result)
        assert len(df) == 2
        assert all(df["age"] < 30)

    def test_filter_equal_operator(self, sample_arframe):
        """Test filtering with == operator."""
        result = ar.filter_rows(sample_arframe, column="age", op="==", value=30)
        df = ar.to_pandas(result)
        assert len(df) == 1
        assert df["age"].iloc[0] == 30

    def test_filter_not_equal_operator(self, sample_arframe):
        """Test filtering with != operator."""
        result = ar.filter_rows(sample_arframe, column="age", op="!=", value=30)
        df = ar.to_pandas(result)
        assert len(df) == 4
        assert all(df["age"] != 30)

    def test_filter_greater_equal_operator(self, sample_arframe):
        """Test filtering with >= operator."""
        result = ar.filter_rows(sample_arframe, column="age", op=">=", value=30)
        df = ar.to_pandas(result)
        assert len(df) == 3
        assert all(df["age"] >= 30)

    def test_filter_less_equal_operator(self, sample_arframe):
        """Test filtering with <= operator."""
        result = ar.filter_rows(sample_arframe, column="age", op="<=", value=30)
        df = ar.to_pandas(result)
        assert len(df) == 3
        assert all(df["age"] <= 30)

    def test_filter_string_column(self, sample_arframe):
        """Test filtering on string column with equality."""
        result = ar.filter_rows(sample_arframe, column="name", op="==", value="Alice")
        df = ar.to_pandas(result)
        assert len(df) == 1
        assert df["name"].iloc[0] == "Alice"

    def test_filter_boolean_column(self, sample_arframe):
        """Test filtering on boolean column."""
        result = ar.filter_rows(sample_arframe, column="active", op="==", value=True)
        df = ar.to_pandas(result)
        assert len(df) == 3
        assert all(df["active"])

    def test_filter_float_column(self, sample_arframe):
        """Test filtering on float column."""
        result = ar.filter_rows(sample_arframe, column="score", op=">", value=85.0)
        df = ar.to_pandas(result)
        assert len(df) == 4
        assert all(df["score"] > 85.0)

    def test_filter_returns_arframe_type(self, sample_arframe):
        """Test that filtering ArFrame returns ArFrame."""
        result = ar.filter_rows(sample_arframe, column="age", op=">", value=25)
        assert isinstance(result, ar.ArFrame)

    def test_filter_returns_pandas_dataframe_type(self, sample_dataframe):
        """Test that filtering pandas DataFrame returns pandas DataFrame."""
        result = ar.filter_rows(sample_dataframe, column="age", op=">", value=25)
        assert isinstance(result, pd.DataFrame)

    def test_filter_empty_result(self, sample_arframe):
        """Test filtering that results in empty frame."""
        result = ar.filter_rows(sample_arframe, column="age", op=">", value=100)
        df = ar.to_pandas(result)
        assert len(df) == 0
        assert list(df.columns) == list(ar.to_pandas(sample_arframe).columns)

    def test_invalid_column_raises_error(self, sample_arframe):
        """Test that filtering on non-existent column raises ValueError."""
        with pytest.raises(ValueError, match="Unknown column"):
            ar.filter_rows(sample_arframe, column="nonexistent", op=">", value=10)

    def test_invalid_operator_raises_error(self, sample_arframe):
        """Test that invalid operator raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported operator"):
            ar.filter_rows(sample_arframe, column="age", op=">>", value=30)

    def test_non_string_column_raises_error(self, sample_arframe):
        """Test that non-string column parameter raises TypeError."""
        with pytest.raises(TypeError, match="column must be a non-empty string"):
            ar.filter_rows(sample_arframe, column=123, op=">", value=30)

    def test_null_values_in_column_handled(self):
        """Test that null/NaN values are handled correctly (excluded from results)."""
        data = {"id": [1, 2, 3], "value": [10.0, np.nan, 30.0]}
        frame = ar.from_pandas(pd.DataFrame(data))
        result = ar.filter_rows(frame, column="value", op=">", value=15)
        df = ar.to_pandas(result)
        # Only non-NaN values greater than 15
        assert len(df) == 1
        assert df["value"].iloc[0] == 30.0
