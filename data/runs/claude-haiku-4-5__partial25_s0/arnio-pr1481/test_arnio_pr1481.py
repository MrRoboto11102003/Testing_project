import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestFilterRows:
    """Test suite for arnio.cleaning.filter_rows function."""

    def test_filter_rows_greater_than_integers(self):
        """Test filtering with > operator on integer column."""
        df = pd.DataFrame({"age": [18, 25, 30, 16, 22]})
        result = ar.filter_rows(df, column="age", op=">", value=20)
        assert len(result) == 3
        assert list(result["age"]) == [25, 30, 22]

    def test_filter_rows_less_than_floats(self):
        """Test filtering with < operator on float column."""
        df = pd.DataFrame({"price": [10.5, 20.3, 5.2, 15.8]})
        result = ar.filter_rows(df, column="price", op="<", value=15.0)
        assert len(result) == 2
        assert list(result["price"]) == [10.5, 5.2]

    def test_filter_rows_greater_equal(self):
        """Test filtering with >= operator."""
        df = pd.DataFrame({"score": [80, 90, 85, 75, 95]})
        result = ar.filter_rows(df, column="score", op=">=", value=85)
        assert len(result) == 3
        assert list(result["score"]) == [90, 85, 95]

    def test_filter_rows_less_equal(self):
        """Test filtering with <= operator."""
        df = pd.DataFrame({"value": [10, 5, 20, 15, 8]})
        result = ar.filter_rows(df, column="value", op="<=", value=12)
        assert len(result) == 3
        assert list(result["value"]) == [10, 5, 8]

    def test_filter_rows_equality_strings(self):
        """Test filtering with == operator on string column."""
        df = pd.DataFrame({"status": ["active", "inactive", "active", "pending"]})
        result = ar.filter_rows(df, column="status", op="==", value="active")
        assert len(result) == 2
        assert list(result["status"]) == ["active", "active"]

    def test_filter_rows_not_equal(self):
        """Test filtering with != operator."""
        df = pd.DataFrame({"category": ["A", "B", "A", "C", "B"]})
        result = ar.filter_rows(df, column="category", op="!=", value="A")
        assert len(result) == 3
        assert list(result["category"]) == ["B", "C", "B"]

    def test_filter_rows_with_booleans(self):
        """Test filtering on boolean column."""
        df = pd.DataFrame({"is_active": [True, False, True, False, True]})
        result = ar.filter_rows(df, column="is_active", op="==", value=True)
        assert len(result) == 3

    def test_filter_rows_returns_pandas_dataframe_type(self):
        """Test that pandas DataFrame input returns pandas DataFrame."""
        df = pd.DataFrame({"id": [1, 2, 3, 4, 5]})
        result = ar.filter_rows(df, column="id", op=">", value=2)
        assert isinstance(result, pd.DataFrame)
        assert not isinstance(result, ar.ArFrame) or isinstance(result, pd.DataFrame)

    def test_filter_rows_returns_arframe_type(self):
        """Test that ArFrame input returns ArFrame."""
        df = pd.DataFrame({"id": [1, 2, 3, 4, 5]})
        frame = ar.from_pandas(df)
        result = ar.filter_rows(frame, column="id", op=">", value=2)
        assert isinstance(result, ar.ArFrame)

    def test_filter_rows_all_rows_match(self):
        """Test filtering where all rows match the condition."""
        df = pd.DataFrame({"value": [10, 20, 30, 40]})
        result = ar.filter_rows(df, column="value", op=">", value=5)
        assert len(result) == 4

    def test_filter_rows_no_rows_match(self):
        """Test filtering where no rows match the condition."""
        df = pd.DataFrame({"value": [10, 20, 30, 40]})
        result = ar.filter_rows(df, column="value", op=">", value=100)
        assert len(result) == 0

    def test_filter_rows_empty_dataframe(self):
        """Test filtering an empty DataFrame."""
        df = pd.DataFrame({"value": pd.Series([], dtype="int64")})
        result = ar.filter_rows(df, column="value", op=">", value=10)
        assert len(result) == 0

    def test_filter_rows_preserves_multicolumn_structure(self):
        """Test filtering preserves all columns in multi-column DataFrame."""
        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "age": [25, 30, 28, 35, 22]
        })
        result = ar.filter_rows(df, column="age", op=">=", value=28)
        assert len(result) == 3
        assert list(result.columns) == ["id", "name", "age"]
        assert list(result["name"]) == ["Charlie", "David"]

    def test_filter_rows_with_null_values(self):
        """Test filtering behavior with null/NaN values in the column."""
        df = pd.DataFrame({"value": [10.0, np.nan, 30.0, 20.0, None]})
        result = ar.filter_rows(df, column="value", op=">", value=15)
        # NaNs and None should be excluded from results
        assert len(result) == 2
        assert list(result["value"]) == [30.0, 20.0]

    def test_filter_rows_nonexistent_column_raises_error(self):
        """Test that filtering on non-existent column raises error."""
        df = pd.DataFrame({"id": [1, 2, 3]})
        with pytest.raises((KeyError, ValueError)):
            ar.filter_rows(df, column="nonexistent", op=">", value=1)

    def test_filter_rows_invalid_operator_raises_error(self):
        """Test that invalid operator raises error."""
        df = pd.DataFrame({"value": [1, 2, 3]})
        with pytest.raises((ValueError, KeyError)):
            ar.filter_rows(df, column="value", op=">>", value=2)

    def test_filter_rows_equality_numeric_zero(self):
        """Test filtering with == operator comparing to zero."""
        df = pd.DataFrame({"count": [0, 1, 0, 2, 0]})
        result = ar.filter_rows(df, column="count", op="==", value=0)
        assert len(result) == 3
        assert list(result["count"]) == [0, 0, 0]

    def test_filter_rows_preserves_column_dtypes(self):
        """Test that column dtypes are preserved after filtering."""
        df = pd.DataFrame({
            "int_col": [1, 2, 3],
            "float_col": [1.5, 2.5, 3.5],
            "str_col": ["a", "b", "c"]
        })
        result = ar.filter_rows(df, column="int_col", op=">", value=1)
        assert result["int_col"].dtype == df["int_col"].dtype
        assert result["float_col"].dtype == df["float_col"].dtype
        assert result["str_col"].dtype == df["str_col"].dtype
