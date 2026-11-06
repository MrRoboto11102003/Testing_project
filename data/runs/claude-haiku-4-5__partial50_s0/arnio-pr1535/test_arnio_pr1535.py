import pytest
import pandas as pd
import arnio as ar


class TestFilterRows:
    """Tests for arnio.cleaning.filter_rows function"""

    def test_filter_greater_than(self):
        """Test filtering with > operator"""
        df = pd.DataFrame({"age": [18, 25, 30, 35]})
        result = ar.filter_rows(ar.from_pandas(df), column="age", op=">", value=25)
        assert list(ar.to_pandas(result)["age"]) == [30, 35]

    def test_filter_less_than(self):
        """Test filtering with < operator"""
        df = pd.DataFrame({"age": [18, 25, 30, 35]})
        result = ar.filter_rows(ar.from_pandas(df), column="age", op="<", value=25)
        assert list(ar.to_pandas(result)["age"]) == [18]

    def test_filter_greater_equal(self):
        """Test filtering with >= operator"""
        df = pd.DataFrame({"value": [1, 2, 3, 4, 5]})
        result = ar.filter_rows(ar.from_pandas(df), column="value", op=">=", value=3)
        assert list(ar.to_pandas(result)["value"]) == [3, 4, 5]

    def test_filter_less_equal(self):
        """Test filtering with <= operator"""
        df = pd.DataFrame({"value": [1, 2, 3, 4, 5]})
        result = ar.filter_rows(ar.from_pandas(df), column="value", op="<=", value=3)
        assert list(ar.to_pandas(result)["value"]) == [1, 2, 3]

    def test_filter_equal(self):
        """Test filtering with == operator"""
        df = pd.DataFrame({"id": [1, 2, 3, 2, 4]})
        result = ar.filter_rows(ar.from_pandas(df), column="id", op="==", value=2)
        assert list(ar.to_pandas(result)["id"]) == [2, 2]

    def test_filter_not_equal(self):
        """Test filtering with != operator"""
        df = pd.DataFrame({"id": [1, 2, 3, 2, 4]})
        result = ar.filter_rows(ar.from_pandas(df), column="id", op="!=", value=2)
        assert list(ar.to_pandas(result)["id"]) == [1, 3, 4]

    def test_filter_float_column(self):
        """Test filtering with float data type"""
        df = pd.DataFrame({"score": [10.5, 20.3, 15.8, 30.2]})
        result = ar.filter_rows(ar.from_pandas(df), column="score", op=">", value=20.0)
        assert ar.to_pandas(result)["score"].tolist() == pytest.approx([20.3, 30.2])

    def test_filter_string_column(self):
        """Test filtering with string data type"""
        df = pd.DataFrame({"status": ["active", "inactive", "active"]})
        result = ar.filter_rows(ar.from_pandas(df), column="status", op="==", value="active")
        assert list(ar.to_pandas(result)["status"]) == ["active", "active"]

    def test_filter_boolean_column(self):
        """Test filtering with boolean data type"""
        df = pd.DataFrame({"is_active": [True, False, True, False]})
        result = ar.filter_rows(ar.from_pandas(df), column="is_active", op="==", value=True)
        assert list(ar.to_pandas(result)["is_active"]) == [True, True]

    def test_filter_no_matches(self):
        """Test filtering when no rows satisfy the condition"""
        df = pd.DataFrame({"age": [18, 19, 20]})
        result = ar.filter_rows(ar.from_pandas(df), column="age", op=">", value=100)
        result_df = ar.to_pandas(result)
        assert len(result_df) == 0
        assert list(result_df.columns) == ["age"]

    def test_filter_all_rows_match(self):
        """Test filtering when all rows satisfy the condition"""
        df = pd.DataFrame({"age": [25, 30, 35]})
        result = ar.filter_rows(ar.from_pandas(df), column="age", op=">", value=20)
        assert len(ar.to_pandas(result)) == 3

    def test_filter_preserves_column_order(self):
        """Test that column order is preserved after filtering"""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        result = ar.filter_rows(ar.from_pandas(df), column="b", op=">", value=4)
        assert list(ar.to_pandas(result).columns) == ["a", "b", "c"]

    def test_filter_with_pandas_dataframe(self):
        """Test filtering directly on pandas DataFrame"""
        df = pd.DataFrame({"age": [20, 25, 30]})
        result = ar.filter_rows(df, column="age", op=">", value=22)
        assert isinstance(result, pd.DataFrame)
        assert list(result["age"]) == [25, 30]

    def test_filter_column_not_found(self):
        """Test that filtering on non-existent column raises error"""
        df = pd.DataFrame({"age": [20, 25, 30]})
        with pytest.raises((ValueError, KeyError)):
            ar.filter_rows(ar.from_pandas(df), column="nonexistent", op=">", value=25)

    def test_filter_invalid_operator(self):
        """Test that invalid operator raises error"""
        df = pd.DataFrame({"age": [20, 25, 30]})
        with pytest.raises((ValueError, KeyError)):
            ar.filter_rows(ar.from_pandas(df), column="age", op=">>>", value=25)
