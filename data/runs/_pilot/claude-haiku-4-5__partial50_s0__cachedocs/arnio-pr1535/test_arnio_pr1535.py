import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestFilterRowsAllOperators:
    """Test all six supported comparison operators."""
    
    def test_filter_rows_greater_than(self):
        """Test > operator with numeric values."""
        df = pd.DataFrame({"age": [10, 20, 30, 40, 50]})
        frame = ar.from_pandas(df)
        result = ar.filter_rows(frame, column="age", op=">", value=25)
        result_df = ar.to_pandas(result)
        assert list(result_df["age"]) == [30, 40, 50]
    
    def test_filter_rows_less_than(self):
        """Test < operator with numeric values."""
        df = pd.DataFrame({"age": [10, 20, 30, 40, 50]})
        frame = ar.from_pandas(df)
        result = ar.filter_rows(frame, column="age", op="<", value=25)
        result_df = ar.to_pandas(result)
        assert list(result_df["age"]) == [10, 20]
    
    def test_filter_rows_greater_equal(self):
        """Test >= operator includes boundary."""
        df = pd.DataFrame({"score": [10, 20, 30, 40, 50]})
        frame = ar.from_pandas(df)
        result = ar.filter_rows(frame, column="score", op=">=", value=30)
        result_df = ar.to_pandas(result)
        assert list(result_df["score"]) == [30, 40, 50]
    
    def test_filter_rows_less_equal(self):
        """Test <= operator includes boundary."""
        df = pd.DataFrame({"score": [10, 20, 30, 40, 50]})
        frame = ar.from_pandas(df)
        result = ar.filter_rows(frame, column="score", op="<=", value=30)
        result_df = ar.to_pandas(result)
        assert list(result_df["score"]) == [10, 20, 30]
    
    def test_filter_rows_equal(self):
        """Test == operator with strings."""
        df = pd.DataFrame({"status": ["active", "inactive", "active", "pending"]})
        frame = ar.from_pandas(df)
        result = ar.filter_rows(frame, column="status", op="==", value="active")
        result_df = ar.to_pandas(result)
        assert list(result_df["status"]) == ["active", "active"]
    
    def test_filter_rows_not_equal(self):
        """Test != operator."""
        df = pd.DataFrame({"status": ["active", "inactive", "active", "pending"]})
        frame = ar.from_pandas(df)
        result = ar.filter_rows(frame, column="status", op="!=", value="inactive")
        result_df = ar.to_pandas(result)
        assert list(result_df["status"]) == ["active", "active", "pending"]


class TestFilterRowsDataTypes:
    """Test filtering with different data types."""
    
    def test_filter_rows_with_floats(self):
        """Test filtering float columns."""
        df = pd.DataFrame({"price": [9.99, 19.99, 29.99, 39.99]})
        frame = ar.from_pandas(df)
        result = ar.filter_rows(frame, column="price", op=">", value=20.0)
        result_df = ar.to_pandas(result)
        assert len(result_df) == 2
        assert all(result_df["price"] > 20.0)
    
    def test_filter_rows_with_booleans(self):
        """Test filtering boolean columns."""
        df = pd.DataFrame({"is_active": [True, False, True, False, True]})
        frame = ar.from_pandas(df)
        result = ar.filter_rows(frame, column="is_active", op="==", value=True)
        result_df = ar.to_pandas(result)
        assert len(result_df) == 3
        assert all(result_df["is_active"])


class TestFilterRowsEdgeCases:
    """Test edge cases: all match, no match, empty input."""
    
    def test_filter_rows_all_match(self):
        """Test when all rows match the condition."""
        df = pd.DataFrame({"value": [1, 2, 3, 4, 5]})
        frame = ar.from_pandas(df)
        result = ar.filter_rows(frame, column="value", op=">", value=0)
        result_df = ar.to_pandas(result)
        assert len(result_df) == 5
    
    def test_filter_rows_no_match(self):
        """Test when no rows match the condition."""
        df = pd.DataFrame({"value": [1, 2, 3, 4, 5]})
        frame = ar.from_pandas(df)
        result = ar.filter_rows(frame, column="value", op=">", value=100)
        result_df = ar.to_pandas(result)
        assert len(result_df) == 0
    
    def test_filter_rows_empty_dataframe(self):
        """Test filtering on an empty DataFrame."""
        df = pd.DataFrame({"value": pd.Series([], dtype="int64")})
        frame = ar.from_pandas(df)
        result = ar.filter_rows(frame, column="value", op=">", value=10)
        result_df = ar.to_pandas(result)
        assert len(result_df) == 0
    
    def test_filter_rows_invalid_column(self):
        """Test that filtering on non-existent column raises error."""
        df = pd.DataFrame({"value": [1, 2, 3]})
        frame = ar.from_pandas(df)
        with pytest.raises((ValueError, KeyError)):
            ar.filter_rows(frame, column="nonexistent", op=">", value=10)


class TestFilterRowsReturnTypes:
    """Test that return type matches input type."""
    
    def test_filter_rows_returns_arframe(self):
        """Test that filtering an ArFrame returns an ArFrame."""
        df = pd.DataFrame({"age": [10, 20, 30, 40, 50]})
        frame = ar.from_pandas(df)
        result = ar.filter_rows(frame, column="age", op=">", value=25)
        assert isinstance(result, ar.ArFrame)
    
    def test_filter_rows_returns_dataframe(self):
        """Test that filtering a DataFrame returns a DataFrame."""
        df = pd.DataFrame({"age": [10, 20, 30, 40, 50]})
        result = ar.filter_rows(df, column="age", op=">", value=25)
        assert isinstance(result, pd.DataFrame)
    
    def test_filter_rows_preserves_column_order(self):
        """Test that column order is preserved after filtering."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        frame = ar.from_pandas(df)
        result = ar.filter_rows(frame, column="b", op=">", value=4)
        result_df = ar.to_pandas(result)
        assert list(result_df.columns) == ["a", "b", "c"]
    
    def test_filter_rows_in_pipeline(self):
        """Test using filter_rows within an ar.pipeline call."""
        df = pd.DataFrame({
            "revenue": [100, 999, 50, 10000, 500],
            "status": ["active", "active", "inactive", "active", "inactive"]
        })
        frame = ar.from_pandas(df)
        clean = ar.pipeline(frame, [
            ("filter_rows", {"column": "revenue", "op": ">=", "value": 100}),
        ])
        result_df = ar.to_pandas(clean)
        assert all(result_df["revenue"] >= 100)
        assert len(result_df) == 4
