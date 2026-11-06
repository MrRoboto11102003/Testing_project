import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestFilterRows:
    """Test suite for arnio.cleaning.filter_rows"""

    def test_filter_greater_than_arframe_returns_arframe(self):
        """Test > operator with ArFrame returns ArFrame"""
        df = pd.DataFrame({"age": [10, 20, 30, 40, 50]})
        frame = ar.from_pandas(df)
        
        result = ar.filter_rows(frame, column="age", op=">", value=25)
        result_df = ar.to_pandas(result)
        
        assert isinstance(result, ar.ArFrame)
        assert result_df["age"].tolist() == [30, 40, 50]

    def test_filter_pandas_dataframe_returns_dataframe(self):
        """Test that pd.DataFrame input returns pd.DataFrame"""
        df = pd.DataFrame({"value": [10, 20, 30]})
        result = ar.filter_rows(df, column="value", op=">", value=15)
        
        assert isinstance(result, pd.DataFrame)
        assert not isinstance(result, ar.ArFrame)
        assert result["value"].tolist() == [20, 30]

    def test_filter_all_supported_operators(self):
        """Test all supported operators: >, <, >=, <=, ==, !="""
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        frame = ar.from_pandas(df)
        
        ops_expected = {
            ">": [3, 4, 5],
            "<": [1, 2],
            ">=": [3, 4, 5],
            "<=": [1, 2, 3],
            "==": [3],
            "!=": [1, 2, 4, 5],
        }
        
        for op, expected in ops_expected.items():
            result = ar.filter_rows(frame, column="x", op=op, value=3)
            result_df = ar.to_pandas(result)
            assert result_df["x"].tolist() == expected, f"Failed for operator {op}"

    def test_filter_string_column_with_equality(self):
        """Test filtering on string column with == operator"""
        df = pd.DataFrame({"status": ["active", "inactive", "active", "pending"]})
        frame = ar.from_pandas(df)
        
        result = ar.filter_rows(frame, column="status", op="==", value="active")
        result_df = ar.to_pandas(result)
        
        assert len(result_df) == 2
        assert all(result_df["status"] == "active")

    def test_filter_boolean_column(self):
        """Test filtering on boolean column"""
        df = pd.DataFrame({"flag": [True, False, True, False, True]})
        frame = ar.from_pandas(df)
        
        result = ar.filter_rows(frame, column="flag", op="!=", value=True)
        result_df = ar.to_pandas(result)
        
        assert len(result_df) == 2
        assert all(~result_df["flag"])

    def test_filter_float_values(self):
        """Test filtering on float column with decimal precision"""
        df = pd.DataFrame({"price": [9.99, 15.50, 20.75, 5.25]})
        frame = ar.from_pandas(df)
        
        result = ar.filter_rows(frame, column="price", op=">=", value=10.0)
        result_df = ar.to_pandas(result)
        
        assert len(result_df) == 3
        assert result_df["price"].tolist() == [15.50, 20.75]

    def test_filter_with_nan_values_excluded(self):
        """Test that NaN values are excluded from filtered results"""
        df = pd.DataFrame({"val": [10.0, np.nan, 30.0, np.nan, 50.0]})
        frame = ar.from_pandas(df)
        
        result = ar.filter_rows(frame, column="val", op=">", value=20)
        result_df = ar.to_pandas(result)
        
        assert len(result_df) == 2
        assert result_df["val"].tolist() == [30.0, 50.0]

    def test_filter_empty_result_set(self):
        """Test when filter matches no rows"""
        df = pd.DataFrame({"age": [10, 20, 30]})
        frame = ar.from_pandas(df)
        
        result = ar.filter_rows(frame, column="age", op=">", value=100)
        result_df = ar.to_pandas(result)
        
        assert len(result_df) == 0
        assert list(result_df.columns) == ["age"]

    def test_filter_all_rows_match(self):
        """Test when all rows match the filter"""
        df = pd.DataFrame({"age": [50, 60, 70]})
        frame = ar.from_pandas(df)
        
        result = ar.filter_rows(frame, column="age", op=">", value=10)
        result_df = ar.to_pandas(result)
        
        assert len(result_df) == 3

    def test_error_column_not_found(self):
        """Test ValueError when column doesn't exist"""
        df = pd.DataFrame({"age": [10, 20, 30]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(ValueError, match="Unknown column"):
            ar.filter_rows(frame, column="nonexistent", op=">", value=15)

    def test_error_unsupported_operator(self):
        """Test ValueError for unsupported operator"""
        df = pd.DataFrame({"age": [10, 20, 30]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(ValueError, match="Unsupported operator"):
            ar.filter_rows(frame, column="age", op=">>", value=15)

    def test_error_column_not_string(self):
        """Test TypeError when column parameter is not a string"""
        df = pd.DataFrame({"age": [10, 20, 30]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(TypeError, match="column must be a non-empty string"):
            ar.filter_rows(frame, column=123, op=">", value=15)

    def test_error_op_not_string(self):
        """Test TypeError when op parameter is not a string"""
        df = pd.DataFrame({"age": [10, 20, 30]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(TypeError, match="op must be a string"):
            ar.filter_rows(frame, column="age", op=5, value=15)

    def test_error_value_not_scalar(self):
        """Test TypeError when value is not a scalar"""
        df = pd.DataFrame({"age": [10, 20, 30]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(TypeError, match="value must be a scalar"):
            ar.filter_rows(frame, column="age", op=">", value=[10, 20])

    def test_error_type_mismatch(self):
        """Test TypeError when comparing incompatible types"""
        df = pd.DataFrame({"name": ["Alice", "Bob", "Charlie"]})
        frame = ar.from_pandas(df)
        
        with pytest.raises(TypeError, match="cannot compare"):
            ar.filter_rows(frame, column="name", op=">", value=10)
