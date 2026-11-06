import pytest
import pandas as pd
import numpy as np
import arnio as ar
from arnio.cleaning.filter_rows import filter_rows


class TestFilterRowsBasicOperators:
    """Test filter_rows with all supported comparison operators."""
    
    def test_greater_than(self):
        """Test > operator."""
        df = pd.DataFrame({"age": [10, 20, 30, 40]})
        result = filter_rows(df, column="age", op=">", value=25)
        assert len(result) == 2
        assert list(result["age"].values) == [30, 40]
    
    def test_less_than(self):
        """Test < operator."""
        df = pd.DataFrame({"age": [10, 20, 30, 40]})
        result = filter_rows(df, column="age", op="<", value=25)
        assert len(result) == 2
        assert list(result["age"].values) == [10, 20]
    
    def test_greater_equal(self):
        """Test >= operator."""
        df = pd.DataFrame({"value": [1, 2, 3, 4]})
        result = filter_rows(df, column="value", op=">=", value=3)
        assert len(result) == 2
        assert list(result["value"].values) == [3, 4]
    
    def test_less_equal(self):
        """Test <= operator."""
        df = pd.DataFrame({"value": [1, 2, 3, 4]})
        result = filter_rows(df, column="value", op="<=", value=2)
        assert len(result) == 2
        assert list(result["value"].values) == [1, 2]
    
    def test_equal(self):
        """Test == operator with strings."""
        df = pd.DataFrame({"category": ["A", "B", "A", "C"]})
        result = filter_rows(df, column="category", op="==", value="A")
        assert len(result) == 2
        assert list(result["category"].values) == ["A", "A"]
    
    def test_not_equal(self):
        """Test != operator."""
        df = pd.DataFrame({"status": ["active", "inactive", "active"]})
        result = filter_rows(df, column="status", op="!=", value="inactive")
        assert len(result) == 2
        assert list(result["status"].values) == ["active", "active"]


class TestFilterRowsReturnTypes:
    """Test that return type matches input type."""
    
    def test_pandas_dataframe_input_returns_dataframe(self):
        """pd.DataFrame input should return pd.DataFrame."""
        df = pd.DataFrame({"x": [1, 2, 3, 4]})
        result = filter_rows(df, column="x", op=">", value=2)
        assert isinstance(result, pd.DataFrame)
        assert not isinstance(result, ar.ArFrame)
    
    def test_arframe_input_returns_arframe(self):
        """ArFrame input should return ArFrame."""
        frame = ar.from_dict({"x": [1, 2, 3, 4]})
        result = filter_rows(frame, column="x", op=">", value=2)
        assert isinstance(result, ar.ArFrame)


class TestFilterRowsEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_no_matching_rows(self):
        """When no rows match, return empty frame with same columns."""
        df = pd.DataFrame({"value": [1, 2, 3]})
        result = filter_rows(df, column="value", op=">", value=100)
        assert len(result) == 0
        assert list(result.columns) == ["value"]
    
    def test_all_rows_match(self):
        """When all rows match, return identical data."""
        df = pd.DataFrame({"value": [1, 2, 3]})
        result = filter_rows(df, column="value", op=">", value=0)
        assert len(result) == 3
        assert list(result["value"].values) == [1, 2, 3]
    
    def test_nan_values_treated_as_false(self):
        """NaN values should be treated as not matching (fillna(False))."""
        df = pd.DataFrame({"value": [1.0, np.nan, 3.0, np.nan, 5.0]})
        result = filter_rows(df, column="value", op=">", value=2.0)
        assert len(result) == 2
        assert list(result["value"].values) == [3.0, 5.0]
    
    def test_all_nan_column(self):
        """Column with all NaN values should return empty result."""
        df = pd.DataFrame({"value": [np.nan, np.nan, np.nan]})
        result = filter_rows(df, column="value", op=">", value=0)
        assert len(result) == 0
    
    def test_multi_column_preserves_all_columns(self):
        """Filtering one column should preserve all columns."""
        df = pd.DataFrame({
            "age": [10, 20, 30, 40],
            "name": ["Alice", "Bob", "Charlie", "David"]
        })
        result = filter_rows(df, column="age", op=">", value=25)
        assert list(result.columns) == ["age", "name"]
        assert len(result) == 2
        assert list(result["name"].values) == ["Charlie", "David"]
    
    def test_float_precision(self):
        """Test filtering with float values."""
        df = pd.DataFrame({"price": [9.99, 19.99, 29.99]})
        result = filter_rows(df, column="price", op=">=", value=19.99)
        assert len(result) == 2


class TestFilterRowsValidation:
    """Test input validation and error handling."""
    
    def test_column_not_found(self):
        """Unknown column should raise ValueError."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(ValueError, match="Unknown column"):
            filter_rows(df, column="nonexistent", op=">", value=1)
    
    def test_invalid_operator(self):
        """Unsupported operator should raise ValueError."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(ValueError, match="Unsupported operator"):
            filter_rows(df, column="x", op=">>", value=1)
    
    def test_non_string_column_name(self):
        """Non-string column should raise TypeError."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(TypeError, match="column must be a non-empty string"):
            filter_rows(df, column=0, op=">", value=1)
    
    def test_empty_string_column_name(self):
        """Empty string column should raise TypeError."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(TypeError, match="column must be a non-empty string"):
            filter_rows(df, column="", op=">", value=1)
    
    def test_non_string_operator(self):
        """Non-string operator should raise TypeError."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(TypeError, match="op must be a string"):
            filter_rows(df, column="x", op=1, value=1)
    
    def test_non_scalar_value(self):
        """Non-scalar value should raise TypeError."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(TypeError, match="value must be a scalar"):
            filter_rows(df, column="x", op=">", value=[1, 2])
    
    def test_type_mismatch_in_comparison(self):
        """Incompatible types should raise TypeError."""
        df = pd.DataFrame({"name": ["Alice", "Bob"]})
        with pytest.raises(TypeError, match="cannot compare column"):
            filter_rows(df, column="name", op=">", value=10)
