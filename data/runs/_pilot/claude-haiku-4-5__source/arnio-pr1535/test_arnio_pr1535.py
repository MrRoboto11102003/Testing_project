import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestFilterRows:
    """Test suite for arnio.filter_rows function."""

    def test_filter_greater_than(self):
        """Test filtering with > operator."""
        df = pd.DataFrame({"age": [15, 20, 25, 30]})
        result = ar.filter_rows(df, column="age", op=">", value=20)
        assert len(result) == 2
        assert list(result["age"].values) == [25, 30]

    def test_filter_less_than(self):
        """Test filtering with < operator."""
        df = pd.DataFrame({"value": [1, 5, 10, 15]})
        result = ar.filter_rows(df, column="value", op="<", value=10)
        assert len(result) == 2

    def test_filter_equal(self):
        """Test filtering with == operator."""
        df = pd.DataFrame({"category": ["A", "B", "A", "C"]})
        result = ar.filter_rows(df, column="category", op="==", value="A")
        assert len(result) == 2

    def test_filter_not_equal(self):
        """Test filtering with != operator."""
        df = pd.DataFrame({"status": ["on", "off", "on"]})
        result = ar.filter_rows(df, column="status", op="!=", value="off")
        assert len(result) == 2

    def test_filter_comparison_operators(self):
        """Test >= and <= operators."""
        df = pd.DataFrame({"score": [10, 20, 30, 40]})
        ge_result = ar.filter_rows(df, column="score", op=">=", value=30)
        le_result = ar.filter_rows(df, column="score", op="<=", value=20)
        assert len(ge_result) == 2 and len(le_result) == 2

    def test_returns_same_type(self):
        """Test that input type is preserved in output."""
        df = pd.DataFrame({"age": [20, 30]})
        assert isinstance(ar.filter_rows(df, column="age", op=">", value=25), pd.DataFrame)
        frame = ar.from_dict({"age": [20, 30, 40]})
        result = ar.filter_rows(frame, column="age", op=">", value=25)
        assert hasattr(result, 'preview')

    def test_no_matching_rows(self):
        """Test filtering when no rows match condition."""
        df = pd.DataFrame({"age": [10, 15, 20]})
        result = ar.filter_rows(df, column="age", op=">", value=100)
        assert len(result) == 0

    def test_all_rows_match(self):
        """Test filtering when all rows match condition."""
        df = pd.DataFrame({"age": [25, 30, 35]})
        result = ar.filter_rows(df, column="age", op=">", value=20)
        assert len(result) == 3

    def test_nan_values_excluded(self):
        """Test that NaN values are excluded from filtering results."""
        df = pd.DataFrame({"value": [1.0, np.nan, 3.0, np.nan, 5.0]})
        result = ar.filter_rows(df, column="value", op=">", value=2.0)
        assert len(result) == 2 and not result["value"].isna().any()

    def test_invalid_column_name(self):
        """Test that invalid column name raises TypeError."""
        df = pd.DataFrame({"age": [20, 30]})
        with pytest.raises(TypeError, match="column must be a non-empty string"):
            ar.filter_rows(df, column=123, op=">", value=25)
        with pytest.raises(TypeError, match="column must be a non-empty string"):
            ar.filter_rows(df, column="", op=">", value=25)

    def test_invalid_op_type(self):
        """Test that non-string operator raises TypeError."""
        df = pd.DataFrame({"age": [20, 30]})
        with pytest.raises(TypeError, match="op must be a string"):
            ar.filter_rows(df, column="age", op=123, value=25)

    def test_unsupported_operator(self):
        """Test that unsupported operator raises ValueError."""
        df = pd.DataFrame({"age": [20, 30]})
        with pytest.raises(ValueError, match="Unsupported operator"):
            ar.filter_rows(df, column="age", op=">>", value=25)

    def test_column_not_found(self):
        """Test that non-existent column raises ValueError."""
        df = pd.DataFrame({"age": [20, 30]})
        with pytest.raises(ValueError, match="Unknown column"):
            ar.filter_rows(df, column="nonexistent", op=">", value=25)

    def test_non_scalar_value(self):
        """Test that non-scalar value raises TypeError."""
        df = pd.DataFrame({"age": [20, 30]})
        with pytest.raises(TypeError, match="value must be a scalar"):
            ar.filter_rows(df, column="age", op=">", value=[20, 30])

    def test_incompatible_types(self):
        """Test that incompatible column and value types raise TypeError."""
        df = pd.DataFrame({"name": ["Alice", "Bob"]})
        with pytest.raises(TypeError, match="cannot compare column"):
            ar.filter_rows(df, column="name", op=">", value=100)
