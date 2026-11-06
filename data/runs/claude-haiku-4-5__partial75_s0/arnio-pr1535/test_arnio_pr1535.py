import pytest
import pandas as pd
import numpy as np
import arnio as ar


def make_frame(data: dict) -> ar.ArFrame:
    """Helper: Create an ArFrame from a dictionary."""
    df = pd.DataFrame(data)
    return ar.from_pandas(df)


def to_list(frame: ar.ArFrame, column: str) -> list:
    """Helper: Extract a column from an ArFrame as a list."""
    df = ar.to_pandas(frame)
    return df[column].tolist()


class TestFilterRowsBasicOperators:
    """Test all supported comparison operators."""

    def test_filter_rows_greater_than(self):
        """Test > operator with integers."""
        frame = make_frame({"age": [10, 20, 30, 40]})
        result = ar.filter_rows(frame, column="age", op=">", value=25)
        assert to_list(result, "age") == [30, 40]

    def test_filter_rows_less_than(self):
        """Test < operator with floats."""
        frame = make_frame({"price": [10.5, 20.3, 30.1, 40.9]})
        result = ar.filter_rows(frame, column="price", op="<", value=30.0)
        assert to_list(result, "price") == [10.5, 20.3]

    def test_filter_rows_greater_equal(self):
        """Test >= operator."""
        frame = make_frame({"score": [50, 60, 70, 80, 90]})
        result = ar.filter_rows(frame, column="score", op=">=", value=70)
        assert to_list(result, "score") == [70, 80, 90]

    def test_filter_rows_less_equal(self):
        """Test <= operator."""
        frame = make_frame({"value": [1, 2, 3, 4, 5]})
        result = ar.filter_rows(frame, column="value", op="<=", value=3)
        assert to_list(result, "value") == [1, 2, 3]

    def test_filter_rows_equal(self):
        """Test == operator with strings."""
        frame = make_frame({"status": ["active", "inactive", "active", "pending"]})
        result = ar.filter_rows(frame, column="status", op="==", value="active")
        assert to_list(result, "status") == ["active", "active"]

    def test_filter_rows_not_equal(self):
        """Test != operator."""
        frame = make_frame({"flag": [1, 0, 1, 0, 1]})
        result = ar.filter_rows(frame, column="flag", op="!=", value=0)
        assert to_list(result, "flag") == [1, 1, 1]


class TestFilterRowsDataTypes:
    """Test filter_rows with various pandas/numpy dtypes."""

    def test_filter_rows_with_strings(self):
        """Filter string column by exact match."""
        frame = make_frame({"name": ["alice", "bob", "charlie", "alice"]})
        result = ar.filter_rows(frame, column="name", op="==", value="alice")
        assert len(ar.to_pandas(result)) == 2
        assert to_list(result, "name") == ["alice", "alice"]

    def test_filter_rows_with_booleans(self):
        """Filter boolean column."""
        frame = make_frame({"is_active": [True, False, True, False]})
        result = ar.filter_rows(frame, column="is_active", op="==", value=True)
        assert len(ar.to_pandas(result)) == 2

    def test_filter_rows_float_comparison(self):
        """Filter float column with numeric comparison."""
        frame = make_frame({"temperature": [98.6, 101.5, 99.2, 100.1]})
        result = ar.filter_rows(frame, column="temperature", op=">", value=100.0)
        assert sorted(to_list(result, "temperature")) == [100.1, 101.5]


class TestFilterRowsReturnTypes:
    """Test that filter_rows preserves input frame type."""

    def test_returns_arframe_when_input_arframe(self):
        """ArFrame input produces ArFrame output."""
        frame = make_frame({"x": [1, 2, 3, 4]})
        result = ar.filter_rows(frame, column="x", op=">", value=2)
        assert isinstance(result, ar.ArFrame)

    def test_returns_dataframe_when_input_dataframe(self):
        """pd.DataFrame input produces pd.DataFrame output."""
        df = pd.DataFrame({"x": [1, 2, 3, 4]})
        result = ar.filter_rows(df, column="x", op=">", value=2)
        assert isinstance(result, pd.DataFrame)
        assert to_list(ar.from_pandas(result), "x") == [3, 4]


class TestFilterRowsEdgeCases:
    """Test boundary conditions and special cases."""

    def test_filter_no_rows_match(self):
        """Filter that matches no rows returns empty frame."""
        frame = make_frame({"age": [10, 20, 30]})
        result = ar.filter_rows(frame, column="age", op=">", value=100)
        assert len(ar.to_pandas(result)) == 0

    def test_filter_all_rows_match(self):
        """Filter where all rows match returns all rows."""
        frame = make_frame({"value": [1, 2, 3, 4, 5]})
        result = ar.filter_rows(frame, column="value", op=">", value=0)
        assert len(ar.to_pandas(result)) == 5

    def test_filter_single_row_frame_match(self):
        """Filter on single-row frame that matches condition."""
        frame = make_frame({"id": [42]})
        result = ar.filter_rows(frame, column="id", op="==", value=42)
        assert len(ar.to_pandas(result)) == 1

    def test_filter_single_row_frame_no_match(self):
        """Filter on single-row frame that doesn't match."""
        frame = make_frame({"id": [42]})
        result = ar.filter_rows(frame, column="id", op="==", value=99)
        assert len(ar.to_pandas(result)) == 0

    def test_filter_preserves_other_columns(self):
        """Filtering preserves all columns in result."""
        frame = make_frame({"id": [1, 2, 3], "name": ["a", "b", "c"], "value": [10, 20, 30]})
        result = ar.filter_rows(frame, column="id", op=">", value=1)
        df = ar.to_pandas(result)
        assert list(df.columns) == ["id", "name", "value"]
        assert len(df) == 2


class TestFilterRowsErrorHandling:
    """Test error cases and invalid inputs."""

    def test_nonexistent_column_raises_error(self):
        """Filtering on non-existent column raises error."""
        frame = make_frame({"age": [10, 20, 30]})
        with pytest.raises((KeyError, ValueError)):
            ar.filter_rows(frame, column="nonexistent", op=">", value=18)

    def test_invalid_operator_raises_error(self):
        """Invalid operator should raise error."""
        frame = make_frame({"age": [10, 20, 30]})
        with pytest.raises((ValueError, KeyError)):
            ar.filter_rows(frame, column="age", op="<>", value=20)
