import pytest
import pandas as pd
import arnio as ar


class TestFilterRowsOperators:
    """Test filter_rows with different comparison operators."""

    def test_filter_greater_than(self):
        """Test > operator filters correctly."""
        df = pd.DataFrame({"age": [10, 20, 30, 40, 50]})
        frame = ar.from_pandas(df)
        result = ar.filter_rows(frame, column="age", op=">", value=25)
        assert ar.to_pandas(result)["age"].tolist() == [30, 40, 50]

    def test_filter_less_than(self):
        """Test < operator filters correctly."""
        df = pd.DataFrame({"age": [10, 20, 30, 40, 50]})
        frame = ar.from_pandas(df)
        result = ar.filter_rows(frame, column="age", op="<", value=30)
        assert ar.to_pandas(result)["age"].tolist() == [10, 20]

    def test_filter_greater_equal(self):
        """Test >= operator includes boundary value."""
        df = pd.DataFrame({"age": [10, 20, 30, 40, 50]})
        frame = ar.from_pandas(df)
        result = ar.filter_rows(frame, column="age", op=">=", value=30)
        assert ar.to_pandas(result)["age"].tolist() == [30, 40, 50]

    def test_filter_less_equal(self):
        """Test <= operator includes boundary value."""
        df = pd.DataFrame({"age": [10, 20, 30, 40, 50]})
        frame = ar.from_pandas(df)
        result = ar.filter_rows(frame, column="age", op="<=", value=30)
        assert ar.to_pandas(result)["age"].tolist() == [10, 20, 30]

    def test_filter_equal(self):
        """Test == operator matches exact values."""
        df = pd.DataFrame({"age": [10, 20, 30, 40, 30]})
        frame = ar.from_pandas(df)
        result = ar.filter_rows(frame, column="age", op="==", value=30)
        assert ar.to_pandas(result)["age"].tolist() == [30, 30]

    def test_filter_not_equal(self):
        """Test != operator excludes matching values."""
        df = pd.DataFrame({"age": [10, 20, 30, 40, 30]})
        frame = ar.from_pandas(df)
        result = ar.filter_rows(frame, column="age", op="!=", value=30)
        assert ar.to_pandas(result)["age"].tolist() == [10, 20, 40]


class TestFilterRowsDataTypes:
    """Test filter_rows with different column data types."""

    def test_filter_floats(self):
        """Test filtering works on float columns."""
        df = pd.DataFrame({"price": [10.5, 20.3, 30.1, 40.9]})
        frame = ar.from_pandas(df)
        result = ar.filter_rows(frame, column="price", op=">", value=25.0)
        result_df = ar.to_pandas(result)
        assert len(result_df) == 2
        assert all(result_df["price"] > 25.0)

    def test_filter_strings(self):
        """Test filtering works on string columns."""
        df = pd.DataFrame({"city": ["NYC", "LA", "NYC", "Chicago"]})
        frame = ar.from_pandas(df)
        result = ar.filter_rows(frame, column="city", op="==", value="NYC")
        assert ar.to_pandas(result)["city"].tolist() == ["NYC", "NYC"]

    def test_filter_booleans(self):
        """Test filtering works on boolean columns."""
        df = pd.DataFrame({"active": [True, False, True, False]})
        frame = ar.from_pandas(df)
        result = ar.filter_rows(frame, column="active", op="==", value=True)
        assert ar.to_pandas(result)["active"].tolist() == [True, True]


class TestFilterRowsInputOutput:
    """Test input type preservation and output correctness."""

    def test_arframe_input_returns_arframe(self):
        """Test that ArFrame input returns ArFrame output."""
        df = pd.DataFrame({"age": [10, 20, 30]})
        frame = ar.from_pandas(df)
        result = ar.filter_rows(frame, column="age", op=">", value=15)
        assert isinstance(result, ar.ArFrame)

    def test_dataframe_input_returns_dataframe(self):
        """Test that DataFrame input returns DataFrame output."""
        df = pd.DataFrame({"age": [10, 20, 30]})
        result = ar.filter_rows(df, column="age", op=">", value=15)
        assert isinstance(result, pd.DataFrame)

    def test_preserves_column_order_and_names(self):
        """Test that filtering preserves all columns in original order."""
        df = pd.DataFrame({
            "age": [10, 20, 30, 40],
            "name": ["Alice", "Bob", "Charlie", "David"],
            "score": [100, 200, 300, 400],
        })
        frame = ar.from_pandas(df)
        result = ar.filter_rows(frame, column="age", op=">", value=15)
        result_df = ar.to_pandas(result)
        assert list(result_df.columns) == ["age", "name", "score"]
        assert result_df["name"].tolist() == ["Bob", "Charlie", "David"]


class TestFilterRowsEdgeCases:
    """Test edge cases and special conditions."""

    def test_empty_result_when_no_rows_match(self):
        """Test filtering with no matching rows returns empty frame."""
        df = pd.DataFrame({"age": [10, 20, 30]})
        frame = ar.from_pandas(df)
        result = ar.filter_rows(frame, column="age", op=">", value=100)
        assert len(ar.to_pandas(result)) == 0

    def test_all_rows_match(self):
        """Test filtering where all rows match condition."""
        df = pd.DataFrame({"age": [10, 20, 30]})
        frame = ar.from_pandas(df)
        result = ar.filter_rows(frame, column="age", op=">", value=0)
        assert len(ar.to_pandas(result)) == 3

    def test_single_row_frame(self):
        """Test filtering on frame with single row."""
        df = pd.DataFrame({"age": [25]})
        frame = ar.from_pandas(df)
        result = ar.filter_rows(frame, column="age", op=">", value=20)
        assert len(ar.to_pandas(result)) == 1

    def test_nulls_excluded_from_results(self):
        """Test that null values are excluded from filtered results."""
        df = pd.DataFrame({"age": [10.0, None, 30.0, 40.0]})
        frame = ar.from_pandas(df)
        result = ar.filter_rows(frame, column="age", op=">", value=15)
        result_df = ar.to_pandas(result)
        assert not result_df["age"].isna().any()
        assert len(result_df) == 2


class TestFilterRowsErrors:
    """Test error handling for invalid inputs."""

    def test_nonexistent_column_raises_error(self):
        """Test that filtering on non-existent column raises error."""
        df = pd.DataFrame({"age": [10, 20, 30]})
        frame = ar.from_pandas(df)
        with pytest.raises((KeyError, ValueError)):
            ar.filter_rows(frame, column="nonexistent", op=">", value=15)

    def test_invalid_operator_raises_error(self):
        """Test that invalid operator raises error."""
        df = pd.DataFrame({"age": [10, 20, 30]})
        frame = ar.from_pandas(df)
        with pytest.raises((ValueError, Exception)):
            ar.filter_rows(frame, column="age", op=">>", value=15)
