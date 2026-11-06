import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestFilterRows:
    """Test suite for ar.filter_rows function."""

    @pytest.fixture
    def sample_frame(self):
        """Create a sample ArFrame for testing."""
        df = pd.DataFrame({
            "age": [20, 25, 30, 35, 40],
            "score": [85.5, 90.0, 88.2, 92.5, 87.3],
            "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        })
        return ar.from_pandas(df)

    @pytest.fixture
    def sample_df(self):
        """Create a sample pandas DataFrame for testing."""
        return pd.DataFrame({
            "age": [20, 25, 30, 35, 40],
            "score": [85.5, 90.0, 88.2, 92.5, 87.3],
            "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        })

    def test_filter_greater_than(self, sample_frame):
        """Test > operator filters correctly."""
        result = ar.filter_rows(sample_frame, "age", ">", 25)
        df = ar.to_pandas(result)
        assert len(df) == 3
        assert df["age"].tolist() == [30, 35, 40]

    def test_filter_less_than(self, sample_frame):
        """Test < operator filters correctly."""
        result = ar.filter_rows(sample_frame, "age", "<", 30)
        df = ar.to_pandas(result)
        assert len(df) == 2
        assert df["age"].tolist() == [20, 25]

    def test_filter_equal(self, sample_frame):
        """Test == operator filters correctly."""
        result = ar.filter_rows(sample_frame, "age", "==", 25)
        df = ar.to_pandas(result)
        assert len(df) == 1
        assert df["age"].iloc[0] == 25

    def test_filter_not_equal(self, sample_frame):
        """Test != operator filters correctly."""
        result = ar.filter_rows(sample_frame, "age", "!=", 25)
        df = ar.to_pandas(result)
        assert len(df) == 4
        assert 25 not in df["age"].tolist()

    def test_filter_greater_equal(self, sample_frame):
        """Test >= operator filters correctly."""
        result = ar.filter_rows(sample_frame, "age", ">=", 30)
        df = ar.to_pandas(result)
        assert len(df) == 3
        assert df["age"].tolist() == [30, 35, 40]

    def test_filter_less_equal(self, sample_frame):
        """Test <= operator filters correctly."""
        result = ar.filter_rows(sample_frame, "age", "<=", 25)
        df = ar.to_pandas(result)
        assert len(df) == 2
        assert df["age"].tolist() == [20, 25]

    def test_filter_returns_arframe_for_arframe_input(self, sample_frame):
        """Test that ArFrame input returns ArFrame."""
        result = ar.filter_rows(sample_frame, "age", ">", 25)
        assert isinstance(result, ar.ArFrame)

    def test_filter_returns_dataframe_for_dataframe_input(self, sample_df):
        """Test that DataFrame input returns DataFrame."""
        result = ar.filter_rows(sample_df, "age", ">", 25)
        assert isinstance(result, pd.DataFrame)

    def test_filter_invalid_operator_raises_error(self, sample_frame):
        """Test that unsupported operator raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported operator"):
            ar.filter_rows(sample_frame, "age", ">>", 25)

    def test_filter_nonexistent_column_raises_error(self, sample_frame):
        """Test that nonexistent column raises ValueError."""
        with pytest.raises(ValueError, match="Unknown column"):
            ar.filter_rows(sample_frame, "nonexistent", ">", 25)

    def test_filter_invalid_column_type_raises_error(self, sample_frame):
        """Test that non-string column raises TypeError."""
        with pytest.raises(TypeError, match="column must be a non-empty string"):
            ar.filter_rows(sample_frame, 123, ">", 25)

    def test_filter_invalid_operator_type_raises_error(self, sample_frame):
        """Test that non-string operator raises TypeError."""
        with pytest.raises(TypeError, match="op must be a string"):
            ar.filter_rows(sample_frame, "age", 123, 25)

    def test_filter_non_scalar_value_raises_error(self, sample_frame):
        """Test that non-scalar value raises TypeError."""
        with pytest.raises(TypeError, match="value must be a scalar"):
            ar.filter_rows(sample_frame, "age", ">", [25, 30])

    def test_filter_with_nulls_excluded(self):
        """Test that NaN values are treated as non-matching."""
        df = pd.DataFrame({
            "value": [10.0, np.nan, 30.0, 40.0]
        })
        frame = ar.from_pandas(df)
        result = ar.filter_rows(frame, "value", ">", 15)
        result_df = ar.to_pandas(result)
        assert len(result_df) == 2
        assert result_df["value"].tolist() == [30.0, 40.0]

    def test_filter_string_column_equality(self, sample_frame):
        """Test filtering on string column with == operator."""
        result = ar.filter_rows(sample_frame, "name", "==", "Bob")
        df = ar.to_pandas(result)
        assert len(df) == 1
        assert df["name"].iloc[0] == "Bob"
