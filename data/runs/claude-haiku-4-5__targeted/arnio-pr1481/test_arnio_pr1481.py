import pytest
import pandas as pd

try:
    import arnio as ar
except ImportError:
    pytest.skip("arnio not installed", allow_module_level=True)


class TestFilterRows:
    """Unit tests for arnio.cleaning.filter_rows function."""

    @pytest.fixture
    def sample_df(self):
        """Sample pandas DataFrame with numeric and string columns."""
        return pd.DataFrame({
            "age": [15, 18, 25, 30, 35, 40],
            "name": ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"],
            "salary": [30000.0, 35000.0, 50000.0, 60000.0, 75000.0, 80000.0],
        })

    @pytest.fixture
    def sample_frame(self):
        """Sample ArFrame for testing."""
        data = {
            "age": [15, 18, 25, 30, 35, 40],
            "name": ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"],
            "salary": [30000.0, 35000.0, 50000.0, 60000.0, 75000.0, 80000.0],
        }
        return ar.from_dict(data)

    def test_filter_greater_than_operator(self, sample_df):
        """Test > operator correctly filters rows."""
        result = ar.filter_rows(sample_df, column="age", op=">", value=25)
        assert len(result) == 3
        assert all(result["age"] > 25)

    def test_filter_less_than_operator(self, sample_df):
        """Test < operator correctly filters rows."""
        result = ar.filter_rows(sample_df, column="age", op="<", value=25)
        assert len(result) == 2
        assert all(result["age"] < 25)

    def test_filter_greater_equal_operator(self, sample_df):
        """Test >= operator correctly filters rows."""
        result = ar.filter_rows(sample_df, column="age", op=">=", value=30)
        assert len(result) == 4
        assert all(result["age"] >= 30)

    def test_filter_less_equal_operator(self, sample_df):
        """Test <= operator correctly filters rows."""
        result = ar.filter_rows(sample_df, column="age", op="<=", value=25)
        assert len(result) == 3
        assert all(result["age"] <= 25)

    def test_filter_equal_operator(self, sample_df):
        """Test == operator correctly filters rows."""
        result = ar.filter_rows(sample_df, column="age", op="==", value=30)
        assert len(result) == 1
        assert result.iloc[0]["age"] == 30

    def test_filter_not_equal_operator(self, sample_df):
        """Test != operator correctly filters rows."""
        result = ar.filter_rows(sample_df, column="age", op="!=", value=30)
        assert len(result) == 5
        assert all(result["age"] != 30)

    def test_pandas_dataframe_input_returns_dataframe(self, sample_df):
        """Test that pandas DataFrame input returns pandas DataFrame."""
        result = ar.filter_rows(sample_df, column="age", op=">", value=20)
        assert isinstance(result, pd.DataFrame)
        assert not isinstance(result, ar.ArFrame)

    def test_arframe_input_returns_arframe(self, sample_frame):
        """Test that ArFrame input returns ArFrame."""
        result = ar.filter_rows(sample_frame, column="age", op=">", value=20)
        assert isinstance(result, ar.ArFrame)

    def test_filter_no_matches_returns_empty_frame(self, sample_df):
        """Test filtering with no matches returns empty frame of same type."""
        result = ar.filter_rows(sample_df, column="age", op=">", value=100)
        assert len(result) == 0
        assert list(result.columns) == list(sample_df.columns)

    def test_filter_all_matches_returns_full_frame(self, sample_df):
        """Test filtering where all rows match returns complete frame."""
        result = ar.filter_rows(sample_df, column="age", op=">", value=10)
        assert len(result) == len(sample_df)

    def test_filter_string_column_equality(self, sample_df):
        """Test filtering on string column with equality operator."""
        result = ar.filter_rows(sample_df, column="name", op="==", value="Alice")
        assert len(result) == 1
        assert result.iloc[0]["name"] == "Alice"

    def test_filter_float_column_comparison(self, sample_df):
        """Test filtering on float column with comparison operator."""
        result = ar.filter_rows(sample_df, column="salary", op=">", value=50000.0)
        assert len(result) == 4
        assert all(result["salary"] > 50000.0)

    def test_filter_preserves_all_columns(self, sample_df):
        """Test that filtering preserves all columns from original frame."""
        result = ar.filter_rows(sample_df, column="age", op=">", value=20)
        assert list(result.columns) == list(sample_df.columns)

    def test_filter_preserves_data_types(self, sample_df):
        """Test that column data types are preserved after filtering."""
        result = ar.filter_rows(sample_df, column="age", op=">", value=20)
        assert result["age"].dtype == sample_df["age"].dtype
        assert result["name"].dtype == sample_df["name"].dtype
        assert result["salary"].dtype == sample_df["salary"].dtype
