import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestDropColumns:
    """Comprehensive tests for arnio.cleaning.drop_columns"""

    @pytest.fixture
    def sample_frame(self):
        """Create a sample ArFrame for testing."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["alice", "bob", "charlie"],
            "age": [25, 30, 35],
            "city": ["NYC", "LA", "SF"],
        })
        return ar.from_pandas(df)

    def test_drop_single_column(self, sample_frame):
        """Test removing a single column."""
        result = ar.drop_columns(sample_frame, ["name"])
        assert result.columns == ["id", "age", "city"]
        assert result.shape == (3, 3)

    def test_drop_multiple_columns(self, sample_frame):
        """Test removing multiple columns."""
        result = ar.drop_columns(sample_frame, ["name", "city"])
        assert result.columns == ["id", "age"]
        assert result.shape == (3, 2)

    def test_drop_columns_preserves_order(self, sample_frame):
        """Test that remaining columns maintain original order."""
        result = ar.drop_columns(sample_frame, ["id", "age"])
        assert result.columns == ["name", "city"]

    def test_drop_empty_list(self, sample_frame):
        """Test with empty columns list returns unchanged frame."""
        result = ar.drop_columns(sample_frame, [])
        assert result.columns == sample_frame.columns
        assert result.shape == sample_frame.shape

    def test_drop_all_columns_raises_error(self, sample_frame):
        """Test that removing all columns raises ValueError."""
        with pytest.raises(ValueError, match="cannot remove all columns"):
            ar.drop_columns(sample_frame, sample_frame.columns)

    def test_drop_nonexistent_column_raises_error(self, sample_frame):
        """Test that requesting non-existent column raises ValueError."""
        with pytest.raises(ValueError, match="Columns not found"):
            ar.drop_columns(sample_frame, ["nonexistent"])

    def test_drop_mixed_valid_and_invalid_columns_raises_error(self, sample_frame):
        """Test that mixing valid and invalid columns raises ValueError."""
        with pytest.raises(ValueError, match="Columns not found"):
            ar.drop_columns(sample_frame, ["name", "invalid_col"])

    def test_drop_columns_string_input_raises_typeerror(self, sample_frame):
        """Test that passing string instead of sequence raises TypeError."""
        with pytest.raises(TypeError):
            ar.drop_columns(sample_frame, "name")

    def test_drop_columns_non_string_items_raises_typeerror(self, sample_frame):
        """Test that non-string column names raise TypeError."""
        with pytest.raises(TypeError):
            ar.drop_columns(sample_frame, ["name", 123])

    def test_drop_columns_preserves_data(self, sample_frame):
        """Test that data values are preserved after dropping columns."""
        result = ar.drop_columns(sample_frame, ["city"])
        df = ar.to_pandas(result)
        assert df["name"].tolist() == ["alice", "bob", "charlie"]
        assert df["age"].tolist() == [25, 30, 35]

    def test_drop_columns_returns_new_frame(self, sample_frame):
        """Test that drop_columns returns new frame without modifying original."""
        original_columns = sample_frame.columns.copy()
        result = ar.drop_columns(sample_frame, ["name"])
        assert sample_frame.columns == original_columns
        assert result.columns != original_columns

    def test_drop_columns_case_sensitive(self, sample_frame):
        """Test that column matching is case-sensitive."""
        with pytest.raises(ValueError, match="Columns not found"):
            ar.drop_columns(sample_frame, ["Name"])  # uppercase N

    def test_drop_duplicate_column_requests(self, sample_frame):
        """Test requesting the same column multiple times."""
        result = ar.drop_columns(sample_frame, ["name", "name"])
        assert result.columns == ["id", "age", "city"]

    def test_drop_columns_numeric_columns(self):
        """Test drop_columns correctly handles numeric data."""
        df = pd.DataFrame({
            "x": [1.5, 2.5, 3.5],
            "y": [4.0, 5.0, 6.0],
            "z": [7, 8, 9],
        })
        frame = ar.from_pandas(df)
        result = ar.drop_columns(frame, ["y"])
        assert result.columns == ["x", "z"]
        result_df = ar.to_pandas(result)
        assert np.allclose(result_df["x"].values, [1.5, 2.5, 3.5])
        assert np.allclose(result_df["z"].values, [7, 8, 9])

    def test_drop_columns_with_nulls(self):
        """Test drop_columns preserves null values in remaining columns."""
        df = pd.DataFrame({
            "a": [1, 2, None],
            "b": [None, "x", "y"],
            "c": [10, 20, 30],
        })
        frame = ar.from_pandas(df)
        result = ar.drop_columns(frame, ["b"])
        result_df = ar.to_pandas(result)
        assert result_df.columns.tolist() == ["a", "c"]
        assert pd.isna(result_df.loc[2, "a"])
