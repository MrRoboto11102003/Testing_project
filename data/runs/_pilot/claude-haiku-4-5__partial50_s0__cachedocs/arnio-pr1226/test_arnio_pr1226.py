import pytest
import pandas as pd
import numpy as np
import arnio as ar
from decimal import Decimal


class TestToPandas:
    """Tests for arnio.convert.to_pandas function."""

    def test_basic_conversion_to_dataframe(self):
        """Test basic conversion returns a pandas DataFrame."""
        df = pd.DataFrame({"id": [1, 2, 3]})
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (3, 1)

    def test_integer_column_conversion(self):
        """Test integer column conversion preserves values."""
        df = pd.DataFrame({"id": [1, 2, 3]})
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert result["id"].tolist() == [1, 2, 3]

    def test_float_column_conversion(self):
        """Test float column conversion preserves values."""
        df = pd.DataFrame({"price": [10.5, 20.3, 30.1]})
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        np.testing.assert_array_almost_equal(
            result["price"].values, [10.5, 20.3, 30.1], decimal=5
        )

    def test_string_column_conversion(self):
        """Test string column conversion preserves values."""
        df = pd.DataFrame({"name": ["alice", "bob", "charlie"]})
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert result["name"].tolist() == ["alice", "bob", "charlie"]

    def test_multiple_columns_different_types(self):
        """Test conversion with multiple columns of different types."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["alice", "bob", "charlie"],
            "score": [95.5, 88.3, 92.1],
        })
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert result.shape == (3, 3)
        assert list(result.columns) == ["id", "name", "score"]
        assert result["id"].tolist() == [1, 2, 3]

    def test_null_values_in_int_column(self):
        """Test that int columns with nulls use nullable Int64 dtype."""
        df = pd.DataFrame({"value": pd.array([1, 2, None], dtype="Int64")})
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert pd.isna(result["value"].iloc[2])
        assert result["value"].iloc[0] == 1

    def test_null_values_in_float_column(self):
        """Test that float columns with nulls are properly handled."""
        df = pd.DataFrame({"value": [1.5, 2.5, np.nan]})
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert pd.isna(result["value"].iloc[2])
        assert result["value"].iloc[0] == 1.5

    def test_null_values_in_string_column(self):
        """Test that string columns with nulls use StringDtype."""
        df = pd.DataFrame({"name": pd.array(["alice", None], dtype="string")})
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert pd.isna(result["name"].iloc[1])
        assert result["name"].iloc[0] == "alice"

    def test_boolean_column_with_nulls(self):
        """Test that boolean columns with nulls use BooleanDtype."""
        df = pd.DataFrame({"active": pd.array([True, False, None], dtype="boolean")})
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert pd.isna(result["active"].iloc[2])
        assert result["active"].iloc[0] is True

    def test_copy_parameter_false_default(self):
        """Test that copy=False is the default (zero-copy for numerics)."""
        df = pd.DataFrame({"id": [1, 2, 3]})
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert isinstance(result, pd.DataFrame)
        assert result["id"].tolist() == [1, 2, 3]

    def test_copy_parameter_true_creates_defensive_buffers(self):
        """Test that copy=True creates pandas-owned defensive buffers."""
        df = pd.DataFrame({"id": [1, 2, 3]})
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame, copy=True)
        
        assert isinstance(result, pd.DataFrame)
        assert result["id"].tolist() == [1, 2, 3]

    def test_index_is_default_rangeindex(self):
        """Test that returned DataFrame has default RangeIndex (not preserved)."""
        df = pd.DataFrame(
            {"id": [1, 2, 3]},
            index=pd.Index([100, 200, 300], name="custom_idx")
        )
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        # Index should be a default RangeIndex, not the custom one
        assert isinstance(result.index, pd.RangeIndex)
        assert result.index.tolist() == [0, 1, 2]

    def test_empty_dataframe(self):
        """Test conversion of an empty DataFrame."""
        df = pd.DataFrame({"id": pd.Series([], dtype="int64")})
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert result.shape[0] == 0
        assert "id" in result.columns

    def test_decimal_values_preserved_as_strings(self):
        """Test that Decimal values are preserved as strings to prevent precision loss."""
        df = pd.DataFrame({
            "price": [Decimal("19.99"), Decimal("29.95"), Decimal("99.99")]
        })
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        # Decimal values should return as strings to preserve exact precision
        assert result["price"].tolist() == ["19.99", "29.95", "99.99"]

    def test_single_row_dataframe(self):
        """Test conversion of a single-row DataFrame."""
        df = pd.DataFrame({"id": [42], "name": ["alice"]})
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert result.shape == (1, 2)
        assert result["id"].iloc[0] == 42
        assert result["name"].iloc[0] == "alice"
