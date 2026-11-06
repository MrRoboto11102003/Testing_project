import pytest
import pandas as pd
import numpy as np
import arnio as ar
from decimal import Decimal


class TestToPandas:
    """Test suite for arnio.convert.to_pandas function."""

    def test_basic_int64_conversion(self):
        """Test basic conversion of int64 column to pandas."""
        frame = ar.from_pandas(pd.DataFrame({"id": [1, 2, 3]}))
        result = ar.to_pandas(frame)
        
        assert isinstance(result, pd.DataFrame)
        assert result["id"].tolist() == [1, 2, 3]
        assert result["id"].dtype == np.int64

    def test_basic_float64_conversion(self):
        """Test basic conversion of float64 column to pandas."""
        frame = ar.from_pandas(pd.DataFrame({"value": [1.5, 2.5, 3.5]}))
        result = ar.to_pandas(frame)
        
        assert isinstance(result, pd.DataFrame)
        assert result["value"].tolist() == [1.5, 2.5, 3.5]
        assert result["value"].dtype == np.float64

    def test_basic_string_conversion(self):
        """Test basic conversion of string column to pandas."""
        frame = ar.from_pandas(pd.DataFrame({"name": ["alice", "bob", "charlie"]}))
        result = ar.to_pandas(frame)
        
        assert isinstance(result, pd.DataFrame)
        assert result["name"].tolist() == ["alice", "bob", "charlie"]

    def test_basic_bool_conversion(self):
        """Test basic conversion of boolean column to pandas."""
        frame = ar.from_pandas(pd.DataFrame({"is_active": [True, False, True]}))
        result = ar.to_pandas(frame)
        
        assert isinstance(result, pd.DataFrame)
        assert result["is_active"].tolist() == [True, False, True]

    def test_conversion_with_int64_nulls(self):
        """Test conversion preserves Int64 nulls as pd.NA."""
        df = pd.DataFrame({"id": pd.array([1, 2, None], dtype="Int64")})
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert isinstance(result, pd.DataFrame)
        assert result["id"].dtype == pd.Int64Dtype()
        assert pd.isna(result["id"].iloc[2])

    def test_conversion_with_float64_nans(self):
        """Test conversion of float64 columns with NaN values."""
        frame = ar.from_pandas(pd.DataFrame({"value": [1.5, np.nan, 3.5]}))
        result = ar.to_pandas(frame)
        
        assert isinstance(result, pd.DataFrame)
        assert result["value"].dtype == np.float64
        assert pd.isna(result["value"].iloc[1])

    def test_conversion_with_string_nulls(self):
        """Test conversion preserves string nulls as pd.NA."""
        df = pd.DataFrame({"name": pd.array(["alice", None, "charlie"], dtype="string")})
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert isinstance(result, pd.DataFrame)
        assert result["name"].dtype == pd.StringDtype()
        assert pd.isna(result["name"].iloc[1])

    def test_empty_frame_conversion(self):
        """Test conversion of an empty ArFrame preserves structure."""
        df = pd.DataFrame({"id": pd.Series([], dtype="int64"), "name": pd.Series([], dtype="string")})
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert list(result.columns) == ["id", "name"]

    def test_single_row_conversion(self):
        """Test conversion of a single-row frame."""
        frame = ar.from_pandas(pd.DataFrame({"id": [1], "name": ["alice"]}))
        result = ar.to_pandas(frame)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result["id"].iloc[0] == 1
        assert result["name"].iloc[0] == "alice"

    def test_single_column_conversion(self):
        """Test conversion of a single-column frame."""
        frame = ar.from_pandas(pd.DataFrame({"id": [1, 2, 3]}))
        result = ar.to_pandas(frame)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) == 1
        assert result["id"].tolist() == [1, 2, 3]

    def test_copy_parameter_default_false(self):
        """Test that copy=False is default (zero-copy for numeric columns)."""
        frame = ar.from_pandas(pd.DataFrame({"value": [1.0, 2.0, 3.0]}))
        result_default = ar.to_pandas(frame)
        
        assert isinstance(result_default, pd.DataFrame)
        assert result_default["value"].dtype == np.float64
        assert result_default["value"].tolist() == [1.0, 2.0, 3.0]

    def test_copy_parameter_true_creates_buffers(self):
        """Test that copy=True creates defensive pandas-owned buffers."""
        frame = ar.from_pandas(pd.DataFrame({"value": [1.0, 2.0, 3.0]}))
        result = ar.to_pandas(frame, copy=True)
        
        assert isinstance(result, pd.DataFrame)
        assert result["value"].dtype == np.float64
        assert result["value"].tolist() == [1.0, 2.0, 3.0]

    def test_mixed_types_conversion(self):
        """Test conversion of frame with multiple data types."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "value": [1.5, 2.5, 3.5],
            "name": ["alice", "bob", "charlie"],
            "is_active": [True, False, True],
        })
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) == 4
        assert result["id"].dtype == np.int64
        assert result["value"].dtype == np.float64
        assert result["is_active"].dtype == bool

    def test_decimal_preserved_as_string(self):
        """Test that Decimal objects are preserved as strings to avoid precision loss."""
        df = pd.DataFrame({"price": [Decimal("19.99"), Decimal("29.95")]})
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert isinstance(result, pd.DataFrame)
        assert result["price"].dtype == object
        assert result["price"].tolist() == ["19.99", "29.95"]

    def test_column_order_preserved(self):
        """Test that column order is preserved during conversion."""
        df = pd.DataFrame({
            "z_col": [1, 2],
            "a_col": [3, 4],
            "m_col": [5, 6],
        })
        frame = ar.from_pandas(df)
        result = ar.to_pandas(frame)
        
        assert list(result.columns) == ["z_col", "a_col", "m_col"]

    def test_returns_pandas_dataframe(self):
        """Test that to_pandas returns a valid pandas DataFrame instance."""
        frame = ar.from_pandas(pd.DataFrame({"id": [1, 2, 3]}))
        result = ar.to_pandas(frame)
        
        assert isinstance(result, pd.DataFrame)
        assert hasattr(result, "columns")
        assert hasattr(result, "iloc")
        assert hasattr(result, "loc")
