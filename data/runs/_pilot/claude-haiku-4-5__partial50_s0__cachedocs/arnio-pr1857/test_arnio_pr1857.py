import pytest
import pandas as pd
import arnio as ar


class TestFromDict:
    """Test suite for arnio.from_dict function."""

    def test_basic_dict_creation(self):
        """Test creating an ArFrame from a basic dictionary."""
        data = {"name": ["Alice", "Bob"], "age": [25, 30]}
        frame = ar.from_dict(data)
        
        assert frame.shape == (2, 2)
        assert frame.columns == ["name", "age"]

    def test_from_dict_with_class_method(self):
        """Test creating an ArFrame using ArFrame.from_dict class method."""
        data = {"name": ["Alice", "Bob"], "age": [25, 30]}
        frame = ar.ArFrame.from_dict(data)
        
        assert frame.shape == (2, 2)
        assert frame.columns == ["name", "age"]

    def test_single_column(self):
        """Test creating an ArFrame with a single column."""
        data = {"ids": [1, 2, 3, 4, 5]}
        frame = ar.from_dict(data)
        
        assert frame.shape == (5, 1)
        assert frame.columns == ["ids"]

    def test_single_row(self):
        """Test creating an ArFrame with a single row."""
        data = {"name": ["Alice"], "age": [25], "city": ["NYC"]}
        frame = ar.from_dict(data)
        
        assert frame.shape == (1, 3)
        assert set(frame.columns) == {"name", "age", "city"}

    def test_integer_columns(self):
        """Test creating an ArFrame with integer columns."""
        data = {"id": [1, 2, 3], "count": [10, 20, 30]}
        frame = ar.from_dict(data)
        
        assert frame.shape == (3, 2)
        df = ar.to_pandas(frame)
        assert df["id"].dtype == "int64"
        assert df["count"].dtype == "int64"

    def test_float_columns(self):
        """Test creating an ArFrame with float columns."""
        data = {"price": [10.5, 20.75, 30.0]}
        frame = ar.from_dict(data)
        
        assert frame.shape == (3, 1)
        df = ar.to_pandas(frame)
        assert df["price"].dtype == "float64"

    def test_string_columns(self):
        """Test creating an ArFrame with string columns."""
        data = {"name": ["Alice", "Bob", "Charlie"]}
        frame = ar.from_dict(data)
        
        assert frame.shape == (3, 1)
        df = ar.to_pandas(frame)
        assert df["name"].dtype in ["object", "string"]

    def test_boolean_columns(self):
        """Test creating an ArFrame with boolean columns."""
        data = {"active": [True, False, True]}
        frame = ar.from_dict(data)
        
        assert frame.shape == (3, 1)
        df = ar.to_pandas(frame)
        assert df["active"].dtype == "bool"

    def test_mixed_data_types(self):
        """Test creating an ArFrame with mixed data types across columns."""
        data = {
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "score": [95.5, 87.3, 92.1],
            "active": [True, False, True],
        }
        frame = ar.from_dict(data)
        
        assert frame.shape == (3, 4)
        assert set(frame.columns) == {"id", "name", "score", "active"}

    def test_with_none_values(self):
        """Test creating an ArFrame with None values in columns."""
        data = {"name": ["Alice", None, "Bob"], "age": [25, 30, None]}
        frame = ar.from_dict(data)
        
        assert frame.shape == (3, 2)
        df = ar.to_pandas(frame)
        assert df["name"].isna().any()
        assert df["age"].isna().any()

    def test_unequal_column_lengths_raises_error(self):
        """Test that unequal column lengths raise an error."""
        data = {"name": ["Alice", "Bob"], "age": [25]}
        with pytest.raises((ValueError, TypeError)):
            ar.from_dict(data)

    def test_column_order_preserved(self):
        """Test that column order is preserved from dictionary."""
        data = {"z_col": [1, 2], "a_col": [3, 4], "m_col": [5, 6]}
        frame = ar.from_dict(data)
        
        assert frame.columns == ["z_col", "a_col", "m_col"]

    def test_convert_back_to_pandas(self):
        """Test converting an ArFrame back to pandas DataFrame."""
        data = {"name": ["Alice", "Bob"], "age": [25, 30]}
        frame = ar.from_dict(data)
        
        df = ar.to_pandas(frame)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 2)
        assert list(df.columns) == ["name", "age"]

    def test_compatibility_with_pipeline(self):
        """Test that from_dict result works with pipeline operations."""
        data = {"name": [" Alice ", " Bob "], "age": [25, 30]}
        frame = ar.from_dict(data)
        
        clean = ar.pipeline(frame, [("strip_whitespace",)])
        
        df = ar.to_pandas(clean)
        assert df["name"].iloc[0] == "Alice"
        assert df["name"].iloc[1] == "Bob"
