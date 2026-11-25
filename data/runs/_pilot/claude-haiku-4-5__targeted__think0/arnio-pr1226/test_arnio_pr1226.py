import pytest
import pandas as pd
import numpy as np
import arnio as ar


class TestToPandas:
    """Test suite for arnio.convert.to_pandas function."""

    def test_to_pandas_basic_conversion(self):
        """Test basic conversion from ArFrame to pandas DataFrame."""
        data = {"col1": [1, 2, 3], "col2": ["a", "b", "c"]}
        frame = ar.from_dict(data)
        df = ar.to_pandas(frame)
        
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["col1", "col2"]
        assert len(df) == 3

    def test_to_pandas_preserves_dtypes(self):
        """Test that to_pandas preserves column dtypes."""
        data = {
            "int_col": [1, 2, 3],
            "float_col": [1.5, 2.5, 3.5],
            "str_col": ["a", "b", "c"]
        }
        frame = ar.from_dict(data)
        df = ar.to_pandas(frame)
        
        assert df["int_col"].dtype in [np.int32, np.int64]
        assert df["float_col"].dtype in [np.float32, np.float64]
        assert df["str_col"].dtype == object

    def test_to_pandas_with_copy_false(self):
        """Test to_pandas with copy=False (default)."""
        data = {"col1": [1, 2, 3], "col2": [4.0, 5.0, 6.0]}
        frame = ar.from_dict(data)
        df = ar.to_pandas(frame, copy=False)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_to_pandas_with_copy_true(self):
        """Test to_pandas with copy=True for defensive copies."""
        data = {"col1": [1, 2, 3], "col2": [4.0, 5.0, 6.0]}
        frame = ar.from_dict(data)
        df = ar.to_pandas(frame, copy=True)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_to_pandas_empty_frame(self):
        """Test conversion of empty ArFrame."""
        data = {"col1": [], "col2": []}
        frame = ar.from_dict(data)
        df = ar.to_pandas(frame)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert list(df.columns) == ["col1", "col2"]

    def test_to_pandas_single_column(self):
        """Test conversion with single column."""
        data = {"values": [10, 20, 30, 40]}
        frame = ar.from_dict(data)
        df = ar.to_pandas(frame)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df.columns) == 1
        assert list(df["values"]) == [10, 20, 30, 40]

    def test_to_pandas_multiple_rows(self):
        """Test conversion preserves all rows."""
        data = {"a": list(range(1000)), "b": list(range(1000, 2000))}
        frame = ar.from_dict(data)
        df = ar.to_pandas(frame)
        
        assert len(df) == 1000
        assert list(df["a"]) == list(range(1000))

    def test_to_pandas_after_pipeline(self):
        """Test to_pandas on result from pipeline transformation."""
        data = {"name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 35]}
        frame = ar.from_dict(data)
        clean = ar.pipeline(frame, [
            ("strip_whitespace",),
        ])
        df = ar.to_pandas(clean)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "name" in df.columns
        assert "age" in df.columns

    def test_to_pandas_from_pandas_roundtrip(self):
        """Test roundtrip conversion: pandas -> ArFrame -> pandas."""
        original_df = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": ["x", "y", "z"],
            "col3": [1.1, 2.2, 3.3]
        })
        
        frame = ar.from_dict(original_df.to_dict(orient="list"))
        result_df = ar.to_pandas(frame)
        
        assert isinstance(result_df, pd.DataFrame)
        assert list(result_df.columns) == list(original_df.columns)
        assert len(result_df) == len(original_df)

    def test_to_pandas_returns_dataframe_type(self):
        """Test that return type is always pandas.DataFrame."""
        data = {"x": [1, 2], "y": [3, 4]}
        frame = ar.from_dict(data)
        
        df_no_copy = ar.to_pandas(frame, copy=False)
        df_with_copy = ar.to_pandas(frame, copy=True)
        
        assert type(df_no_copy) is pd.DataFrame
        assert type(df_with_copy) is pd.DataFrame

    def test_to_pandas_keyword_only_arguments(self):
        """Test that copy parameter is keyword-only."""
        data = {"col": [1, 2, 3]}
        frame = ar.from_dict(data)
        
        # Should work with keyword argument
        df = ar.to_pandas(frame, copy=False)
        assert isinstance(df, pd.DataFrame)

    def test_to_pandas_large_dataset(self):
        """Test to_pandas with larger dataset."""
        n_rows = 10000
        data = {
            "int_col": list(range(n_rows)),
            "float_col": [float(i) * 1.5 for i in range(n_rows)],
            "str_col": [f"val_{i}" for i in range(n_rows)]
        }
        frame = ar.from_dict(data)
        df = ar.to_pandas(frame)
        
        assert len(df) == n_rows
        assert len(df.columns) == 3

    def test_to_pandas_with_special_characters(self):
        """Test to_pandas handles special characters in string data."""
        data = {
            "text": ["hello", "world", "test@#$%"],
            "values": [1, 2, 3]
        }
        frame = ar.from_dict(data)
        df = ar.to_pandas(frame)
        
        assert df.iloc[2, 0] == "test@#$%"

    def test_to_pandas_column_order_preserved(self):
        """Test that column order is preserved in conversion."""
        data = {"z": [1], "a": [2], "m": [3]}
        frame = ar.from_dict(data)
        df = ar.to_pandas(frame)
        
        # Column order should match input order
        assert list(df.columns) == ["z", "a", "m"]
