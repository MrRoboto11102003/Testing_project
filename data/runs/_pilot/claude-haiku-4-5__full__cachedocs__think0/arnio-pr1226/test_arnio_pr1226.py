import pytest
import pandas as pd
import numpy as np
import arnio as ar
from io import StringIO


class TestToPandas:
    """Tests for ar.to_pandas() conversion function."""

    def test_basic_conversion_from_read_csv(self, tmp_path):
        """Test basic conversion of ArFrame from CSV to pandas DataFrame."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("name,age,score\nAlice,30,95.5\nBob,25,87.0")
        
        frame = ar.read_csv(str(csv_file))
        df = ar.to_pandas(frame)
        
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 3)
        assert list(df.columns) == ["name", "age", "score"]
        assert df["name"].tolist() == ["Alice", "Bob"]

    def test_conversion_preserves_dtypes(self, tmp_path):
        """Test that dtypes are preserved during conversion."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("id,value,active\n1,10.5,true\n2,20.3,false")
        
        frame = ar.read_csv(str(csv_file))
        df = ar.to_pandas(frame)
        
        assert df["id"].dtype == np.int64
        assert df["value"].dtype == np.float64
        assert df["active"].dtype == bool

    def test_conversion_handles_nulls(self, tmp_path):
        """Test that null values are properly handled."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("name,score\nAlice,95\nBob,\nCharlie,88")
        
        frame = ar.read_csv(str(csv_file))
        df = ar.to_pandas(frame)
        
        assert df["name"].notna().sum() == 3
        assert pd.isna(df["score"].iloc[1])
        assert df["score"].iloc[0] == 95.0
        assert df["score"].iloc[2] == 88.0

    def test_conversion_with_copy_false(self, tmp_path):
        """Test zero-copy path when copy=False (default)."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("id,value\n1,10.5\n2,20.3\n3,30.1")
        
        frame = ar.read_csv(str(csv_file))
        df = ar.to_pandas(frame, copy=False)
        
        # Should still be a valid DataFrame
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert df["value"].dtype == np.float64

    def test_conversion_with_copy_true(self, tmp_path):
        """Test defensive copy when copy=True."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("id,value\n1,10.5\n2,20.3")
        
        frame = ar.read_csv(str(csv_file))
        df_nocopy = ar.to_pandas(frame, copy=False)
        df_copy = ar.to_pandas(frame, copy=True)
        
        # Both should be valid DataFrames with same data
        assert df_nocopy.equals(df_copy)
        assert isinstance(df_copy, pd.DataFrame)

    def test_conversion_from_from_dict(self):
        """Test conversion when ArFrame created via from_dict."""
        data = {"name": ["Alice", "Bob"], "age": [30, 25]}
        frame = ar.from_dict(data)
        df = ar.to_pandas(frame)
        
        assert df.shape == (2, 2)
        assert df["name"].tolist() == ["Alice", "Bob"]
        assert df["age"].tolist() == [30, 25]

    def test_conversion_from_from_pandas_roundtrip(self):
        """Test roundtrip conversion: pandas -> ArFrame -> pandas."""
        original_df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "score": [95.5, 87.0, None]
        })
        
        frame = ar.from_pandas(original_df)
        result_df = ar.to_pandas(frame)
        
        # Should have same shape and columns
        assert result_df.shape == original_df.shape
        assert list(result_df.columns) == list(original_df.columns)
        assert result_df["id"].tolist() == original_df["id"].tolist()

    def test_conversion_empty_frame(self):
        """Test conversion of empty ArFrame."""
        data = {"name": [], "value": []}
        frame = ar.from_dict(data)
        df = ar.to_pandas(frame)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert list(df.columns) == ["name", "value"]

    def test_conversion_single_row(self, tmp_path):
        """Test conversion of ArFrame with single row."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("id,name\n1,Alice")
        
        frame = ar.read_csv(str(csv_file))
        df = ar.to_pandas(frame)
        
        assert len(df) == 1
        assert df["id"].iloc[0] == 1
        assert df["name"].iloc[0] == "Alice"

    def test_conversion_many_columns(self):
        """Test conversion with many columns."""
        data = {f"col_{i}": [i, i+1, i+2] for i in range(20)}
        frame = ar.from_dict(data)
        df = ar.to_pandas(frame)
        
        assert df.shape == (3, 20)
        assert len(df.columns) == 20
        for col in df.columns:
            assert col in data

    def test_conversion_string_column(self, tmp_path):
        """Test conversion preserves string columns."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("text\nhello world\nfoo bar\nempty")
        
        frame = ar.read_csv(str(csv_file))
        df = ar.to_pandas(frame)
        
        assert df["text"].dtype == object or df["text"].dtype.name == "string"
        assert df["text"].tolist()[0] == "hello world"

    def test_conversion_with_special_characters(self, tmp_path):
        """Test conversion handles special characters correctly."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text('name,comment\nAlice,"Hello, World!"\nBob,"Line1\\nLine2"')
        
        frame = ar.read_csv(str(csv_file))
        df = ar.to_pandas(frame)
        
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 2)

    def test_conversion_maintains_column_order(self, tmp_path):
        """Test that column order is preserved during conversion."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("z,a,m,b\n1,2,3,4")
        
        frame = ar.read_csv(str(csv_file))
        df = ar.to_pandas(frame)
        
        assert list(df.columns) == ["z", "a", "m", "b"]

    def test_conversion_after_pipeline(self, tmp_path):
        """Test conversion after applying pipeline operations."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("name,score\n  Alice  ,95\n  Bob  ,87")
        
        frame = ar.read_csv(str(csv_file))
        clean = ar.pipeline(frame, [("strip_whitespace",)])
        df = ar.to_pandas(clean)
        
        assert df["name"].iloc[0] == "Alice"
        assert df["name"].iloc[1] == "Bob"

    def test_conversion_numeric_nulls(self, tmp_path):
        """Test that numeric columns with nulls are handled properly."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("score\n95\n\n87")
        
        frame = ar.read_csv(str(csv_file))
        df = ar.to_pandas(frame)
        
        # Should have numeric dtype with NaN for missing values
        assert pd.api.types.is_numeric_dtype(df["score"])
        assert pd.isna(df["score"].iloc[1])
        assert df["score"].iloc[0] == 95.0
