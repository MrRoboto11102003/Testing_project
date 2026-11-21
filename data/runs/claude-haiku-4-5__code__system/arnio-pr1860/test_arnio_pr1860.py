import pytest
import sys
import os
import json
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd

import arnio as ar


def create_arframe(data=None):
    """Helper to create an ArFrame from data dict or DataFrame."""
    if data is None:
        data = {"col1": [1, 2, 3, 4, 5], "col2": ["a", "b", "c", "d", "e"]}
    
    if isinstance(data, dict):
        df = pd.DataFrame(data)
    else:
        df = data
    
    # Try multiple API patterns to create ArFrame
    if hasattr(ar, "from_pandas"):
        return ar.from_pandas(df)
    elif hasattr(ar, "read_pandas"):
        return ar.read_pandas(df)
    elif hasattr(df, "arnio"):
        return df.arnio
    else:
        # Last resort: direct constructor
        return ar.ArFrame(df)


class TestWriteParquetBasic:
    """Test basic write_parquet functionality."""
    
    def test_write_parquet_simple(self, tmp_path):
        """Write a simple ArFrame to parquet and verify content."""
        frame = create_arframe({"x": [1, 2, 3], "y": [4.5, 5.5, 6.5]})
        output = tmp_path / "output.parquet"
        
        ar.write_parquet(frame, output)
        
        assert output.exists()
        result = pd.read_parquet(output)
        assert list(result.columns) == ["x", "y"]
        assert len(result) == 3
        np.testing.assert_array_equal(result["x"], [1, 2, 3])
    
    def test_write_parquet_pq_extension(self, tmp_path):
        """Write with .pq file extension."""
        frame = create_arframe({"data": [10, 20, 30]})
        output = tmp_path / "output.pq"
        
        ar.write_parquet(frame, output)
        
        assert output.exists()
        result = pd.read_parquet(output)
        np.testing.assert_array_equal(result["data"], [10, 20, 30])
    
    def test_write_parquet_various_dtypes(self, tmp_path):
        """Write DataFrame with various data types."""
        df = pd.DataFrame({
            "int_col": np.array([1, 2, 3], dtype=np.int32),
            "float_col": np.array([1.1, 2.2, 3.3], dtype=np.float64),
            "str_col": ["a", "b", "c"],
            "bool_col": [True, False, True]
        })
        frame = create_arframe(df)
        output = tmp_path / "output.parquet"
        
        ar.write_parquet(frame, output)
        
        result = pd.read_parquet(output)
        assert len(result) == 3
        assert len(result.columns) == 4


class TestWriteParquetCompressions:
    """Test various compression codecs."""
    
    @pytest.mark.parametrize("compression", ["snappy", "gzip", "none"])
    def test_write_parquet_compression_codecs(self, tmp_path, compression):
        """Test writing with different compression codecs."""
        frame = create_arframe({"nums": list(range(100))})
        output = tmp_path / f"output_{compression}.parquet"
        
        ar.write_parquet(frame, output, compression=compression)
        
        assert output.exists()
        result = pd.read_parquet(output)
        assert len(result) == 100
        np.testing.assert_array_equal(result["nums"], list(range(100)))


class TestWriteParquetRowGroupSize:
    """Test row_group_size parameter."""
    
    def test_write_parquet_custom_row_group_size(self, tmp_path):
        """Write with custom row group size."""
        frame = create_arframe({"data": list(range(1000))})
        output = tmp_path / "output.parquet"
        
        ar.write_parquet(frame, output, row_group_size=100)
        
        assert output.exists()
        result = pd.read_parquet(output)
        assert len(result) == 1000


class TestWriteParquetAttrs:
    """Test preserve_attrs parameter."""
    
    def test_write_parquet_preserve_attrs_false(self, tmp_path):
        """Write with preserve_attrs=False."""
        frame = create_arframe({"col": [1, 2, 3]})
        output = tmp_path / "output.parquet"
        
        ar.write_parquet(frame, output, preserve_attrs=False)
        
        assert output.exists()
        result = pd.read_parquet(output)
        assert len(result) == 3


class TestWriteParquetErrors:
    """Test error conditions and validation."""
    
    def test_write_parquet_non_arframe_error(self, tmp_path):
        """Raise TypeError when frame is not an ArFrame."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        output = tmp_path / "output.parquet"
        
        with pytest.raises(TypeError, match="frame must be an ArFrame"):
            ar.write_parquet(df, output)
    
    def test_write_parquet_invalid_extension_csv(self, tmp_path):
        """Raise ValueError for unsupported file extension."""
        frame = create_arframe()
        output = tmp_path / "output.csv"
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            ar.write_parquet(frame, output)
    
    def test_write_parquet_invalid_extension_txt(self, tmp_path):
        """Raise ValueError for .txt extension."""
        frame = create_arframe()
        output = tmp_path / "output.txt"
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            ar.write_parquet(frame, output)
    
    def test_write_parquet_invalid_compression_codec(self, tmp_path):
        """Raise ValueError for unknown compression codec."""
        frame = create_arframe()
        output = tmp_path / "output.parquet"
        
        with pytest.raises(ValueError, match="Unknown compression codec"):
            ar.write_parquet(frame, output, compression="unknown_codec")
    
    def test_write_parquet_negative_row_group_size(self, tmp_path):
        """Raise ValueError for negative row_group_size."""
        frame = create_arframe()
        output = tmp_path / "output.parquet"
        
        with pytest.raises(ValueError, match="row_group_size must be a positive integer"):
            ar.write_parquet(frame, output, row_group_size=-1)
    
    def test_write_parquet_zero_row_group_size(self, tmp_path):
        """Raise ValueError for zero row_group_size."""
        frame = create_arframe()
        output = tmp_path / "output.parquet"
        
        with pytest.raises(ValueError, match="row_group_size must be a positive integer"):
            ar.write_parquet(frame, output, row_group_size=0)
    
    def test_write_parquet_non_int_row_group_size(self, tmp_path):
        """Raise TypeError for non-integer row_group_size."""
        frame = create_arframe()
        output = tmp_path / "output.parquet"
        
        with pytest.raises(TypeError, match="row_group_size must be an integer"):
            ar.write_parquet(frame, output, row_group_size="100")
    
    def test_write_parquet_non_bool_preserve_attrs(self, tmp_path):
        """Raise TypeError for non-boolean preserve_attrs."""
        frame = create_arframe()
        output = tmp_path / "output.parquet"
        
        with pytest.raises(TypeError, match="preserve_attrs must be a bool"):
            ar.write_parquet(frame, output, preserve_attrs="true")
    
    def test_write_parquet_zero_columns_nonzero_rows(self, tmp_path):
        """Raise ValueError for zero-column DataFrame with rows."""
        df = pd.DataFrame(index=range(5))
        frame = create_arframe(df)
        output = tmp_path / "output.parquet"
        
        with pytest.raises(ValueError, match="Cannot write a zero-column"):
            ar.write_parquet(frame, output)


class TestWriteParquetEdgeCases:
    """Test edge cases and special scenarios."""
    
    def test_write_parquet_empty_dataframe(self, tmp_path):
        """Write an empty DataFrame with columns."""
        frame = create_arframe({"a": [], "b": []})
        output = tmp_path / "output.parquet"
        
        ar.write_parquet(frame, output)
        
        result = pd.read_parquet(output)
        assert len(result) == 0
        assert list(result.columns) == ["a", "b"]
    
    def test_write_parquet_single_row(self, tmp_path):
        """Write DataFrame with a single row."""
        frame = create_arframe({"x": [42], "y": ["single"]})
        output = tmp_path / "output.parquet"
        
        ar.write_parquet(frame, output)
        
        result = pd.read_parquet(output)
        assert len(result) == 1
        assert result["x"].iloc[0] == 42
    
    def test_write_parquet_large_integers(self, tmp_path):
        """Write DataFrame with large integer values."""
        frame = create_arframe({"big": [10**15, 10**16, 10**17]})
        output = tmp_path / "output.parquet"
        
        ar.write_parquet(frame, output)
        
        result = pd.read_parquet(output)
        assert len(result) == 3
    
    def test_write_parquet_pathlib_path(self, tmp_path):
        """Write using pathlib.Path object."""
        frame = create_arframe()
        output = tmp_path / "output.parquet"
        
        ar.write_parquet(frame, output)
        
        assert output.exists()
