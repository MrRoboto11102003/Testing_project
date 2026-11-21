import pytest
import pandas as pd
import numpy as np
import os
import re
from pathlib import Path

import arnio as ar


class TestWriteParquetBasic:
    """Test basic write_parquet functionality."""
    
    def test_write_parquet_simple(self, tmp_path):
        """Write a simple frame to parquet and read it back."""
        p = tmp_path / "output.parquet"
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}))
        ar.write_parquet(frame, str(p))
        assert p.exists()
        
        result = pd.read_parquet(p)
        assert result["a"].tolist() == [1, 2, 3]
        assert result["b"].tolist() == ["x", "y", "z"]
    
    def test_write_parquet_pq_extension(self, tmp_path):
        """Write to .pq file extension."""
        p = tmp_path / "output.pq"
        frame = ar.from_pandas(pd.DataFrame({"x": [1, 2]}))
        ar.write_parquet(frame, str(p))
        assert p.exists()
        
        result = pd.read_parquet(p)
        assert result["x"].tolist() == [1, 2]
    
    def test_write_parquet_case_insensitive_extension(self, tmp_path):
        """File extension check is case-insensitive."""
        p = tmp_path / "output.PARQUET"
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        ar.write_parquet(frame, str(p))
        assert p.exists()
    
    def test_write_parquet_pathlib(self, tmp_path):
        """Accept pathlib.Path objects."""
        p = tmp_path / "output.parquet"
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        ar.write_parquet(frame, p)
        assert p.exists()


class TestWriteParquetValidationOrder:
    """Test input validation in correct order."""
    
    def test_frame_must_be_arframe(self, tmp_path):
        """Frame must be an ArFrame, not a pandas DataFrame."""
        p = tmp_path / "output.parquet"
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(TypeError, match="frame must be an ArFrame"):
            ar.write_parquet(df, str(p))
    
    def test_path_type_validation(self, tmp_path):
        """Path must be string, bytes, or PathLike."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        with pytest.raises(TypeError, match="path must be a string, bytes, or os.PathLike object, got 'int'"):
            ar.write_parquet(frame, 123)
    
    def test_path_extension_validation(self, tmp_path):
        """Path must end with .parquet or .pq."""
        p = tmp_path / "output.csv"
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        with pytest.raises(ValueError, match="Unsupported file format:.*\\.parquet.*\\.pq"):
            ar.write_parquet(frame, str(p))
        assert not p.exists()
    
    def test_compression_type_validation(self, tmp_path):
        """Compression must be a string."""
        p = tmp_path / "output.parquet"
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        with pytest.raises(TypeError, match="compression must be a string"):
            ar.write_parquet(frame, str(p), compression=None)
    
    def test_compression_value_validation(self, tmp_path):
        """Compression must be a valid codec."""
        p = tmp_path / "output.parquet"
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        with pytest.raises(ValueError, match=re.escape("Unknown compression codec: 'lz4'")):
            ar.write_parquet(frame, str(p), compression="lz4")
    
    def test_compression_case_sensitive(self, tmp_path):
        """Compression codec check is case-sensitive."""
        p = tmp_path / "output.parquet"
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        with pytest.raises(ValueError, match="Unknown compression codec"):
            ar.write_parquet(frame, str(p), compression="SNAPPY")
    
    def test_row_group_size_type_validation(self, tmp_path):
        """row_group_size must be an integer (not bool or float)."""
        p = tmp_path / "output.parquet"
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        
        with pytest.raises(TypeError, match="row_group_size must be an integer"):
            ar.write_parquet(frame, str(p), row_group_size=True)
        
        with pytest.raises(TypeError, match="row_group_size must be an integer"):
            ar.write_parquet(frame, str(p), row_group_size=2.5)
    
    def test_row_group_size_positive_validation(self, tmp_path):
        """row_group_size must be positive."""
        p = tmp_path / "output.parquet"
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        
        with pytest.raises(ValueError, match="row_group_size must be a positive integer"):
            ar.write_parquet(frame, str(p), row_group_size=0)
        
        with pytest.raises(ValueError, match="row_group_size must be a positive integer"):
            ar.write_parquet(frame, str(p), row_group_size=-1)
    
    def test_preserve_attrs_type_validation(self, tmp_path):
        """preserve_attrs must be a bool."""
        p = tmp_path / "output.parquet"
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))
        
        with pytest.raises(TypeError, match="preserve_attrs must be a bool"):
            ar.write_parquet(frame, str(p), preserve_attrs=1)
        
        with pytest.raises(TypeError, match="preserve_attrs must be a bool"):
            ar.write_parquet(frame, str(p), preserve_attrs=None)


class TestWriteParquetCompressionOptions:
    """Test different compression codecs."""
    
    @pytest.mark.parametrize("codec", ["snappy", "gzip", "brotli", "zstd", "none"])
    def test_valid_compression_codecs(self, tmp_path, codec):
        """All valid compression codecs can be written."""
        p = tmp_path / f"output_{codec}.parquet"
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
        ar.write_parquet(frame, str(p), compression=codec)
        assert p.exists()
        
        result = pd.read_parquet(p)
        assert result["a"].tolist() == [1, 2, 3]


class TestWriteParquetRowGroupSize:
    """Test row_group_size parameter."""
    
    def test_row_group_size_valid(self, tmp_path):
        """row_group_size controls row group splitting."""
        p = tmp_path / "output.parquet"
        frame = ar.from_pandas(pd.DataFrame({"a": range(100)}))
        ar.write_parquet(frame, str(p), row_group_size=25)
        assert p.exists()
        
        result = pd.read_parquet(p)
        assert result["a"].tolist() == list(range(100))


class TestWriteParquetEdgeCases:
    """Test edge cases and special scenarios."""
    
    def test_empty_frame_no_rows(self, tmp_path):
        """Frame with no rows but columns is allowed."""
        p = tmp_path / "output.parquet"
        frame = ar.from_pandas(pd.DataFrame({"a": [], "b": []}))
        ar.write_parquet(frame, str(p))
        assert p.exists()
        
        result = pd.read_parquet(p)
        assert len(result) == 0
        assert list(result.columns) == ["a", "b"]
    
    def test_nulls_preserved(self, tmp_path):
        """Null values are preserved in parquet output."""
        p = tmp_path / "output.parquet"
        frame = ar.from_pandas(pd.DataFrame({"a": [1, None, 3]}))
        ar.write_parquet(frame, str(p))
        
        result = pd.read_parquet(p)
        assert result["a"].iloc[0] == 1
        assert pd.isna(result["a"].iloc[1])
        assert result["a"].iloc[2] == 3
    
    def test_column_order_preserved(self, tmp_path):
        """Column order is preserved in output."""
        p = tmp_path / "output.parquet"
        frame = ar.from_pandas(pd.DataFrame({"z": [1], "a": [2], "m": [3]}))
        ar.write_parquet(frame, str(p))
        
        result = pd.read_parquet(p)
        assert result.columns.tolist() == ["z", "a", "m"]
    
    def test_mixed_dtypes(self, tmp_path):
        """Frame with mixed dtypes can be written."""
        p = tmp_path / "output.parquet"
        df = pd.DataFrame({
            "int_col": [1, 2, 3],
            "float_col": [1.1, 2.2, 3.3],
            "str_col": ["a", "b", "c"],
            "bool_col": [True, False, True]
        })
        frame = ar.from_pandas(df)
        ar.write_parquet(frame, str(p))
        assert p.exists()
    
    def test_preserve_attrs_false(self, tmp_path):
        """preserve_attrs=False silently drops attrs."""
        p = tmp_path / "output.parquet"
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))
        ar.write_parquet(frame, str(p), preserve_attrs=False)
        assert p.exists()
    
    def test_zero_column_with_rows_raises(self, tmp_path):
        """Frame with zero columns but rows raises ValueError."""
        p = tmp_path / "output.parquet"
        # Try to create a frame with shape (n, 0)
        df_empty_cols = pd.DataFrame(index=range(3))
        frame = ar.from_pandas(df_empty_cols)
        
        # Guard: only run if frame actually has 0 columns and rows > 0
        if frame.shape[1] == 0 and frame.shape[0] > 0:
            with pytest.raises(ValueError, match="Cannot write a zero-column ArFrame"):
                ar.write_parquet(frame, str(p))
            assert not p.exists()
        else:
            pytest.skip("Cannot construct (n, 0) frame with this arnio version")
