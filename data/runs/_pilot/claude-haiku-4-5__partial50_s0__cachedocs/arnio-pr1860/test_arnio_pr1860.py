import pytest
import os
from pathlib import Path
import pandas as pd

try:
    import pyarrow as pa
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

import arnio as ar


@pytest.mark.skipif(not HAS_PYARROW, reason="pyarrow not installed")
class TestWriteParquet:
    """Test suite for arnio.io.write_parquet"""

    def test_basic_write(self, tmp_path):
        """Write simple DataFrame to parquet with defaults"""
        df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
        frame = ar.from_pandas(df)
        path = tmp_path / "data.parquet"
        
        ar.write_parquet(frame, path)
        
        assert path.exists()
        assert path.stat().st_size > 0

    def test_pq_extension(self, tmp_path):
        """Accept .pq file extension"""
        df = pd.DataFrame({"id": [1, 2, 3]})
        frame = ar.from_pandas(df)
        path = tmp_path / "data.pq"
        
        ar.write_parquet(frame, path)
        assert path.exists()

    def test_roundtrip_integrity(self, tmp_path):
        """Data integrity preserved through write/read cycle"""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "score": [95.5, 87.3, 91.0]
        })
        frame = ar.from_pandas(df)
        path = tmp_path / "roundtrip.parquet"
        
        ar.write_parquet(frame, path)
        frame_read = ar.read_parquet(path)
        df_read = ar.to_pandas(frame_read)
        
        pd.testing.assert_frame_equal(df, df_read)

    def test_compression_snappy(self, tmp_path):
        """Snappy compression (default codec)"""
        df = pd.DataFrame({"col": range(100)})
        frame = ar.from_pandas(df)
        path = tmp_path / "snappy.parquet"
        
        ar.write_parquet(frame, path, compression="snappy")
        
        assert path.exists()
        frame_read = ar.read_parquet(path)
        assert frame_read.shape[0] == 100

    def test_compression_gzip(self, tmp_path):
        """Gzip compression codec"""
        df = pd.DataFrame({"col": range(100)})
        frame = ar.from_pandas(df)
        path = tmp_path / "gzip.parquet"
        
        ar.write_parquet(frame, path, compression="gzip")
        assert path.exists()

    def test_compression_zstd(self, tmp_path):
        """Zstd compression codec"""
        df = pd.DataFrame({"col": range(100)})
        frame = ar.from_pandas(df)
        path = tmp_path / "zstd.parquet"
        
        ar.write_parquet(frame, path, compression="zstd")
        assert path.exists()

    def test_compression_brotli(self, tmp_path):
        """Brotli compression codec"""
        df = pd.DataFrame({"col": range(100)})
        frame = ar.from_pandas(df)
        path = tmp_path / "brotli.parquet"
        
        ar.write_parquet(frame, path, compression="brotli")
        assert path.exists()

    def test_compression_none(self, tmp_path):
        """No compression"""
        df = pd.DataFrame({"col": range(100)})
        frame = ar.from_pandas(df)
        path = tmp_path / "uncompressed.parquet"
        
        ar.write_parquet(frame, path, compression="none")
        assert path.exists()

    def test_row_group_size(self, tmp_path):
        """Custom row_group_size parameter"""
        df = pd.DataFrame({"col": range(5000)})
        frame = ar.from_pandas(df)
        path = tmp_path / "row_groups.parquet"
        
        ar.write_parquet(frame, path, row_group_size=500)
        
        assert path.exists()
        frame_read = ar.read_parquet(path)
        assert frame_read.shape[0] == 5000

    def test_mixed_dtypes(self, tmp_path):
        """Multiple data types are preserved"""
        df = pd.DataFrame({
            "int_col": [1, 2, 3],
            "float_col": [1.1, 2.2, 3.3],
            "str_col": ["a", "b", "c"],
            "bool_col": [True, False, True]
        })
        frame = ar.from_pandas(df)
        path = tmp_path / "mixed.parquet"
        
        ar.write_parquet(frame, path)
        
        frame_read = ar.read_parquet(path)
        df_read = ar.to_pandas(frame_read)
        assert df_read.shape == df.shape

    def test_null_values(self, tmp_path):
        """Null values are preserved"""
        df = pd.DataFrame({
            "col1": [1.0, None, 3.0],
            "col2": ["a", "b", None]
        })
        frame = ar.from_pandas(df)
        path = tmp_path / "nulls.parquet"
        
        ar.write_parquet(frame, path)
        
        frame_read = ar.read_parquet(path)
        df_read = ar.to_pandas(frame_read)
        assert df_read.isna().sum().sum() > 0

    def test_empty_frame(self, tmp_path):
        """Empty DataFrame can be written"""
        df = pd.DataFrame({"col1": [], "col2": []})
        frame = ar.from_pandas(df)
        path = tmp_path / "empty.parquet"
        
        ar.write_parquet(frame, path)
        
        assert path.exists()
        frame_read = ar.read_parquet(path)
        assert frame_read.shape[0] == 0

    def test_string_path(self, tmp_path):
        """String path argument"""
        df = pd.DataFrame({"col": [1, 2, 3]})
        frame = ar.from_pandas(df)
        path = str(tmp_path / "string.parquet")
        
        ar.write_parquet(frame, path)
        assert os.path.exists(path)

    def test_pathlib_path(self, tmp_path):
        """pathlib.Path argument"""
        df = pd.DataFrame({"col": [1, 2, 3]})
        frame = ar.from_pandas(df)
        path = tmp_path / "pathlib.parquet"
        
        ar.write_parquet(frame, path)
        assert path.exists()

    def test_large_frame(self, tmp_path):
        """Large DataFrame handling"""
        df = pd.DataFrame({"col": range(100000)})
        frame = ar.from_pandas(df)
        path = tmp_path / "large.parquet"
        
        ar.write_parquet(frame, path)
        
        frame_read = ar.read_parquet(path)
        assert frame_read.shape[0] == 100000
