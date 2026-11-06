import pytest
import pandas as pd
import arnio as ar
import tempfile
import os
from pathlib import Path


class TestWriteParquet:
    """Test suite for arnio.io.write_parquet"""
    
    @pytest.fixture
    def sample_frame(self):
        """Create a sample ArFrame for testing"""
        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "score": [95.5, 87.3, 92.1, 78.9, 88.5],
            "active": [True, False, True, False, True]
        })
        return ar.from_pandas(df)
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    def test_basic_write_parquet_extension(self, sample_frame, temp_dir):
        """Test basic parquet writing with .parquet extension"""
        path = os.path.join(temp_dir, "output.parquet")
        result = ar.write_parquet(sample_frame, path)
        
        assert result is None, "write_parquet should return None"
        assert os.path.exists(path), "File should be created"
    
    def test_write_with_pq_extension(self, sample_frame, temp_dir):
        """Test that .pq extension is accepted"""
        path = os.path.join(temp_dir, "output.pq")
        ar.write_parquet(sample_frame, path)
        
        assert os.path.exists(path), "File with .pq extension should be created"
    
    def test_write_with_pathlike(self, sample_frame, temp_dir):
        """Test that PathLike objects are accepted"""
        path = Path(temp_dir) / "output.parquet"
        ar.write_parquet(sample_frame, path)
        
        assert path.exists(), "File should be created from PathLike path"
    
    def test_compression_snappy_default(self, sample_frame, temp_dir):
        """Test default snappy compression (no compression specified)"""
        path = os.path.join(temp_dir, "output.parquet")
        ar.write_parquet(sample_frame, path)
        
        assert os.path.exists(path), "File should be created with default compression"
    
    def test_compression_snappy_explicit(self, sample_frame, temp_dir):
        """Test explicit snappy compression"""
        path = os.path.join(temp_dir, "output.parquet")
        ar.write_parquet(sample_frame, path, compression="snappy")
        
        assert os.path.exists(path), "File should be created with snappy compression"
    
    def test_compression_gzip(self, sample_frame, temp_dir):
        """Test gzip compression codec"""
        path = os.path.join(temp_dir, "output_gzip.parquet")
        ar.write_parquet(sample_frame, path, compression="gzip")
        
        assert os.path.exists(path), "File should be created with gzip compression"
    
    def test_compression_zstd(self, sample_frame, temp_dir):
        """Test zstd compression codec"""
        path = os.path.join(temp_dir, "output_zstd.parquet")
        ar.write_parquet(sample_frame, path, compression="zstd")
        
        assert os.path.exists(path), "File should be created with zstd compression"
    
    def test_compression_brotli(self, sample_frame, temp_dir):
        """Test brotli compression codec"""
        path = os.path.join(temp_dir, "output_brotli.parquet")
        ar.write_parquet(sample_frame, path, compression="brotli")
        
        assert os.path.exists(path), "File should be created with brotli compression"
    
    def test_compression_none(self, sample_frame, temp_dir):
        """Test no compression codec"""
        path = os.path.join(temp_dir, "output_none.parquet")
        ar.write_parquet(sample_frame, path, compression="none")
        
        assert os.path.exists(path), "File should be created with no compression"
    
    def test_row_group_size_small(self, sample_frame, temp_dir):
        """Test row_group_size parameter with small value"""
        path = os.path.join(temp_dir, "output_rgs_small.parquet")
        ar.write_parquet(sample_frame, path, row_group_size=1)
        
        assert os.path.exists(path), "File should be created with row_group_size=1"
    
    def test_row_group_size_large(self, sample_frame, temp_dir):
        """Test row_group_size parameter with large value"""
        path = os.path.join(temp_dir, "output_rgs_large.parquet")
        ar.write_parquet(sample_frame, path, row_group_size=50_000)
        
        assert os.path.exists(path), "File should be created with row_group_size=50000"
    
    def test_preserve_attrs_true(self, sample_frame, temp_dir):
        """Test preserve_attrs parameter set to True"""
        path = os.path.join(temp_dir, "output_preserve_true.parquet")
        ar.write_parquet(sample_frame, path, preserve_attrs=True)
        
        assert os.path.exists(path), "File should be created with preserve_attrs=True"
    
    def test_preserve_attrs_false(self, sample_frame, temp_dir):
        """Test preserve_attrs parameter set to False"""
        path = os.path.join(temp_dir, "output_preserve_false.parquet")
        ar.write_parquet(sample_frame, path, preserve_attrs=False)
        
        assert os.path.exists(path), "File should be created with preserve_attrs=False"
    
    def test_overwrite_existing_file(self, sample_frame, temp_dir):
        """Test overwriting an existing file"""
        path = os.path.join(temp_dir, "output_overwrite.parquet")
        
        ar.write_parquet(sample_frame, path, compression="snappy")
        assert os.path.exists(path), "File should be created first time"
        
        ar.write_parquet(sample_frame, path, compression="gzip")
        assert os.path.exists(path), "File should still exist after overwrite"
    
    def test_roundtrip_data_integrity(self, sample_frame, temp_dir):
        """Test that written data can be read back correctly"""
        path = os.path.join(temp_dir, "roundtrip.parquet")
        ar.write_parquet(sample_frame, path)
        
        read_frame = ar.read_parquet(path)
        
        assert read_frame.shape == sample_frame.shape, "Shape should match after roundtrip"
        assert read_frame.columns == sample_frame.columns, "Columns should match after roundtrip"
