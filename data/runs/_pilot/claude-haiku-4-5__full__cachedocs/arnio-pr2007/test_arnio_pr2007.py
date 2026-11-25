import pytest
import os
from pathlib import Path
import pandas as pd
import arnio as ar


class TestWriteParquet:
    """Tests for arnio.io.write_parquet function."""
    
    @pytest.fixture
    def sample_frame(self):
        """Create a simple test ArFrame."""
        data = {
            "id": [1, 2, 3],
            "name": ["alice", "bob", "charlie"],
            "score": [95.5, 87.0, 92.3]
        }
        return ar.from_pandas(pd.DataFrame(data))
    
    # ─── Basic functionality tests ───────────────────────────────────
    
    def test_write_parquet_basic(self, sample_frame, tmp_path):
        """Test basic write with default parameters."""
        path = tmp_path / "output.parquet"
        ar.write_parquet(sample_frame, path)
        assert path.exists()
        assert path.stat().st_size > 0
    
    def test_write_parquet_pq_extension(self, sample_frame, tmp_path):
        """Test write with .pq extension."""
        path = tmp_path / "output.pq"
        ar.write_parquet(sample_frame, path)
        assert path.exists()
    
    @pytest.mark.parametrize("compression", [
        "snappy", "gzip", "zstd", "brotli", "none"
    ])
    def test_write_parquet_compression_codecs(self, sample_frame, tmp_path, compression):
        """Test write with various supported compression codecs."""
        path = tmp_path / f"output_{compression}.parquet"
        ar.write_parquet(sample_frame, path, compression=compression)
        assert path.exists()
    
    def test_write_parquet_row_group_size(self, sample_frame, tmp_path):
        """Test write with custom row_group_size."""
        path = tmp_path / "output.parquet"
        ar.write_parquet(sample_frame, path, row_group_size=50_000)
        assert path.exists()
    
    def test_write_parquet_preserve_attrs_false(self, sample_frame, tmp_path):
        """Test write with preserve_attrs=False."""
        path = tmp_path / "output.parquet"
        ar.write_parquet(sample_frame, path, preserve_attrs=False)
        assert path.exists()
    
    def test_write_parquet_roundtrip(self, sample_frame, tmp_path):
        """Test write and read back consistency."""
        path = tmp_path / "output.parquet"
        ar.write_parquet(sample_frame, path)
        
        df_original = ar.to_pandas(sample_frame)
        df_read = pd.read_parquet(path)
        
        pd.testing.assert_frame_equal(df_original, df_read)
    
    def test_write_parquet_pathlib_path(self, sample_frame, tmp_path):
        """Test write accepts pathlib.Path as path argument."""
        path = tmp_path / "output.parquet"
        ar.write_parquet(sample_frame, path)
        assert isinstance(path, Path)
        assert path.exists()
    
    # ─── Error handling tests ────────────────────────────────────────
    
    def test_write_parquet_invalid_extension(self, sample_frame, tmp_path):
        """Test that invalid file extension raises ValueError."""
        path = tmp_path / "output.csv"
        with pytest.raises(ValueError):
            ar.write_parquet(sample_frame, path)
    
    def test_write_parquet_invalid_compression(self, sample_frame, tmp_path):
        """Test that unrecognized compression codec raises ValueError."""
        path = tmp_path / "output.parquet"
        with pytest.raises(ValueError):
            ar.write_parquet(sample_frame, path, compression="lz4")
    
    def test_write_parquet_invalid_row_group_size_negative(self, sample_frame, tmp_path):
        """Test that negative row_group_size raises ValueError."""
        path = tmp_path / "output.parquet"
        with pytest.raises(ValueError):
            ar.write_parquet(sample_frame, path, row_group_size=-100)
    
    def test_write_parquet_invalid_row_group_size_zero(self, sample_frame, tmp_path):
        """Test that zero row_group_size raises ValueError."""
        path = tmp_path / "output.parquet"
        with pytest.raises(ValueError):
            ar.write_parquet(sample_frame, path, row_group_size=0)
    
    def test_write_parquet_invalid_preserve_attrs_type(self, sample_frame, tmp_path):
        """Test that non-boolean preserve_attrs raises TypeError."""
        path = tmp_path / "output.parquet"
        with pytest.raises(TypeError):
            ar.write_parquet(sample_frame, path, preserve_attrs="yes")
    
    def test_write_parquet_string_path(self, sample_frame, tmp_path):
        """Test write accepts string path."""
        path_str = str(tmp_path / "output.parquet")
        ar.write_parquet(sample_frame, path_str)
        assert os.path.exists(path_str)
    
    def test_write_parquet_combined_options(self, sample_frame, tmp_path):
        """Test write with multiple options combined."""
        path = tmp_path / "output.parquet"
        ar.write_parquet(
            sample_frame,
            path,
            compression="gzip",
            row_group_size=10_000,
            preserve_attrs=False
        )
        assert path.exists()
        # Verify data integrity
        df_read = pd.read_parquet(path)
        assert len(df_read) == 3
        assert list(df_read.columns) == ["id", "name", "score"]
