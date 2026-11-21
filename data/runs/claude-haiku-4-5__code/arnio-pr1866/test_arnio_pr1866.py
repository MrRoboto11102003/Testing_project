import pytest
import pandas as pd
import tempfile
import os
from pathlib import Path

import arnio as ar


class TestWriteParquet:
    """Test suite for arnio.io.write_parquet"""

    @pytest.fixture
    def sample_frame(self):
        """Create a simple ArFrame for testing"""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c'],
            'col3': [1.1, 2.2, 3.3]
        })
        return ar.from_pandas(df)

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_write_parquet_basic(self, sample_frame, temp_dir):
        """Test basic write_parquet with .parquet extension and defaults"""
        path = os.path.join(temp_dir, "output.parquet")
        ar.write_parquet(sample_frame, path)
        assert os.path.exists(path)

    def test_write_parquet_pq_extension(self, sample_frame, temp_dir):
        """Test write_parquet with .pq extension"""
        path = os.path.join(temp_dir, "output.pq")
        ar.write_parquet(sample_frame, path)
        assert os.path.exists(path)

    def test_write_parquet_pathlib_path(self, sample_frame, temp_dir):
        """Test write_parquet accepts pathlib.Path objects"""
        path = Path(temp_dir) / "output.parquet"
        ar.write_parquet(sample_frame, path)
        assert path.exists()

    def test_write_parquet_compression_gzip(self, sample_frame, temp_dir):
        """Test write_parquet with gzip compression"""
        path = os.path.join(temp_dir, "output.parquet")
        ar.write_parquet(sample_frame, path, compression="gzip")
        assert os.path.exists(path)

    def test_write_parquet_compression_zstd(self, sample_frame, temp_dir):
        """Test write_parquet with zstd compression"""
        path = os.path.join(temp_dir, "output.parquet")
        ar.write_parquet(sample_frame, path, compression="zstd")
        assert os.path.exists(path)

    def test_write_parquet_compression_none(self, sample_frame, temp_dir):
        """Test write_parquet with no compression"""
        path = os.path.join(temp_dir, "output.parquet")
        ar.write_parquet(sample_frame, path, compression="none")
        assert os.path.exists(path)

    def test_write_parquet_row_group_size(self, sample_frame, temp_dir):
        """Test write_parquet with row_group_size parameter"""
        path = os.path.join(temp_dir, "output.parquet")
        ar.write_parquet(sample_frame, path, row_group_size=1024)
        assert os.path.exists(path)

    def test_write_parquet_preserve_attrs_false(self, sample_frame, temp_dir):
        """Test write_parquet with preserve_attrs=False"""
        path = os.path.join(temp_dir, "output.parquet")
        ar.write_parquet(sample_frame, path, preserve_attrs=False)
        assert os.path.exists(path)

    def test_write_parquet_invalid_extension(self, sample_frame, temp_dir):
        """Test ValueError for invalid file extension"""
        path = os.path.join(temp_dir, "output.csv")
        with pytest.raises(ValueError, match="Unsupported file format"):
            ar.write_parquet(sample_frame, path)

    def test_write_parquet_invalid_compression(self, sample_frame, temp_dir):
        """Test ValueError for unknown compression codec"""
        path = os.path.join(temp_dir, "output.parquet")
        with pytest.raises(ValueError, match="Unknown compression codec"):
            ar.write_parquet(sample_frame, path, compression="invalid")

    def test_write_parquet_invalid_row_group_size(self, sample_frame, temp_dir):
        """Test ValueError for non-positive row_group_size"""
        path = os.path.join(temp_dir, "output.parquet")
        with pytest.raises(ValueError, match="row_group_size must be a positive integer"):
            ar.write_parquet(sample_frame, path, row_group_size=0)

    def test_write_parquet_non_arframe(self, temp_dir):
        """Test TypeError when input is not an ArFrame"""
        path = os.path.join(temp_dir, "output.parquet")
        df = pd.DataFrame({'a': [1, 2, 3]})
        with pytest.raises(TypeError, match="frame must be an ArFrame"):
            ar.write_parquet(df, path)

    def test_write_parquet_invalid_path_type(self, sample_frame):
        """Test TypeError for invalid path type"""
        with pytest.raises(TypeError):
            ar.write_parquet(sample_frame, 12345)

    def test_write_parquet_non_json_attrs(self, temp_dir):
        """Test TypeError when attrs contain non-JSON values with preserve_attrs=True"""
        df = pd.DataFrame({'a': [1, 2, 3]})
        frame = ar.from_pandas(df)
        frame.attrs = {'key': set([1, 2, 3])}
        
        path = os.path.join(temp_dir, "output.parquet")
        with pytest.raises(TypeError, match="JSON-serializable"):
            ar.write_parquet(frame, path, preserve_attrs=True)

    def test_write_parquet_zero_columns_with_rows(self, temp_dir):
        """Test ValueError for zero-column frame with rows"""
        df = pd.DataFrame(index=range(5))
        frame = ar.from_pandas(df)
        
        path = os.path.join(temp_dir, "output.parquet")
        with pytest.raises(ValueError, match="Cannot write a zero-column ArFrame"):
            ar.write_parquet(frame, path)
