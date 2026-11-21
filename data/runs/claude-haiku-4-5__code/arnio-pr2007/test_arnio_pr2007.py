import pytest
import pandas as pd
import tempfile
import os
from pathlib import Path

import arnio as ar


class TestWriteParquet:
    """Tests for arnio.io.write_parquet function."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def sample_frame(self):
        """Create a sample ArFrame for testing."""
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4.0, 5.0, 6.0],
            'c': ['x', 'y', 'z']
        })
        return ar.ArFrame(df)
    
    @pytest.fixture
    def frame_with_attrs(self):
        """Create an ArFrame with JSON-serializable attrs."""
        df = pd.DataFrame({'a': [1, 2, 3]})
        frame = ar.ArFrame(df)
        frame.attrs = {'key': 'value', 'count': 42, 'enabled': True}
        return frame
    
    # Basic functionality tests
    def test_write_parquet_basic(self, sample_frame, temp_dir):
        """Test basic writing to parquet file with .parquet extension."""
        path = os.path.join(temp_dir, 'output.parquet')
        ar.write_parquet(sample_frame, path)
        assert os.path.exists(path)
    
    def test_write_parquet_with_pq_extension(self, sample_frame, temp_dir):
        """Test writing with .pq extension."""
        path = os.path.join(temp_dir, 'output.pq')
        ar.write_parquet(sample_frame, path)
        assert os.path.exists(path)
    
    def test_write_parquet_with_pathlike(self, sample_frame, temp_dir):
        """Test writing with Path object."""
        path = Path(temp_dir) / 'output.parquet'
        ar.write_parquet(sample_frame, path)
        assert path.exists()
    
    # Compression tests
    def test_write_parquet_compression_zstd(self, sample_frame, temp_dir):
        """Test with zstd compression codec."""
        path = os.path.join(temp_dir, 'output.parquet')
        ar.write_parquet(sample_frame, path, compression='zstd')
        assert os.path.exists(path)
    
    def test_write_parquet_invalid_compression(self, sample_frame, temp_dir):
        """Test with invalid compression codec raises ValueError."""
        path = os.path.join(temp_dir, 'output.parquet')
        with pytest.raises(ValueError, match="Unknown compression codec"):
            ar.write_parquet(sample_frame, path, compression='invalid')
    
    def test_write_parquet_compression_not_string(self, sample_frame, temp_dir):
        """Test that non-string compression raises TypeError."""
        path = os.path.join(temp_dir, 'output.parquet')
        with pytest.raises(TypeError, match="compression must be a string"):
            ar.write_parquet(sample_frame, path, compression=123)
    
    # row_group_size tests
    def test_write_parquet_row_group_size(self, sample_frame, temp_dir):
        """Test with valid row_group_size parameter."""
        path = os.path.join(temp_dir, 'output.parquet')
        ar.write_parquet(sample_frame, path, row_group_size=1000)
        assert os.path.exists(path)
    
    def test_write_parquet_row_group_size_zero(self, sample_frame, temp_dir):
        """Test that row_group_size=0 raises ValueError."""
        path = os.path.join(temp_dir, 'output.parquet')
        with pytest.raises(ValueError, match="row_group_size must be a positive integer"):
            ar.write_parquet(sample_frame, path, row_group_size=0)
    
    def test_write_parquet_row_group_size_bool(self, sample_frame, temp_dir):
        """Test that bool is not accepted for row_group_size."""
        path = os.path.join(temp_dir, 'output.parquet')
        with pytest.raises(TypeError, match="row_group_size must be an integer"):
            ar.write_parquet(sample_frame, path, row_group_size=True)
    
    # preserve_attrs tests
    def test_write_parquet_preserve_attrs_true(self, frame_with_attrs, temp_dir):
        """Test with preserve_attrs=True writes metadata."""
        path = os.path.join(temp_dir, 'output.parquet')
        ar.write_parquet(frame_with_attrs, path, preserve_attrs=True)
        assert os.path.exists(path)
    
    def test_write_parquet_preserve_attrs_not_bool(self, sample_frame, temp_dir):
        """Test that non-boolean preserve_attrs raises TypeError."""
        path = os.path.join(temp_dir, 'output.parquet')
        with pytest.raises(TypeError, match="preserve_attrs must be a bool"):
            ar.write_parquet(sample_frame, path, preserve_attrs='true')
    
    def test_write_parquet_non_serializable_attrs(self, temp_dir):
        """Test that non-JSON-serializable attrs raises TypeError."""
        df = pd.DataFrame({'a': [1, 2, 3]})
        frame = ar.ArFrame(df)
        frame.attrs = {'func': lambda x: x}
        path = os.path.join(temp_dir, 'output.parquet')
        with pytest.raises(TypeError, match="JSON-serializable"):
            ar.write_parquet(frame, path, preserve_attrs=True)
    
    # Path validation tests
    def test_write_parquet_invalid_extension(self, sample_frame, temp_dir):
        """Test that invalid file extension raises ValueError."""
        path = os.path.join(temp_dir, 'output.csv')
        with pytest.raises(ValueError, match="Unsupported file format"):
            ar.write_parquet(sample_frame, path)
    
    # Frame validation tests
    def test_write_parquet_non_arframe(self, temp_dir):
        """Test that non-ArFrame raises TypeError."""
        df = pd.DataFrame({'a': [1, 2, 3]})
        path = os.path.join(temp_dir, 'output.parquet')
        with pytest.raises(TypeError, match="frame must be an ArFrame"):
            ar.write_parquet(df, path)
    
    def test_write_parquet_zero_cols_with_rows(self, temp_dir):
        """Test that zero-column frame with rows raises ValueError."""
        df = pd.DataFrame(index=range(5))
        frame = ar.ArFrame(df)
        path = os.path.join(temp_dir, 'output.parquet')
        with pytest.raises(ValueError, match="Cannot write a zero-column ArFrame"):
            ar.write_parquet(frame, path)
