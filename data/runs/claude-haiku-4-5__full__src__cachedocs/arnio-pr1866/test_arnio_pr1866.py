import pytest
import tempfile
import os
from pathlib import Path
import pandas as pd
import numpy as np
import arnio as ar


@pytest.fixture
def test_frame():
    """Create a simple test ArFrame."""
    df = pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35],
        "score": [85.5, 90.0, 88.2]
    })
    return ar.from_pandas(df)


class TestWriteParquetBasic:
    """Test basic write_parquet functionality."""

    def test_write_parquet_to_parquet_extension(self, test_frame):
        """Test writing to .parquet file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.parquet")
            ar.write_parquet(test_frame, path)
            
            assert os.path.exists(path)
            result = ar.read_parquet(path)
            assert result.shape == test_frame.shape
            assert result.columns == test_frame.columns

    def test_write_parquet_to_pq_extension(self, test_frame):
        """Test writing to .pq file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.pq")
            ar.write_parquet(test_frame, path)
            
            assert os.path.exists(path)
            result = ar.read_parquet(path)
            assert result.shape == test_frame.shape

    def test_write_parquet_pathlib_path(self, test_frame):
        """Test writing with pathlib.Path object."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "output.parquet"
            ar.write_parquet(test_frame, path)
            
            assert path.exists()


class TestWriteParquetCompression:
    """Test compression codec parameter."""

    @pytest.mark.parametrize("codec", ["snappy", "gzip", "zstd", "none"])
    def test_write_parquet_valid_compression_codecs(self, test_frame, codec):
        """Test valid compression codecs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, f"output_{codec}.parquet")
            ar.write_parquet(test_frame, path, compression=codec)
            
            assert os.path.exists(path)
            result = ar.read_parquet(path)
            assert result.shape == test_frame.shape

    def test_write_parquet_invalid_compression_codec(self, test_frame):
        """Test that invalid compression codec raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.parquet")
            
            with pytest.raises(ValueError, match="Unknown compression codec"):
                ar.write_parquet(test_frame, path, compression="invalid_codec")


class TestWriteParquetRowGroupSize:
    """Test row_group_size parameter."""

    def test_write_parquet_custom_row_group_size(self, test_frame):
        """Test writing with custom row group size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.parquet")
            ar.write_parquet(test_frame, path, row_group_size=1000)
            
            assert os.path.exists(path)

    def test_write_parquet_row_group_size_zero_raises(self, test_frame):
        """Test that zero row_group_size raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.parquet")
            
            with pytest.raises(ValueError, match="positive integer"):
                ar.write_parquet(test_frame, path, row_group_size=0)

    def test_write_parquet_row_group_size_negative_raises(self, test_frame):
        """Test that negative row_group_size raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.parquet")
            
            with pytest.raises(ValueError, match="positive integer"):
                ar.write_parquet(test_frame, path, row_group_size=-100)

    def test_write_parquet_row_group_size_non_integer_raises(self, test_frame):
        """Test that non-integer row_group_size raises TypeError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.parquet")
            
            with pytest.raises(TypeError, match="integer"):
                ar.write_parquet(test_frame, path, row_group_size=10.5)


class TestWriteParquetPreserveAttrs:
    """Test preserve_attrs parameter."""

    def test_write_parquet_preserve_attrs_true_valid(self, test_frame):
        """Test preserve_attrs=True with JSON-serializable attrs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.parquet")
            
            df = ar.to_pandas(test_frame)
            df.attrs = {"source": "test", "version": 1, "flag": True}
            frame = ar.from_pandas(df)
            
            ar.write_parquet(frame, path, preserve_attrs=True)
            assert os.path.exists(path)

    def test_write_parquet_preserve_attrs_false(self, test_frame):
        """Test preserve_attrs=False ignores attrs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.parquet")
            
            df = ar.to_pandas(test_frame)
            df.attrs = {"source": "test"}
            frame = ar.from_pandas(df)
            
            ar.write_parquet(frame, path, preserve_attrs=False)
            assert os.path.exists(path)

    def test_write_parquet_preserve_attrs_non_serializable_raises(self, test_frame):
        """Test that non-JSON-serializable attrs raise TypeError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.parquet")
            
            df = ar.to_pandas(test_frame)
            df.attrs = {"obj": object()}  # Non-serializable
            frame = ar.from_pandas(df)
            
            with pytest.raises(TypeError, match="JSON-serializable"):
                ar.write_parquet(frame, path, preserve_attrs=True)

    def test_write_parquet_preserve_attrs_non_bool_raises(self, test_frame):
        """Test that non-boolean preserve_attrs raises TypeError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.parquet")
            
            with pytest.raises(TypeError, match="preserve_attrs must be a bool"):
                ar.write_parquet(test_frame, path, preserve_attrs="true")


class TestWriteParquetValidation:
    """Test input validation."""

    def test_write_parquet_non_arframe_raises(self):
        """Test that non-ArFrame raises TypeError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.parquet")
            df = pd.DataFrame({"a": [1, 2, 3]})
            
            with pytest.raises(TypeError, match="frame must be an ArFrame"):
                ar.write_parquet(df, path)

    def test_write_parquet_invalid_path_type_raises(self, test_frame):
        """Test that invalid path type raises TypeError."""
        with pytest.raises(TypeError, match="path must be"):
            ar.write_parquet(test_frame, 12345)

    def test_write_parquet_invalid_extension_csv_raises(self, test_frame):
        """Test that .csv extension raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.csv")
            
            with pytest.raises(ValueError, match="Unsupported file format"):
                ar.write_parquet(test_frame, path)

    def test_write_parquet_invalid_extension_txt_raises(self, test_frame):
        """Test that .txt extension raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.txt")
            
            with pytest.raises(ValueError, match="Unsupported file format"):
                ar.write_parquet(test_frame, path)

    def test_write_parquet_empty_frame(self):
        """Test writing an empty frame."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.parquet")
            df = pd.DataFrame({"a": pd.Series(dtype="int64"), "b": pd.Series(dtype="str")})
            frame = ar.from_pandas(df)
            
            ar.write_parquet(frame, path)
            assert os.path.exists(path)
