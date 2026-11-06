import pytest
import pandas as pd
import os
import tempfile
from pathlib import Path
import arnio as ar


class TestWriteParquetBasic:
    """Basic write_parquet functionality."""

    def test_write_parquet_basic(self):
        """Test writing a simple ArFrame to parquet."""
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
            frame = ar.from_pandas(df)
            path = os.path.join(tmpdir, "output.parquet")
            ar.write_parquet(frame, path)
            
            assert os.path.exists(path)
            result = pd.read_parquet(path)
            pd.testing.assert_frame_equal(result, df)

    def test_write_parquet_pq_extension(self):
        """Test .pq file extension is accepted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
            path = os.path.join(tmpdir, "output.pq")
            ar.write_parquet(frame, path)
            
            assert os.path.exists(path)

    def test_write_parquet_pathlib_path(self):
        """Test pathlib.Path objects are accepted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
            path = Path(tmpdir) / "output.parquet"
            ar.write_parquet(frame, path)
            
            assert path.exists()

    def test_write_parquet_empty_frame(self):
        """Test writing an empty frame with columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            frame = ar.from_pandas(pd.DataFrame({"a": [], "b": []}))
            path = os.path.join(tmpdir, "empty.parquet")
            ar.write_parquet(frame, path)
            
            result = pd.read_parquet(path)
            assert len(result) == 0
            assert list(result.columns) == ["a", "b"]


class TestWriteParquetCompression:
    """Compression codec handling."""

    @pytest.mark.parametrize("codec", ["snappy", "gzip", "zstd", "brotli", "none"])
    def test_write_parquet_compression_codecs(self, codec):
        """Test all supported compression codecs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
            path = os.path.join(tmpdir, f"output.parquet")
            ar.write_parquet(frame, path, compression=codec)
            
            assert os.path.exists(path)
            result = pd.read_parquet(path)
            assert len(result) == 3


class TestWriteParquetRowGroupSize:
    """Row group size parameter handling."""

    def test_write_parquet_row_group_size(self):
        """Test setting custom row_group_size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            frame = ar.from_pandas(pd.DataFrame({"a": range(100)}))
            path = os.path.join(tmpdir, "output.parquet")
            ar.write_parquet(frame, path, row_group_size=25)
            
            assert os.path.exists(path)
            result = pd.read_parquet(path)
            assert len(result) == 100


class TestWriteParquetAttributes:
    """DataFrame.attrs preservation."""

    def test_write_parquet_preserve_attrs_true(self):
        """Test preserving JSON-serializable attrs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({"a": [1, 2, 3]})
            df.attrs = {"version": "1.0", "count": 3}
            frame = ar.from_pandas(df)
            path = os.path.join(tmpdir, "output.parquet")
            ar.write_parquet(frame, path, preserve_attrs=True)
            
            assert os.path.exists(path)

    def test_write_parquet_preserve_attrs_false(self):
        """Test dropping attrs with preserve_attrs=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({"a": [1, 2, 3]})
            df.attrs = {"version": "1.0"}
            frame = ar.from_pandas(df)
            path = os.path.join(tmpdir, "output.parquet")
            ar.write_parquet(frame, path, preserve_attrs=False)
            
            assert os.path.exists(path)

    def test_write_parquet_non_serializable_attrs_raises(self):
        """Test that non-JSON-serializable attrs raise TypeError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({"a": [1, 2, 3]})
            df.attrs = {"obj": object()}
            frame = ar.from_pandas(df)
            path = os.path.join(tmpdir, "output.parquet")
            
            with pytest.raises(TypeError, match="JSON-serializable"):
                ar.write_parquet(frame, path, preserve_attrs=True)


class TestWriteParquetErrors:
    """Error condition handling."""

    def test_write_parquet_not_arframe_raises(self):
        """Test TypeError when frame is not ArFrame."""
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({"a": [1, 2, 3]})
            path = os.path.join(tmpdir, "output.parquet")
            
            with pytest.raises(TypeError, match="ArFrame"):
                ar.write_parquet(df, path)

    def test_write_parquet_invalid_extension_raises(self):
        """Test ValueError for unsupported file extensions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
            
            with pytest.raises(ValueError, match="Unsupported file format"):
                ar.write_parquet(frame, os.path.join(tmpdir, "output.csv"))

    def test_write_parquet_invalid_compression_raises(self):
        """Test ValueError for unknown compression codec."""
        with tempfile.TemporaryDirectory() as tmpdir:
            frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
            path = os.path.join(tmpdir, "output.parquet")
            
            with pytest.raises(ValueError, match="Unknown compression codec"):
                ar.write_parquet(frame, path, compression="invalid")

    def test_write_parquet_compression_not_string_raises(self):
        """Test TypeError when compression is not a string."""
        with tempfile.TemporaryDirectory() as tmpdir:
            frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
            path = os.path.join(tmpdir, "output.parquet")
            
            with pytest.raises(TypeError, match="compression must be a string"):
                ar.write_parquet(frame, path, compression=123)

    def test_write_parquet_row_group_size_zero_raises(self):
        """Test ValueError when row_group_size is not positive."""
        with tempfile.TemporaryDirectory() as tmpdir:
            frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
            path = os.path.join(tmpdir, "output.parquet")
            
            with pytest.raises(ValueError, match="positive integer"):
                ar.write_parquet(frame, path, row_group_size=0)

    def test_write_parquet_row_group_size_bool_raises(self):
        """Test TypeError when row_group_size is a bool."""
        with tempfile.TemporaryDirectory() as tmpdir:
            frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
            path = os.path.join(tmpdir, "output.parquet")
            
            with pytest.raises(TypeError, match="row_group_size must be an integer"):
                ar.write_parquet(frame, path, row_group_size=True)

    def test_write_parquet_preserve_attrs_not_bool_raises(self):
        """Test TypeError when preserve_attrs is not a bool."""
        with tempfile.TemporaryDirectory() as tmpdir:
            frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
            path = os.path.join(tmpdir, "output.parquet")
            
            with pytest.raises(TypeError, match="preserve_attrs must be a bool"):
                ar.write_parquet(frame, path, preserve_attrs="yes")
