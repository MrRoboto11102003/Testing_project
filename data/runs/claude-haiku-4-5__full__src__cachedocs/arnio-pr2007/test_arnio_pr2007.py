import pytest
import tempfile
import os
from pathlib import Path
import arnio as ar
import pandas as pd


@pytest.fixture
def simple_frame():
    """Create a simple ArFrame for testing."""
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["alice", "bob", "charlie"],
        "value": [10.5, 20.3, 30.1],
    })
    return ar.from_pandas(df)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestWriteParquetBasic:
    """Basic write functionality tests."""

    def test_write_parquet_parquet_extension(self, simple_frame, temp_dir):
        """Test writing to a .parquet file."""
        path = os.path.join(temp_dir, "output.parquet")
        ar.write_parquet(simple_frame, path)
        assert os.path.exists(path)

    def test_write_parquet_pq_extension(self, simple_frame, temp_dir):
        """Test writing to a .pq file."""
        path = os.path.join(temp_dir, "output.pq")
        ar.write_parquet(simple_frame, path)
        assert os.path.exists(path)

    def test_write_parquet_pathlike(self, simple_frame, temp_dir):
        """Test writing with pathlib.Path."""
        path = Path(temp_dir) / "output.parquet"
        ar.write_parquet(simple_frame, path)
        assert path.exists()


class TestWriteParquetCompression:
    """Compression parameter tests."""

    @pytest.mark.parametrize("compression", ["snappy", "gzip", "zstd", "brotli", "none"])
    def test_compression_valid_codecs(self, simple_frame, temp_dir, compression):
        """Test all valid compression codecs."""
        path = os.path.join(temp_dir, "output.parquet")
        ar.write_parquet(simple_frame, path, compression=compression)
        assert os.path.exists(path)

    def test_compression_invalid_codec(self, simple_frame, temp_dir):
        """Test invalid compression codec raises ValueError."""
        path = os.path.join(temp_dir, "output.parquet")
        with pytest.raises(ValueError, match="Unknown compression codec"):
            ar.write_parquet(simple_frame, path, compression="invalid")

    def test_compression_not_string(self, simple_frame, temp_dir):
        """Test non-string compression raises TypeError."""
        path = os.path.join(temp_dir, "output.parquet")
        with pytest.raises(TypeError, match="compression must be a string"):
            ar.write_parquet(simple_frame, path, compression=123)


class TestWriteParquetRowGroupSize:
    """Row group size parameter tests."""

    def test_row_group_size_valid(self, simple_frame, temp_dir):
        """Test valid positive row_group_size."""
        path = os.path.join(temp_dir, "output.parquet")
        ar.write_parquet(simple_frame, path, row_group_size=1024)
        assert os.path.exists(path)

    def test_row_group_size_zero_raises_error(self, simple_frame, temp_dir):
        """Test row_group_size=0 raises ValueError."""
        path = os.path.join(temp_dir, "output.parquet")
        with pytest.raises(ValueError, match="positive integer"):
            ar.write_parquet(simple_frame, path, row_group_size=0)

    def test_row_group_size_negative_raises_error(self, simple_frame, temp_dir):
        """Test negative row_group_size raises ValueError."""
        path = os.path.join(temp_dir, "output.parquet")
        with pytest.raises(ValueError, match="positive integer"):
            ar.write_parquet(simple_frame, path, row_group_size=-100)

    def test_row_group_size_float_raises_error(self, simple_frame, temp_dir):
        """Test float row_group_size raises TypeError."""
        path = os.path.join(temp_dir, "output.parquet")
        with pytest.raises(TypeError, match="integer"):
            ar.write_parquet(simple_frame, path, row_group_size=1.5)

    def test_row_group_size_bool_raises_error(self, simple_frame, temp_dir):
        """Test bool row_group_size raises TypeError (bool is subclass of int)."""
        path = os.path.join(temp_dir, "output.parquet")
        with pytest.raises(TypeError, match="integer"):
            ar.write_parquet(simple_frame, path, row_group_size=True)


class TestWriteParquetPreserveAttrs:
    """Preserve attrs parameter tests."""

    def test_preserve_attrs_false(self, simple_frame, temp_dir):
        """Test preserve_attrs=False."""
        path = os.path.join(temp_dir, "output.parquet")
        ar.write_parquet(simple_frame, path, preserve_attrs=False)
        assert os.path.exists(path)

    def test_preserve_attrs_not_bool(self, simple_frame, temp_dir):
        """Test non-bool preserve_attrs raises TypeError."""
        path = os.path.join(temp_dir, "output.parquet")
        with pytest.raises(TypeError, match="preserve_attrs must be a bool"):
            ar.write_parquet(simple_frame, path, preserve_attrs="yes")


class TestWriteParquetErrors:
    """Error handling tests."""

    def test_frame_not_arframe(self, temp_dir):
        """Test non-ArFrame input raises TypeError."""
        path = os.path.join(temp_dir, "output.parquet")
        df = pd.DataFrame({"col": [1, 2, 3]})
        with pytest.raises(TypeError, match="ArFrame"):
            ar.write_parquet(df, path)

    def test_invalid_file_extension(self, simple_frame, temp_dir):
        """Test invalid file extension raises ValueError."""
        path = os.path.join(temp_dir, "output.csv")
        with pytest.raises(ValueError, match="Unsupported file format"):
            ar.write_parquet(simple_frame, path)

    def test_path_not_string_or_pathlike(self, simple_frame):
        """Test non-string/PathLike path raises TypeError."""
        with pytest.raises(TypeError, match="path must be"):
            ar.write_parquet(simple_frame, 123)

    def test_zero_column_frame_with_rows(self, temp_dir):
        """Test zero-column frame with rows raises ValueError."""
        path = os.path.join(temp_dir, "output.parquet")
        # Create a DataFrame with index (rows) but no columns
        df = pd.DataFrame(index=[1, 2, 3])
        frame = ar.from_pandas(df)
        with pytest.raises(ValueError, match="Cannot write a zero-column"):
            ar.write_parquet(frame, path)
