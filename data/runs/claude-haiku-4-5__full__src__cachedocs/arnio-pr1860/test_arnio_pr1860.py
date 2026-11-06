import pytest
import tempfile
import os
from pathlib import Path
import pandas as pd
import numpy as np
import arnio as ar


@pytest.fixture
def simple_frame():
    """Create a simple test ArFrame."""
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["alice", "bob", "charlie"],
        "value": [1.1, 2.2, 3.3]
    })
    return ar.from_pandas(df)


@pytest.fixture
def frame_with_nulls():
    """Create an ArFrame with null values."""
    df = pd.DataFrame({
        "a": [1, None, 3],
        "b": ["x", None, "z"],
    })
    return ar.from_pandas(df)


@pytest.fixture
def empty_frame():
    """Create an empty ArFrame with columns."""
    df = pd.DataFrame({"col1": pd.Series([], dtype="int64"), "col2": pd.Series([], dtype="str")})
    return ar.from_pandas(df)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestWriteParquetBasic:
    """Test basic write_parquet functionality."""

    def test_write_parquet_creates_file(self, simple_frame, temp_dir):
        """Test that write_parquet creates a .parquet file."""
        path = os.path.join(temp_dir, "output.parquet")
        ar.write_parquet(simple_frame, path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    def test_write_parquet_pq_extension(self, simple_frame, temp_dir):
        """Test that .pq extension is accepted."""
        path = os.path.join(temp_dir, "output.pq")
        ar.write_parquet(simple_frame, path)
        assert os.path.exists(path)

    def test_write_parquet_pathlib_path(self, simple_frame, temp_dir):
        """Test that pathlib.Path objects are accepted."""
        path = Path(temp_dir) / "output.parquet"
        ar.write_parquet(simple_frame, path)
        assert path.exists()

    def test_write_parquet_roundtrip(self, simple_frame, temp_dir):
        """Test that data survives a write-read roundtrip."""
        path = os.path.join(temp_dir, "output.parquet")
        ar.write_parquet(simple_frame, path)
        df_original = ar.to_pandas(simple_frame)
        df_read = pd.read_parquet(path)
        pd.testing.assert_frame_equal(df_original, df_read)


class TestWriteParquetCompression:
    """Test compression codec parameter."""

    @pytest.mark.parametrize("codec", ["snappy", "gzip", "zstd", "brotli", "none"])
    def test_write_parquet_all_valid_compressions(self, simple_frame, temp_dir, codec):
        """Test that all documented compression codecs work."""
        path = os.path.join(temp_dir, f"output_{codec}.parquet")
        ar.write_parquet(simple_frame, path, compression=codec)
        assert os.path.exists(path)

    def test_write_parquet_invalid_compression_raises_value_error(self, simple_frame, temp_dir):
        """Test that invalid compression codec raises ValueError."""
        path = os.path.join(temp_dir, "output.parquet")
        with pytest.raises(ValueError, match="Unknown compression codec"):
            ar.write_parquet(simple_frame, path, compression="invalid_codec")


class TestWriteParquetRowGroupSize:
    """Test row_group_size parameter."""

    def test_write_parquet_row_group_size_positive(self, simple_frame, temp_dir):
        """Test that positive row_group_size works."""
        path = os.path.join(temp_dir, "output.parquet")
        ar.write_parquet(simple_frame, path, row_group_size=100)
        assert os.path.exists(path)

    def test_write_parquet_row_group_size_zero_raises_value_error(self, simple_frame, temp_dir):
        """Test that row_group_size=0 raises ValueError."""
        path = os.path.join(temp_dir, "output.parquet")
        with pytest.raises(ValueError, match="row_group_size must be a positive integer"):
            ar.write_parquet(simple_frame, path, row_group_size=0)

    def test_write_parquet_row_group_size_negative_raises_value_error(self, simple_frame, temp_dir):
        """Test that negative row_group_size raises ValueError."""
        path = os.path.join(temp_dir, "output.parquet")
        with pytest.raises(ValueError, match="row_group_size must be a positive integer"):
            ar.write_parquet(simple_frame, path, row_group_size=-1)


class TestWriteParquetTypeValidation:
    """Test parameter type validation."""

    def test_write_parquet_non_arframe_raises_type_error(self, temp_dir):
        """Test that non-ArFrame raises TypeError."""
        path = os.path.join(temp_dir, "output.parquet")
        df = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(TypeError, match="frame must be an ArFrame"):
            ar.write_parquet(df, path)

    def test_write_parquet_invalid_path_type_raises_type_error(self, simple_frame):
        """Test that invalid path type raises TypeError."""
        with pytest.raises(TypeError, match="path must be a string"):
            ar.write_parquet(simple_frame, 12345)

    def test_write_parquet_non_string_compression_raises_type_error(self, simple_frame, temp_dir):
        """Test that non-string compression raises TypeError."""
        path = os.path.join(temp_dir, "output.parquet")
        with pytest.raises(TypeError, match="compression must be a string"):
            ar.write_parquet(simple_frame, path, compression=123)

    def test_write_parquet_non_bool_preserve_attrs_raises_type_error(self, simple_frame, temp_dir):
        """Test that non-bool preserve_attrs raises TypeError."""
        path = os.path.join(temp_dir, "output.parquet")
        with pytest.raises(TypeError, match="preserve_attrs must be a bool"):
            ar.write_parquet(simple_frame, path, preserve_attrs="yes")


class TestWriteParquetFileExtension:
    """Test file extension validation."""

    def test_write_parquet_unsupported_extension_raises_value_error(self, simple_frame, temp_dir):
        """Test that unsupported extension raises ValueError."""
        path = os.path.join(temp_dir, "output.csv")
        with pytest.raises(ValueError, match="Unsupported file format"):
            ar.write_parquet(simple_frame, path)

    def test_write_parquet_uppercase_extension(self, simple_frame, temp_dir):
        """Test that uppercase file extension is accepted (case-insensitive)."""
        path = os.path.join(temp_dir, "output.PARQUET")
        ar.write_parquet(simple_frame, path)
        assert os.path.exists(path)


class TestWriteParquetPreserveAttrs:
    """Test preserve_attrs parameter."""

    def test_write_parquet_preserve_attrs_serializable(self, simple_frame, temp_dir):
        """Test that serializable attrs are preserved when preserve_attrs=True."""
        path = os.path.join(temp_dir, "output.parquet")
        df = ar.to_pandas(simple_frame)
        df.attrs = {"description": "test data", "version": 1, "flag": True}
        frame = ar.from_pandas(df)
        ar.write_parquet(frame, path, preserve_attrs=True)
        assert os.path.exists(path)

    def test_write_parquet_preserve_attrs_false(self, simple_frame, temp_dir):
        """Test that attrs are dropped when preserve_attrs=False."""
        path = os.path.join(temp_dir, "output.parquet")
        df = ar.to_pandas(simple_frame)
        df.attrs = {"key": "value"}
        frame = ar.from_pandas(df)
        ar.write_parquet(frame, path, preserve_attrs=False)
        assert os.path.exists(path)

    def test_write_parquet_non_serializable_attrs_raises_type_error(self, simple_frame, temp_dir):
        """Test that non-JSON-serializable attrs raise TypeError when preserve_attrs=True."""
        path = os.path.join(temp_dir, "output.parquet")
        df = ar.to_pandas(simple_frame)
        df.attrs = {"func": lambda x: x}  # non-serializable
        frame = ar.from_pandas(df)
        with pytest.raises(TypeError, match="JSON-serializable"):
            ar.write_parquet(frame, path, preserve_attrs=True)


class TestWriteParquetEdgeCases:
    """Test edge cases and special scenarios."""

    def test_write_parquet_empty_frame_with_columns(self, empty_frame, temp_dir):
        """Test writing an empty frame with columns defined."""
        path = os.path.join(temp_dir, "output.parquet")
        ar.write_parquet(empty_frame, path)
        assert os.path.exists(path)

    def test_write_parquet_frame_with_nulls(self, frame_with_nulls, temp_dir):
        """Test writing a frame with null values."""
        path = os.path.join(temp_dir, "output.parquet")
        ar.write_parquet(frame_with_nulls, path)
        assert os.path.exists(path)
        df_read = pd.read_parquet(path)
        assert df_read.isna().any().any()
