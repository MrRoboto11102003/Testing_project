import pytest
import pandas as pd
import tempfile
import os
from pathlib import Path

import arnio as ar


class TestWriteParquetBasic:
    """Test basic write_parquet functionality."""

    def test_write_parquet_basic(self):
        """Test writing a basic ArFrame to parquet."""
        data = {"id": [1, 2, 3], "name": ["a", "b", "c"]}
        frame = ar.from_dict(data)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.parquet")
            ar.write_parquet(frame, path)
            assert os.path.exists(path)
            df = pd.read_parquet(path)
            assert len(df) == 3
            assert list(df.columns) == ["id", "name"]

    def test_write_parquet_pq_extension(self):
        """Test writing with .pq extension."""
        data = {"id": [1, 2, 3]}
        frame = ar.from_dict(data)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.pq")
            ar.write_parquet(frame, path)
            assert os.path.exists(path)
            df = pd.read_parquet(path)
            assert len(df) == 3


class TestWriteParquetCompression:
    """Test compression parameter."""

    @pytest.mark.parametrize("compression", ["snappy", "gzip", "brotli", "zstd", "none"])
    def test_valid_compressions(self, compression):
        """Test all valid compression codecs."""
        data = {"value": [1.0, 2.0, 3.0]}
        frame = ar.from_dict(data)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.parquet")
            ar.write_parquet(frame, path, compression=compression)
            assert os.path.exists(path)

    def test_invalid_compression(self):
        """Test that invalid compression raises ValueError."""
        frame = ar.from_dict({"id": [1, 2, 3]})
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.parquet")
            with pytest.raises(ValueError, match="Unknown compression codec"):
                ar.write_parquet(frame, path, compression="invalid_codec")

    def test_compression_not_string(self):
        """Test that non-string compression raises TypeError."""
        frame = ar.from_dict({"id": [1, 2, 3]})
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.parquet")
            with pytest.raises(TypeError, match="compression must be a string"):
                ar.write_parquet(frame, path, compression=123)


class TestWriteParquetRowGroupSize:
    """Test row_group_size parameter."""

    def test_valid_row_group_size(self):
        """Test with valid row_group_size."""
        frame = ar.from_dict({"id": list(range(1000))})
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.parquet")
            ar.write_parquet(frame, path, row_group_size=100)
            assert os.path.exists(path)

    def test_row_group_size_none_default(self):
        """Test that row_group_size=None works (default behavior)."""
        frame = ar.from_dict({"id": [1, 2, 3]})
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.parquet")
            ar.write_parquet(frame, path, row_group_size=None)
            assert os.path.exists(path)

    def test_row_group_size_zero(self):
        """Test that row_group_size=0 raises ValueError."""
        frame = ar.from_dict({"id": [1, 2, 3]})
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.parquet")
            with pytest.raises(ValueError, match="row_group_size must be a positive integer"):
                ar.write_parquet(frame, path, row_group_size=0)

    def test_row_group_size_negative(self):
        """Test that negative row_group_size raises ValueError."""
        frame = ar.from_dict({"id": [1, 2, 3]})
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.parquet")
            with pytest.raises(ValueError, match="row_group_size must be a positive integer"):
                ar.write_parquet(frame, path, row_group_size=-1)

    def test_row_group_size_not_integer(self):
        """Test that non-integer row_group_size raises TypeError."""
        frame = ar.from_dict({"id": [1, 2, 3]})
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.parquet")
            with pytest.raises(TypeError, match="row_group_size must be an integer"):
                ar.write_parquet(frame, path, row_group_size=1.5)


class TestWriteParquetPreserveAttrs:
    """Test preserve_attrs parameter."""

    def test_preserve_attrs_default_true(self):
        """Test that attrs are preserved by default."""
        df = pd.DataFrame({"id": [1, 2, 3]})
        df.attrs = {"version": 1, "source": "test"}
        frame = ar.from_pandas(df)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.parquet")
            ar.write_parquet(frame, path)
            assert os.path.exists(path)

    def test_preserve_attrs_false_drops_attrs(self):
        """Test that preserve_attrs=False works."""
        df = pd.DataFrame({"id": [1, 2, 3]})
        df.attrs = {"version": 1}
        frame = ar.from_pandas(df)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.parquet")
            ar.write_parquet(frame, path, preserve_attrs=False)
            assert os.path.exists(path)

    def test_preserve_attrs_non_serializable_raises_error(self):
        """Test that non-JSON-serializable attrs raise TypeError."""
        df = pd.DataFrame({"id": [1, 2, 3]})
        df.attrs = {"func": lambda x: x}  # Non-serializable
        frame = ar.from_pandas(df)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.parquet")
            with pytest.raises(TypeError, match="JSON-serializable"):
                ar.write_parquet(frame, path, preserve_attrs=True)

    def test_preserve_attrs_not_boolean(self):
        """Test that non-boolean preserve_attrs raises TypeError."""
        frame = ar.from_dict({"id": [1, 2, 3]})
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.parquet")
            with pytest.raises(TypeError, match="preserve_attrs must be a bool"):
                ar.write_parquet(frame, path, preserve_attrs="yes")


class TestWriteParquetPathValidation:
    """Test path validation."""

    def test_invalid_extension(self):
        """Test that invalid file extension raises ValueError."""
        frame = ar.from_dict({"id": [1, 2, 3]})
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.csv")
            with pytest.raises(ValueError, match="Unsupported file format"):
                ar.write_parquet(frame, path)

    def test_case_insensitive_extension(self):
        """Test that extension check is case-insensitive."""
        frame = ar.from_dict({"id": [1, 2, 3]})
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.PARQUET")
            ar.write_parquet(frame, path)
            assert os.path.exists(path)

    def test_pathlib_path(self):
        """Test that pathlib.Path is accepted."""
        frame = ar.from_dict({"id": [1, 2, 3]})
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "output.parquet"
            ar.write_parquet(frame, path)
            assert path.exists()

    def test_path_not_path_like(self):
        """Test that non-path-like object raises TypeError."""
        frame = ar.from_dict({"id": [1, 2, 3]})
        
        with pytest.raises(TypeError, match="path must be"):
            ar.write_parquet(frame, 123)


class TestWriteParquetInputValidation:
    """Test input validation."""

    def test_frame_not_arframe(self):
        """Test that non-ArFrame input raises TypeError."""
        df = pd.DataFrame({"id": [1, 2, 3]})
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.parquet")
            with pytest.raises(TypeError, match="frame must be an ArFrame"):
                ar.write_parquet(df, path)


class TestWriteParquetEdgeCases:
    """Test edge cases."""

    def test_empty_frame(self):
        """Test writing an empty frame (0 rows)."""
        frame = ar.from_dict({"id": [], "name": []})
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.parquet")
            ar.write_parquet(frame, path)
            assert os.path.exists(path)
            df = pd.read_parquet(path)
            assert len(df) == 0

    def test_numeric_data_types(self):
        """Test writing frame with various numeric types."""
        frame = ar.from_dict({
            "int_col": [1, 2, 3],
            "float_col": [1.5, 2.5, 3.5],
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.parquet")
            ar.write_parquet(frame, path)
            df = pd.read_parquet(path)
            assert len(df) == 3
