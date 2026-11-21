import pytest
import os
import tempfile
from pathlib import Path
import pandas as pd

import arnio as ar


class TestWriteParquetBasicFunctionality:
    """Test basic write_parquet functionality."""

    def test_write_parquet_creates_file(self):
        """Verify write_parquet creates a parquet file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.parquet")
            frame = ar.from_dict({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
            ar.write_parquet(frame, path)
            assert os.path.exists(path)

    def test_write_parquet_pq_extension(self):
        """Verify write_parquet accepts .pq extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.pq")
            frame = ar.from_dict({"col": [1, 2, 3]})
            ar.write_parquet(frame, path)
            assert os.path.exists(path)

    def test_write_parquet_pathlib_path(self):
        """Verify write_parquet accepts pathlib.Path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.parquet"
            frame = ar.from_dict({"col": [1, 2, 3]})
            ar.write_parquet(frame, path)
            assert path.exists()

    def test_write_parquet_round_trip(self):
        """Verify data integrity after write and read."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.parquet")
            original = ar.from_dict({
                "ints": [1, 2, 3],
                "floats": [1.1, 2.2, 3.3],
                "strings": ["x", "y", "z"]
            })
            ar.write_parquet(original, path)
            loaded = ar.read_parquet(path)
            assert loaded.shape == original.shape
            assert loaded.columns == original.columns


class TestWriteParquetCompressionParameter:
    """Test compression parameter."""

    @pytest.mark.parametrize("compression", ["snappy", "gzip", "brotli", "zstd", "none"])
    def test_write_parquet_valid_compressions(self, compression):
        """Verify all documented compression codecs are accepted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, f"test_{compression}.parquet")
            frame = ar.from_dict({"col": list(range(100))})
            ar.write_parquet(frame, path, compression=compression)
            assert os.path.exists(path)

    def test_write_parquet_invalid_compression_raises_valueerror(self):
        """Verify invalid compression codec raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.parquet")
            frame = ar.from_dict({"col": [1, 2, 3]})
            with pytest.raises(ValueError, match="Unknown compression codec"):
                ar.write_parquet(frame, path, compression="deflate")

    def test_write_parquet_compression_type_error(self):
        """Verify compression must be a string."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.parquet")
            frame = ar.from_dict({"col": [1, 2, 3]})
            with pytest.raises(TypeError, match="compression must be a string"):
                ar.write_parquet(frame, path, compression=123)


class TestWriteParquetRowGroupSize:
    """Test row_group_size parameter."""

    def test_write_parquet_with_row_group_size(self):
        """Verify row_group_size parameter is accepted with positive integer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.parquet")
            frame = ar.from_dict({"col": list(range(1000))})
            ar.write_parquet(frame, path, row_group_size=100)
            assert os.path.exists(path)

    def test_write_parquet_row_group_size_zero_raises(self):
        """Verify row_group_size=0 raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.parquet")
            frame = ar.from_dict({"col": [1, 2, 3]})
            with pytest.raises(ValueError, match="positive integer"):
                ar.write_parquet(frame, path, row_group_size=0)

    def test_write_parquet_row_group_size_negative_raises(self):
        """Verify row_group_size must be positive."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.parquet")
            frame = ar.from_dict({"col": [1, 2, 3]})
            with pytest.raises(ValueError, match="positive integer"):
                ar.write_parquet(frame, path, row_group_size=-5)

    def test_write_parquet_row_group_size_not_integer(self):
        """Verify row_group_size must be an integer, not float."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.parquet")
            frame = ar.from_dict({"col": [1, 2, 3]})
            with pytest.raises(TypeError, match="row_group_size must be an integer"):
                ar.write_parquet(frame, path, row_group_size=10.5)


class TestWriteParquetFileExtension:
    """Test file extension validation."""

    def test_write_parquet_invalid_extension_txt(self):
        """Verify .txt extension raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.txt")
            frame = ar.from_dict({"col": [1, 2, 3]})
            with pytest.raises(ValueError, match="Unsupported file format"):
                ar.write_parquet(frame, path)

    def test_write_parquet_invalid_extension_csv(self):
        """Verify .csv extension raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.csv")
            frame = ar.from_dict({"col": [1, 2, 3]})
            with pytest.raises(ValueError, match="Unsupported file format"):
                ar.write_parquet(frame, path)

    def test_write_parquet_case_insensitive_extension(self):
        """Verify file extension check is case-insensitive."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.PARQUET")
            frame = ar.from_dict({"col": [1, 2, 3]})
            ar.write_parquet(frame, path)
            assert os.path.exists(path)


class TestWriteParquetFrameValidation:
    """Test frame parameter validation."""

    def test_write_parquet_frame_must_be_arframe(self):
        """Verify frame must be an ArFrame, not pandas DataFrame."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.parquet")
            df = pd.DataFrame({"col": [1, 2, 3]})
            with pytest.raises(TypeError, match="frame must be an ArFrame"):
                ar.write_parquet(df, path)


class TestWriteParquetPreserveAttrs:
    """Test preserve_attrs parameter."""

    def test_write_parquet_preserve_attrs_false(self):
        """Verify preserve_attrs=False works without error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.parquet")
            frame = ar.from_dict({"col": [1, 2, 3]})
            ar.write_parquet(frame, path, preserve_attrs=False)
            assert os.path.exists(path)

    def test_write_parquet_preserve_attrs_not_bool(self):
        """Verify preserve_attrs must be a boolean."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.parquet")
            frame = ar.from_dict({"col": [1, 2, 3]})
            with pytest.raises(TypeError, match="preserve_attrs must be a bool"):
                ar.write_parquet(frame, path, preserve_attrs="yes")


class TestWriteParquetEdgeCases:
    """Test edge cases and special scenarios."""

    def test_write_parquet_empty_frame(self):
        """Verify writing empty frame (0 rows, 2+ columns) works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.parquet")
            frame = ar.from_dict({"col1": [], "col2": []})
            ar.write_parquet(frame, path)
            assert os.path.exists(path)

    def test_write_parquet_single_row(self):
        """Verify writing single-row frame works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.parquet")
            frame = ar.from_dict({"col": [42]})
            ar.write_parquet(frame, path)
            assert os.path.exists(path)

    def test_write_parquet_multiple_dtypes(self):
        """Verify writing frame with multiple data types works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.parquet")
            frame = ar.from_dict({
                "ints": [1, 2, 3],
                "floats": [1.5, 2.5, 3.5],
                "strings": ["a", "b", "c"]
            })
            ar.write_parquet(frame, path)
            assert os.path.exists(path)

    def test_write_parquet_with_nulls(self):
        """Verify writing frame with null values works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.parquet")
            frame = ar.from_dict({"col1": [1, None, 3], "col2": ["a", "b", None]})
            ar.write_parquet(frame, path)
            assert os.path.exists(path)
