import pytest
import pandas as pd
import os
import tempfile
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
        """Create a simple ArFrame for testing."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "value": [10.5, 20.3, 15.7]
        })
        return ar.from_pandas(df)

    # ===== Core functionality =====

    def test_write_parquet_basic(self, temp_dir, sample_frame):
        """Test basic write_parquet with default parameters."""
        output_path = os.path.join(temp_dir, "output.parquet")
        ar.write_parquet(sample_frame, output_path)
        assert os.path.exists(output_path)
        result = ar.read_parquet(output_path)
        assert result.shape == sample_frame.shape

    def test_write_parquet_pq_extension(self, temp_dir, sample_frame):
        """Test write_parquet with .pq file extension."""
        output_path = os.path.join(temp_dir, "output.pq")
        ar.write_parquet(sample_frame, output_path)
        assert os.path.exists(output_path)

    def test_write_parquet_pathlib(self, temp_dir, sample_frame):
        """Test write_parquet with pathlib.Path object."""
        output_path = Path(temp_dir) / "output.parquet"
        ar.write_parquet(sample_frame, output_path)
        assert output_path.exists()

    # ===== File extension validation =====

    def test_write_parquet_invalid_file_extension(self, temp_dir, sample_frame):
        """Test that invalid file extension raises ValueError."""
        output_path = os.path.join(temp_dir, "output.csv")
        with pytest.raises(ValueError, match="Unsupported file format"):
            ar.write_parquet(sample_frame, output_path)

    def test_write_parquet_case_insensitive_extension(self, temp_dir, sample_frame):
        """Test that file extension check is case-insensitive."""
        output_path = os.path.join(temp_dir, "output.PARQUET")
        ar.write_parquet(sample_frame, output_path)
        assert os.path.exists(output_path)

    # ===== Compression parameter =====

    def test_write_parquet_compression_zstd(self, temp_dir, sample_frame):
        """Test write_parquet with zstd compression codec."""
        output_path = os.path.join(temp_dir, "output.parquet")
        ar.write_parquet(sample_frame, output_path, compression="zstd")
        assert os.path.exists(output_path)

    def test_write_parquet_compression_none(self, temp_dir, sample_frame):
        """Test write_parquet with compression='none'."""
        output_path = os.path.join(temp_dir, "output.parquet")
        ar.write_parquet(sample_frame, output_path, compression="none")
        assert os.path.exists(output_path)

    def test_write_parquet_invalid_compression(self, temp_dir, sample_frame):
        """Test that invalid compression codec raises ValueError."""
        output_path = os.path.join(temp_dir, "output.parquet")
        with pytest.raises(ValueError, match="Unknown compression codec"):
            ar.write_parquet(sample_frame, output_path, compression="invalid_codec")

    def test_write_parquet_non_string_compression(self, temp_dir, sample_frame):
        """Test that non-string compression parameter raises TypeError."""
        output_path = os.path.join(temp_dir, "output.parquet")
        with pytest.raises(TypeError, match="compression must be a string"):
            ar.write_parquet(sample_frame, output_path, compression=123)

    # ===== Row group size parameter =====

    def test_write_parquet_row_group_size_valid(self, temp_dir, sample_frame):
        """Test write_parquet with valid row_group_size parameter."""
        output_path = os.path.join(temp_dir, "output.parquet")
        ar.write_parquet(sample_frame, output_path, row_group_size=1000)
        assert os.path.exists(output_path)

    def test_write_parquet_row_group_size_invalid(self, temp_dir, sample_frame):
        """Test that non-positive row_group_size raises ValueError."""
        output_path = os.path.join(temp_dir, "output.parquet")
        with pytest.raises(ValueError, match="row_group_size must be a positive integer"):
            ar.write_parquet(sample_frame, output_path, row_group_size=0)
        with pytest.raises(ValueError, match="row_group_size must be a positive integer"):
            ar.write_parquet(sample_frame, output_path, row_group_size=-1)

    def test_write_parquet_row_group_size_non_integer(self, temp_dir, sample_frame):
        """Test that non-integer row_group_size raises TypeError."""
        output_path = os.path.join(temp_dir, "output.parquet")
        with pytest.raises(TypeError, match="row_group_size must be an integer"):
            ar.write_parquet(sample_frame, output_path, row_group_size=100.5)

    # ===== Input validation =====

    def test_write_parquet_non_arframe(self, temp_dir):
        """Test that non-ArFrame input raises TypeError."""
        output_path = os.path.join(temp_dir, "output.parquet")
        df = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(TypeError, match="frame must be an ArFrame"):
            ar.write_parquet(df, output_path)

    def test_write_parquet_invalid_path_type(self, sample_frame):
        """Test that invalid path type raises TypeError."""
        with pytest.raises(TypeError, match="path must be a string"):
            ar.write_parquet(sample_frame, 12345)

    def test_write_parquet_preserve_attrs_non_bool(self, temp_dir, sample_frame):
        """Test that non-boolean preserve_attrs parameter raises TypeError."""
        output_path = os.path.join(temp_dir, "output.parquet")
        with pytest.raises(TypeError, match="preserve_attrs must be a bool"):
            ar.write_parquet(sample_frame, output_path, preserve_attrs="yes")
