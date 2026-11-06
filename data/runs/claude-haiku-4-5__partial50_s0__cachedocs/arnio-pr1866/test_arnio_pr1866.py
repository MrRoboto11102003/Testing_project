import pytest
import tempfile
import os
from pathlib import Path
import pandas as pd
import numpy as np
import arnio as ar


@pytest.fixture
def sample_frame():
    """Create a basic test frame with various data types"""
    data = {
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        "score": [95.5, 87.3, 92.1, 88.9, 91.2],
        "active": [True, False, True, True, False],
    }
    df = pd.DataFrame(data)
    return ar.from_pandas(df)


@pytest.fixture
def frame_with_nulls():
    """Create a frame with null values"""
    data = {
        "id": [1, 2, None, 4, 5],
        "name": ["Alice", None, "Charlie", "Diana", "Eve"],
        "score": [95.5, 87.3, 92.1, None, 91.2],
    }
    df = pd.DataFrame(data)
    return ar.from_pandas(df)


@pytest.fixture
def empty_frame():
    """Create an empty frame"""
    df = pd.DataFrame({"col1": [], "col2": [], "col3": []})
    return ar.from_pandas(df)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestWriteParquetBasic:
    """Test basic write_parquet functionality"""

    def test_write_parquet_creates_file(self, sample_frame, temp_dir):
        """Test that write_parquet creates a .parquet file"""
        output_path = os.path.join(temp_dir, "output.parquet")
        ar.write_parquet(sample_frame, output_path)
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0

    def test_write_parquet_pq_extension(self, sample_frame, temp_dir):
        """Test write with .pq extension"""
        output_path = os.path.join(temp_dir, "output.pq")
        ar.write_parquet(sample_frame, output_path)
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0

    def test_write_parquet_pathlib_path(self, sample_frame, temp_dir):
        """Test write with pathlib.Path object"""
        output_path = Path(temp_dir) / "output.parquet"
        ar.write_parquet(sample_frame, output_path)
        assert output_path.exists()

    def test_write_read_roundtrip(self, sample_frame, temp_dir):
        """Test data integrity through write/read cycle"""
        output_path = os.path.join(temp_dir, "roundtrip.parquet")
        ar.write_parquet(sample_frame, output_path)

        read_frame = ar.read_parquet(output_path)
        result_df = ar.to_pandas(read_frame)
        expected_df = ar.to_pandas(sample_frame)

        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_overwrite_existing_file(self, sample_frame, temp_dir):
        """Test that write_parquet can overwrite existing files"""
        output_path = os.path.join(temp_dir, "overwrite.parquet")

        ar.write_parquet(sample_frame, output_path)
        first_size = os.path.getsize(output_path)

        ar.write_parquet(sample_frame, output_path)
        second_size = os.path.getsize(output_path)

        assert first_size == second_size


class TestWriteParquetCompression:
    """Test compression codec parameter"""

    @pytest.mark.parametrize("compression", ["snappy", "gzip", "zstd", "brotli", "none"])
    def test_compression_codecs(self, sample_frame, temp_dir, compression):
        """Test all supported compression codecs"""
        output_path = os.path.join(temp_dir, f"output_{compression}.parquet")
        ar.write_parquet(sample_frame, output_path, compression=compression)
        assert os.path.exists(output_path)

        # Verify data can be read back
        read_frame = ar.read_parquet(output_path)
        result_df = ar.to_pandas(read_frame)
        expected_df = ar.to_pandas(sample_frame)
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_default_compression_is_snappy(self, sample_frame, temp_dir):
        """Test that default compression is snappy"""
        path_default = os.path.join(temp_dir, "default.parquet")
        path_explicit = os.path.join(temp_dir, "explicit_snappy.parquet")

        ar.write_parquet(sample_frame, path_default)
        ar.write_parquet(sample_frame, path_explicit, compression="snappy")

        # Both should produce similar file sizes
        size_default = os.path.getsize(path_default)
        size_explicit = os.path.getsize(path_explicit)
        assert abs(size_default - size_explicit) < 100


class TestWriteParquetRowGroupSize:
    """Test row_group_size parameter"""

    def test_row_group_size_parameter(self, sample_frame, temp_dir):
        """Test custom row_group_size parameter"""
        output_path = os.path.join(temp_dir, "custom_rg.parquet")
        ar.write_parquet(sample_frame, output_path, row_group_size=2)
        assert os.path.exists(output_path)

        # Verify data integrity
        read_frame = ar.read_parquet(output_path)
        result_df = ar.to_pandas(read_frame)
        expected_df = ar.to_pandas(sample_frame)
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_row_group_size_none_default(self, sample_frame, temp_dir):
        """Test with row_group_size=None uses default"""
        output_path = os.path.join(temp_dir, "rg_none.parquet")
        ar.write_parquet(sample_frame, output_path, row_group_size=None)
        assert os.path.exists(output_path)


class TestWriteParquetDataTypes:
    """Test handling of various data types and null values"""

    def test_write_with_null_values(self, frame_with_nulls, temp_dir):
        """Test writing frame with null/missing values"""
        output_path = os.path.join(temp_dir, "with_nulls.parquet")
        ar.write_parquet(frame_with_nulls, output_path)

        read_frame = ar.read_parquet(output_path)
        result_df = ar.to_pandas(read_frame)
        expected_df = ar.to_pandas(frame_with_nulls)

        assert result_df.shape == expected_df.shape

    def test_write_empty_frame(self, empty_frame, temp_dir):
        """Test writing an empty frame with schema but no rows"""
        output_path = os.path.join(temp_dir, "empty.parquet")
        ar.write_parquet(empty_frame, output_path)
        assert os.path.exists(output_path)

        # Verify columns are preserved
        read_frame = ar.read_parquet(output_path)
        result_df = ar.to_pandas(read_frame)
        expected_df = ar.to_pandas(empty_frame)
        assert result_df.columns.tolist() == expected_df.columns.tolist()


class TestWriteParquetPreserveAttrs:
    """Test preserve_attrs parameter"""

    def test_preserve_attrs_true(self, sample_frame, temp_dir):
        """Test write with preserve_attrs=True"""
        output_path = os.path.join(temp_dir, "preserve_true.parquet")
        ar.write_parquet(sample_frame, output_path, preserve_attrs=True)
        assert os.path.exists(output_path)

    def test_preserve_attrs_false(self, sample_frame, temp_dir):
        """Test write with preserve_attrs=False"""
        output_path = os.path.join(temp_dir, "preserve_false.parquet")
        ar.write_parquet(sample_frame, output_path, preserve_attrs=False)
        assert os.path.exists(output_path)


class TestWriteParquetErrorHandling:
    """Test error handling and edge cases"""

    def test_invalid_compression_raises_error(self, sample_frame, temp_dir):
        """Test that invalid compression codec raises an error"""
        output_path = os.path.join(temp_dir, "invalid.parquet")
        with pytest.raises((ValueError, KeyError, Exception)):
            ar.write_parquet(
                sample_frame, output_path, compression="invalid_codec"
            )
