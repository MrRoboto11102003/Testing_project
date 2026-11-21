import os
import re
from pathlib import Path

import pandas as pd
import pytest

import arnio as ar


class TestWriteParquet:
    """Tests for arnio.io.write_parquet."""

    def test_write_parquet_basic(self, tmp_path):
        """Test successful write with default parameters."""
        output = tmp_path / "output.parquet"
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        frame = ar.from_pandas(df)

        result = ar.write_parquet(frame, str(output))

        assert result is None
        assert output.exists()
        read_df = pd.read_parquet(output)
        assert read_df.shape == (3, 2)
        assert list(read_df.columns) == ["a", "b"]

    def test_write_parquet_pathlib_path(self, tmp_path):
        """Test that pathlib.Path works as path argument."""
        output = tmp_path / "output.parquet"
        frame = ar.from_pandas(pd.DataFrame({"x": [1, 2]}))

        ar.write_parquet(frame, output)

        assert output.exists()

    def test_write_parquet_pq_extension(self, tmp_path):
        """Test write with .pq extension."""
        output = tmp_path / "output.pq"
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))

        ar.write_parquet(frame, str(output))

        assert output.exists()

    def test_write_parquet_compression_gzip(self, tmp_path):
        """Test write with gzip compression."""
        import pyarrow.parquet as pq

        output = tmp_path / "output.parquet"
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))

        ar.write_parquet(frame, str(output), compression="gzip")

        pf = pq.ParquetFile(str(output))
        assert pf.metadata.row_group(0).column(0).compression == "GZIP"

    def test_write_parquet_compression_none(self, tmp_path):
        """Test write with compression='none' (uncompressed)."""
        import pyarrow.parquet as pq

        output = tmp_path / "output.parquet"
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))

        ar.write_parquet(frame, str(output), compression="none")

        pf = pq.ParquetFile(str(output))
        assert pf.metadata.row_group(0).column(0).compression == "UNCOMPRESSED"

    def test_write_parquet_row_group_size(self, tmp_path):
        """Test write with custom row_group_size."""
        import pyarrow.parquet as pq

        output = tmp_path / "output.parquet"
        frame = ar.from_pandas(pd.DataFrame({"a": range(10)}))

        ar.write_parquet(frame, str(output), row_group_size=3)

        pf = pq.ParquetFile(str(output))
        # 10 rows with row_group_size=3 gives 4 row groups
        assert pf.metadata.num_row_groups == 4

    def test_write_parquet_invalid_frame_type(self, tmp_path):
        """Test that pandas DataFrame raises TypeError."""
        output = tmp_path / "output.parquet"
        df = pd.DataFrame({"a": [1, 2, 3]})

        with pytest.raises(TypeError, match="frame must be an ArFrame"):
            ar.write_parquet(df, str(output))

    def test_write_parquet_invalid_path_type(self, tmp_path):
        """Test that invalid path type raises TypeError."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))

        with pytest.raises(TypeError, match=r"path must be a string, bytes, or os\.PathLike object, got 'int'"):
            ar.write_parquet(frame, 123)

    def test_write_parquet_invalid_extension(self, tmp_path):
        """Test that unsupported file extension raises ValueError."""
        output = tmp_path / "output.csv"
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))

        with pytest.raises(ValueError, match="Unsupported file format"):
            ar.write_parquet(frame, str(output))

        # File should not be created on error
        assert not output.exists()

    def test_write_parquet_invalid_compression_type(self, tmp_path):
        """Test that non-string compression raises TypeError."""
        output = tmp_path / "output.parquet"
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))

        with pytest.raises(TypeError, match="compression must be a string"):
            ar.write_parquet(frame, str(output), compression=None)

    def test_write_parquet_invalid_compression_value(self, tmp_path):
        """Test that unknown compression codec raises ValueError."""
        output = tmp_path / "output.parquet"
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))

        with pytest.raises(ValueError, match=r"Unknown compression codec: 'lz4'"):
            ar.write_parquet(frame, str(output), compression="lz4")

    def test_write_parquet_compression_case_sensitive(self, tmp_path):
        """Test that compression codec check is case-sensitive."""
        output = tmp_path / "output.parquet"
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))

        with pytest.raises(ValueError, match="Unknown compression codec"):
            ar.write_parquet(frame, str(output), compression="SNAPPY")

    def test_write_parquet_row_group_size_bool_rejected(self, tmp_path):
        """Test that bool is rejected for row_group_size."""
        output = tmp_path / "output.parquet"
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))

        with pytest.raises(TypeError, match="row_group_size must be an integer"):
            ar.write_parquet(frame, str(output), row_group_size=True)

    def test_write_parquet_row_group_size_not_positive(self, tmp_path):
        """Test that non-positive row_group_size raises ValueError."""
        output = tmp_path / "output.parquet"
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))

        with pytest.raises(ValueError, match="row_group_size must be a positive integer"):
            ar.write_parquet(frame, str(output), row_group_size=0)

    def test_write_parquet_preserve_attrs_invalid_type(self, tmp_path):
        """Test that non-bool preserve_attrs raises TypeError."""
        output = tmp_path / "output.parquet"
        frame = ar.from_pandas(pd.DataFrame({"a": [1]}))

        with pytest.raises(TypeError, match="preserve_attrs must be a bool"):
            ar.write_parquet(frame, str(output), preserve_attrs=1)

    def test_write_parquet_empty_frame_valid(self, tmp_path):
        """Test write of empty frame (0 rows, 1+ columns) succeeds."""
        output = tmp_path / "output.parquet"
        frame = ar.from_pandas(pd.DataFrame({"a": [], "b": []}))

        ar.write_parquet(frame, str(output))

        assert output.exists()
        read_df = pd.read_parquet(output)
        assert read_df.shape == (0, 2)
