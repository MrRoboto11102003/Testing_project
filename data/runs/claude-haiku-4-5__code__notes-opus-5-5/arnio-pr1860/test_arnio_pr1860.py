import json
import re
from pathlib import Path

import pandas as pd
import pytest

import arnio as ar


@pytest.fixture
def sample_frame():
    """Create a simple ArFrame for testing."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    return ar.from_pandas(df)


@pytest.fixture
def empty_frame():
    """Create an empty ArFrame."""
    df = pd.DataFrame()
    return ar.from_pandas(df)


class TestWriteParquetBasic:
    """Basic functionality and successful writes."""

    def test_basic_write_returns_none(self, sample_frame, tmp_path):
        """write_parquet returns None on success."""
        path = tmp_path / "out.parquet"
        result = ar.write_parquet(sample_frame, str(path))
        assert result is None
        assert path.exists()

    def test_write_pq_extension(self, sample_frame, tmp_path):
        """Both .parquet and .pq extensions are valid."""
        path1 = tmp_path / "out.parquet"
        ar.write_parquet(sample_frame, str(path1))
        
        path2 = tmp_path / "out.pq"
        ar.write_parquet(sample_frame, str(path2))
        
        assert path1.exists() and path2.exists()

    def test_write_case_insensitive_extension(self, sample_frame, tmp_path):
        """Extension check is case-insensitive."""
        path = tmp_path / "out.PARQUET"
        ar.write_parquet(sample_frame, str(path))
        assert path.exists()

    def test_write_pathlib_and_bytes(self, sample_frame, tmp_path):
        """Path can be pathlib.Path or bytes."""
        path1 = tmp_path / "out1.parquet"
        ar.write_parquet(sample_frame, path1)
        
        path2 = str(tmp_path / "out2.parquet").encode("utf-8")
        ar.write_parquet(sample_frame, path2)
        
        assert (tmp_path / "out1.parquet").exists()
        assert (tmp_path / "out2.parquet").exists()

    def test_write_all_valid_compressions(self, sample_frame, tmp_path):
        """All valid compression codecs work."""
        for codec in ["snappy", "gzip", "brotli", "zstd", "none"]:
            path = tmp_path / f"out_{codec}.parquet"
            ar.write_parquet(sample_frame, str(path), compression=codec)
            assert path.exists()


class TestWriteParquetValidationOrder:
    """Validation happens in documented order: frame, path, extension, compression, etc."""

    def test_frame_not_arframe_raises_typeerror(self, tmp_path):
        """Frame must be an ArFrame; checked first."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        path = tmp_path / "out.parquet"
        with pytest.raises(TypeError, match=re.escape("frame must be an ArFrame")):
            ar.write_parquet(df, str(path))

    def test_path_invalid_type_raises_typeerror(self, sample_frame):
        """Path type is validated before extension."""
        with pytest.raises(
            TypeError,
            match=re.escape(
                "path must be a string, bytes, or os.PathLike object, got 'int'"
            ),
        ):
            ar.write_parquet(sample_frame, 123)

    def test_path_invalid_extension_raises_valueerror(self, sample_frame, tmp_path):
        """Extension must be .parquet or .pq."""
        path = tmp_path / "out.csv"
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Unsupported file format: {str(path)}. "
                "write_parquet only supports .parquet and .pq files."
            ),
        ):
            ar.write_parquet(sample_frame, str(path))

    def test_compression_not_string_raises_typeerror(self, sample_frame, tmp_path):
        """Compression must be a string."""
        path = tmp_path / "out.parquet"
        with pytest.raises(TypeError, match=re.escape("compression must be a string")):
            ar.write_parquet(sample_frame, str(path), compression=None)

    def test_compression_invalid_codec_raises_valueerror(self, sample_frame, tmp_path):
        """Compression codec must be valid; check is case-sensitive."""
        path = tmp_path / "out.parquet"
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Unknown compression codec: 'lz4'. "
                "Valid options are: ['brotli', 'gzip', 'none', 'snappy', 'zstd']"
            ),
        ):
            ar.write_parquet(sample_frame, str(path), compression="lz4")

    def test_compression_case_sensitive(self, sample_frame, tmp_path):
        """Compression codec check is case-sensitive."""
        path = tmp_path / "out.parquet"
        with pytest.raises(ValueError, match="Unknown compression codec: 'SNAPPY'"):
            ar.write_parquet(sample_frame, str(path), compression="SNAPPY")


class TestWriteParquetRowGroupSize:
    """row_group_size validation and behavior."""

    def test_row_group_size_bool_rejected(self, sample_frame, tmp_path):
        """row_group_size as bool is rejected (not an int)."""
        path = tmp_path / "out.parquet"
        with pytest.raises(TypeError, match=re.escape("row_group_size must be an integer")):
            ar.write_parquet(sample_frame, str(path), row_group_size=True)

    def test_row_group_size_float_rejected(self, sample_frame, tmp_path):
        """row_group_size as float is rejected."""
        path = tmp_path / "out.parquet"
        with pytest.raises(TypeError, match=re.escape("row_group_size must be an integer")):
            ar.write_parquet(sample_frame, str(path), row_group_size=2.5)

    def test_row_group_size_nonpositive_rejected(self, sample_frame, tmp_path):
        """row_group_size must be positive."""
        path = tmp_path / "out.parquet"
        with pytest.raises(
            ValueError, match=re.escape("row_group_size must be a positive integer")
        ):
            ar.write_parquet(sample_frame, str(path), row_group_size=0)
        
        with pytest.raises(
            ValueError, match=re.escape("row_group_size must be a positive integer")
        ):
            ar.write_parquet(sample_frame, str(path), row_group_size=-5)

    def test_row_group_size_valid(self, sample_frame, tmp_path):
        """Positive row_group_size works."""
        path = tmp_path / "out.parquet"
        ar.write_parquet(sample_frame, str(path), row_group_size=1)
        assert path.exists()


class TestWriteParquetPreserveAttrs:
    """preserve_attrs parameter validation and behavior."""

    def test_preserve_attrs_not_bool_raises_typeerror(self, sample_frame, tmp_path):
        """preserve_attrs must be a bool (not int or None)."""
        path = tmp_path / "out.parquet"
        with pytest.raises(TypeError, match=re.escape("preserve_attrs must be a bool")):
            ar.write_parquet(sample_frame, str(path), preserve_attrs=1)
        
        with pytest.raises(TypeError, match=re.escape("preserve_attrs must be a bool")):
            ar.write_parquet(sample_frame, str(path), preserve_attrs=None)

    def test_preserve_attrs_false_works(self, sample_frame, tmp_path):
        """preserve_attrs=False succeeds."""
        path = tmp_path / "out.parquet"
        ar.write_parquet(sample_frame, str(path), preserve_attrs=False)
        assert path.exists()


class TestWriteParquetEdgeCases:
    """Edge cases: empty frames, file behavior, data preservation."""

    def test_empty_frame_zero_by_zero(self, empty_frame, tmp_path):
        """Empty frame (0 rows, 0 columns) can be written."""
        path = tmp_path / "out.parquet"
        ar.write_parquet(empty_frame, str(path))
        assert path.exists()

    def test_zero_rows_with_columns(self, tmp_path):
        """Frame with columns but zero rows can be written."""
        df = pd.DataFrame({"a": [], "b": []})
        frame = ar.from_pandas(df)
        path = tmp_path / "out.parquet"
        ar.write_parquet(frame, str(path))
        assert path.exists()

    def test_no_file_created_on_validation_error(self, sample_frame, tmp_path):
        """File is not created when validation fails."""
        path = tmp_path / "out.csv"
        try:
            ar.write_parquet(sample_frame, str(path))
        except ValueError:
            pass
        assert not path.exists()

    def test_index_not_written(self, tmp_path):
        """DataFrame index is not included in output file."""
        df = pd.DataFrame({"a": [1, 2, 3]}, index=[10, 20, 30])
        frame = ar.from_pandas(df)
        path = tmp_path / "out.parquet"
        ar.write_parquet(frame, str(path))
        
        result = pd.read_parquet(str(path))
        assert list(result.columns) == ["a"]
        assert list(result.index) == [0, 1, 2]

    def test_column_order_preserved(self, tmp_path):
        """Column order from ArFrame is preserved in output."""
        df = pd.DataFrame({"c": [3], "a": [1], "b": [2]})
        frame = ar.from_pandas(df)
        path = tmp_path / "out.parquet"
        ar.write_parquet(frame, str(path))
        
        result = pd.read_parquet(str(path))
        assert list(result.columns) == ["c", "a", "b"]

    def test_file_overwrite(self, tmp_path):
        """Writing to existing file overwrites it."""
        path = tmp_path / "out.parquet"
        
        df1 = pd.DataFrame({"x": [1, 2, 3]})
        frame1 = ar.from_pandas(df1)
        ar.write_parquet(frame1, str(path))
        
        df2 = pd.DataFrame({"y": [4, 5, 6], "z": [7, 8, 9]})
        frame2 = ar.from_pandas(df2)
        ar.write_parquet(frame2, str(path))
        
        result = pd.read_parquet(str(path))
        assert set(result.columns) == {"y", "z"}
        assert len(result) == 3
