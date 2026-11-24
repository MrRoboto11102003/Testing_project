import pytest
import pandas as pd
from pathlib import Path
import re

try:
    import pyarrow  # noqa: F401
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

import arnio as ar


@pytest.mark.skipif(not HAS_PYARROW, reason="pyarrow not installed")
class TestWriteParquetValidation:
    """Tests for argument validation."""

    def test_frame_must_be_arframe(self, tmp_path):
        """TypeError if frame is not an ArFrame."""
        p = tmp_path / "out.parquet"
        with pytest.raises(TypeError, match=re.escape("frame must be an ArFrame")):
            ar.write_parquet(pd.DataFrame({"a": [1, 2]}), p)
        assert not p.exists()

    def test_path_type_validation(self, tmp_path):
        """TypeError if path is not str, bytes, or os.PathLike."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))
        with pytest.raises(TypeError, match="path must be a string, bytes, or os.PathLike object"):
            ar.write_parquet(frame, 123)

    def test_path_extension_and_case_insensitive(self, tmp_path):
        """ValueError if path extension is not .parquet/.pq; check is case-insensitive."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))

        # Invalid extension
        p_invalid = tmp_path / "out.csv"
        msg = f"Unsupported file format: {p_invalid}. write_parquet only supports .parquet and .pq files."
        with pytest.raises(ValueError, match=re.escape(msg)):
            ar.write_parquet(frame, p_invalid)
        assert not p_invalid.exists()

        # Valid extensions (case-insensitive)
        for ext in ["PARQUET", "Pq", "pq"]:
            p = tmp_path / f"out.{ext}"
            ar.write_parquet(frame, p)
            assert p.exists()

    def test_compression_type_and_value_validation(self, tmp_path):
        """TypeError if compression not string; ValueError if not recognized codec."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))
        p = tmp_path / "out.parquet"

        # Type error
        with pytest.raises(TypeError, match=re.escape("compression must be a string")):
            ar.write_parquet(frame, p, compression=None)
        assert not p.exists()

        # Value errors (case-sensitive, unknown codec)
        with pytest.raises(ValueError, match="Unknown compression codec"):
            ar.write_parquet(frame, p, compression="lz4")
        with pytest.raises(ValueError, match="Unknown compression codec"):
            ar.write_parquet(frame, p, compression="SNAPPY")

    def test_row_group_size_type_validation(self, tmp_path):
        """TypeError if row_group_size is not an int (including bool)."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))
        p = tmp_path / "out.parquet"

        with pytest.raises(TypeError, match=re.escape("row_group_size must be an integer")):
            ar.write_parquet(frame, p, row_group_size=True)
        with pytest.raises(TypeError, match=re.escape("row_group_size must be an integer")):
            ar.write_parquet(frame, p, row_group_size=2.5)

    def test_row_group_size_positive_validation(self, tmp_path):
        """ValueError if row_group_size is not positive."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))
        p = tmp_path / "out.parquet"

        with pytest.raises(ValueError, match=re.escape("row_group_size must be a positive integer")):
            ar.write_parquet(frame, p, row_group_size=0)
        with pytest.raises(ValueError, match=re.escape("row_group_size must be a positive integer")):
            ar.write_parquet(frame, p, row_group_size=-5)

    def test_preserve_attrs_type_validation(self, tmp_path):
        """TypeError if preserve_attrs is not a bool."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))
        p = tmp_path / "out.parquet"

        with pytest.raises(TypeError, match=re.escape("preserve_attrs must be a bool")):
            ar.write_parquet(frame, p, preserve_attrs=1)
        with pytest.raises(TypeError, match=re.escape("preserve_attrs must be a bool")):
            ar.write_parquet(frame, p, preserve_attrs=None)


@pytest.mark.skipif(not HAS_PYARROW, reason="pyarrow not installed")
class TestWriteParquetFunctionality:
    """Tests for successful writes and file properties."""

    def test_basic_write(self, tmp_path):
        """Basic write with default parameters succeeds."""
        p = tmp_path / "out.parquet"
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}))
        result = ar.write_parquet(frame, p)
        assert result is None
        assert p.exists()
        df = pd.read_parquet(p)
        assert df.shape == (3, 2)
        assert list(df.columns) == ["a", "b"]

    def test_path_formats(self, tmp_path):
        """Path works as str, bytes, pathlib.Path; .parquet and .pq extensions."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))

        p_pq = tmp_path / "out.pq"
        ar.write_parquet(frame, p_pq)
        assert p_pq.exists()

        p_pathlib = tmp_path / "out_pathlib.parquet"
        ar.write_parquet(frame, Path(p_pathlib))
        assert p_pathlib.exists()

        p_bytes = tmp_path / "out_bytes.parquet"
        ar.write_parquet(frame, str(p_bytes).encode("utf-8"))
        assert p_bytes.exists()

    def test_compression_codecs(self, tmp_path):
        """All valid compression codecs work."""
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))
        for codec in ["snappy", "gzip", "brotli", "zstd", "none"]:
            p = tmp_path / f"out_{codec}.parquet"
            ar.write_parquet(frame, p, compression=codec)
            assert p.exists()

    def test_row_group_size_parameter(self, tmp_path):
        """row_group_size parameter is accepted and used."""
        p = tmp_path / "out.parquet"
        frame = ar.from_pandas(pd.DataFrame({"a": range(100)}))
        ar.write_parquet(frame, p, row_group_size=25)
        assert p.exists()
        df = pd.read_parquet(p)
        assert len(df) == 100

    def test_empty_frame(self, tmp_path):
        """Frame with 0 rows can be written."""
        p = tmp_path / "out.parquet"
        frame = ar.from_pandas(pd.DataFrame({"a": [], "b": []}))
        ar.write_parquet(frame, p)
        assert p.exists()
        df = pd.read_parquet(p)
        assert len(df) == 0
        assert list(df.columns) == ["a", "b"]

    def test_preserve_attrs_false(self, tmp_path):
        """preserve_attrs=False allows write to succeed."""
        p = tmp_path / "out.parquet"
        frame = ar.from_pandas(pd.DataFrame({"a": [1, 2]}))
        ar.write_parquet(frame, p, preserve_attrs=False)
        assert p.exists()

    def test_zero_column_frame_with_rows(self, tmp_path):
        """Zero-column frame with rows raises ValueError."""
        p = tmp_path / "out.parquet"
        df = pd.DataFrame(index=range(3))
        frame = ar.from_pandas(df)
        if frame.shape == (3, 0):
            with pytest.raises(ValueError, match="Cannot write a zero-column ArFrame"):
                ar.write_parquet(frame, p)
            assert not p.exists()
        else:
            pytest.skip("Cannot create zero-column frame")
