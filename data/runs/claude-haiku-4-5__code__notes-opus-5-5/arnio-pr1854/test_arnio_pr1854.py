import pytest
import io
import gzip
from pathlib import Path
import arnio as ar


class TestReadCsvChunked:
    """Tests for arnio.io.read_csv_chunked"""

    def test_basic_chunked_iteration(self, tmp_path):
        """Test basic CSV reading with chunking"""
        p = tmp_path / "data.csv"
        p.write_text("a,b\n1,x\n2,y\n3,z\n")
        chunks = list(ar.read_csv_chunked(p, chunksize=2))
        assert len(chunks) == 2
        assert chunks[0].shape == (2, 2)
        assert chunks[1].shape == (1, 2)

    def test_generator_validates_on_consumption(self, tmp_path):
        """Test that validation errors occur on first next(), not on function call"""
        p = tmp_path / "data.csv"
        p.write_text("a,b\n1,x\n2,y\n")
        gen = ar.read_csv_chunked(p, chunksize=0)
        with pytest.raises(ValueError, match="chunksize must be a positive integer"):
            next(gen)

    def test_chunksize_bool_rejected(self, tmp_path):
        """Test that boolean chunksize raises TypeError"""
        p = tmp_path / "data.csv"
        p.write_text("a,b\n1,x\n")
        with pytest.raises(TypeError, match="chunksize must be an integer"):
            next(ar.read_csv_chunked(p, chunksize=True))

    def test_tsv_auto_delimiter_detection(self, tmp_path):
        """Test automatic tab delimiter detection for .tsv extension"""
        p = tmp_path / "data.tsv"
        p.write_text("a\tb\n1\tx\n2\ty\n")
        chunks = list(ar.read_csv_chunked(p, chunksize=10))
        assert chunks[0].shape == (2, 2)

    def test_explicit_delimiter_overrides_extension(self, tmp_path):
        """Test explicit delimiter takes precedence over .tsv extension"""
        p = tmp_path / "data.tsv"
        p.write_text("a,b\n1,x\n2,y\n")
        chunks = list(ar.read_csv_chunked(p, delimiter=",", chunksize=10))
        assert chunks[0].shape == (2, 2)

    def test_unsupported_file_extension_raises(self, tmp_path):
        """Test that unsupported file extension raises ValueError"""
        p = tmp_path / "data.dat"
        p.write_text("a,b\n1,x\n")
        with pytest.raises(ValueError, match="Unsupported file format"):
            next(ar.read_csv_chunked(p))

    def test_skip_rows_skips_data_rows(self, tmp_path):
        """Test skip_rows parameter skips data rows after header"""
        p = tmp_path / "data.csv"
        p.write_text("a,b\n1,x\n2,y\n3,z\n")
        chunks = list(ar.read_csv_chunked(p, chunksize=10, skip_rows=1))
        assert chunks[0].shape[0] == 2

    def test_skiprows_alias_conflict(self, tmp_path):
        """Test that conflicting skiprows and skip_rows raise ValueError"""
        p = tmp_path / "data.csv"
        p.write_text("a,b\n1,x\n")
        with pytest.raises(ValueError, match="Conflicting values"):
            next(ar.read_csv_chunked(p, skip_rows=1, skiprows=2))

    def test_skiprows_and_skip_rows_agreement(self, tmp_path):
        """Test identical skiprows and skip_rows values work together"""
        p = tmp_path / "data.csv"
        p.write_text("a,b\n1,x\n2,y\n3,z\n")
        chunks = list(ar.read_csv_chunked(p, chunksize=10, skip_rows=1, skiprows=1))
        assert chunks[0].shape[0] == 2

    def test_nrows_limits_total_rows(self, tmp_path):
        """Test nrows limits total data rows across all chunks"""
        p = tmp_path / "data.csv"
        p.write_text("a,b\n1,x\n2,y\n3,z\n4,w\n5,v\n")
        chunks = list(ar.read_csv_chunked(p, chunksize=2, nrows=3))
        total_rows = sum(c.shape[0] for c in chunks)
        assert total_rows == 3

    def test_pathlib_path_input(self, tmp_path):
        """Test that pathlib.Path input is accepted"""
        p = tmp_path / "data.csv"
        p.write_text("a,b\n1,x\n2,y\n")
        chunks = list(ar.read_csv_chunked(Path(p), chunksize=10))
        assert chunks[0].shape == (2, 2)

    def test_stringio_file_like_input(self, tmp_path):
        """Test that StringIO file-like input works"""
        data = "a,b\n1,x\n2,y\n"
        chunks = list(ar.read_csv_chunked(io.StringIO(data), chunksize=10))
        assert chunks[0].shape == (2, 2)

    def test_usecols_filters_columns(self, tmp_path):
        """Test usecols parameter selects specific columns"""
        p = tmp_path / "data.csv"
        p.write_text("a,b,c\n1,x,10\n2,y,20\n")
        chunks = list(ar.read_csv_chunked(p, chunksize=10, usecols=["a", "b"]))
        assert chunks[0].shape == (2, 2)

    def test_decimal_separator_validation(self, tmp_path):
        """Test that decimal_separator must be non-alphanumeric"""
        p = tmp_path / "data.csv"
        p.write_text("a,b\n1,x\n")
        with pytest.raises(ValueError, match="decimal_separator must be a single non-alphanumeric character"):
            next(ar.read_csv_chunked(p, decimal_separator="a"))

    def test_gzip_compressed_file(self, tmp_path):
        """Test reading gzip-compressed CSV file"""
        p = tmp_path / "data.csv.gz"
        with gzip.open(p, "wt") as f:
            f.write("a,b\n1,x\n2,y\n")
        chunks = list(ar.read_csv_chunked(p, chunksize=10))
        assert chunks[0].shape == (2, 2)
