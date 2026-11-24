import pytest
import io
import gzip
import re
from pathlib import Path
import arnio as ar


class TestReadCsvChunked:
    """Tests for arnio.io.read_csv_chunked"""

    def test_basic_chunking(self, tmp_path):
        """Test basic CSV chunking with correct chunk sizes"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,x\n2,y\n3,z\n4,w\n5,v\n")

        chunks = list(ar.read_csv_chunked(str(csv_file), chunksize=2))

        assert len(chunks) == 3
        assert [c.shape[0] for c in chunks] == [2, 2, 1]
        assert all(c.shape[1] == 2 for c in chunks)

    def test_tsv_auto_delimiter_detection(self, tmp_path):
        """Test automatic tab delimiter detection for .tsv files"""
        tsv_file = tmp_path / "data.tsv"
        tsv_file.write_text("a\tb\n1\tx\n2\ty\n")

        chunks = list(ar.read_csv_chunked(str(tsv_file)))
        assert chunks[0].shape[1] == 2

    def test_explicit_delimiter_overrides_tsv(self, tmp_path):
        """Test explicit delimiter parameter overrides .tsv auto-detection"""
        tsv_file = tmp_path / "data.tsv"
        tsv_file.write_text("a,b\n1,x\n2,y\n")

        chunks = list(ar.read_csv_chunked(str(tsv_file), delimiter=","))
        assert chunks[0].shape[1] == 2

    def test_pathlib_path_input(self, tmp_path):
        """Test that pathlib.Path input is accepted"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,x\n2,y\n")

        chunks = list(ar.read_csv_chunked(csv_file, chunksize=10))
        assert chunks[0].shape == (2, 2)

    def test_chunksize_zero_raises_on_consumption(self, tmp_path):
        """Test that chunksize=0 raises ValueError on first consumption"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a\n1\n")

        with pytest.raises(ValueError, match="positive integer"):
            next(ar.read_csv_chunked(str(csv_file), chunksize=0))

    def test_chunksize_bool_raises_on_consumption(self, tmp_path):
        """Test that bool chunksize raises TypeError on consumption"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a\n1\n")

        with pytest.raises(TypeError, match="integer"):
            next(ar.read_csv_chunked(str(csv_file), chunksize=True))

    def test_skiprows_skips_data_rows(self, tmp_path):
        """Test skiprows parameter skips data rows after header"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,x\n2,y\n3,z\n4,w\n")

        chunks = list(ar.read_csv_chunked(str(csv_file), skiprows=1, chunksize=10))
        assert chunks[0].shape[0] == 3

    def test_skiprows_conflict_raises_on_consumption(self, tmp_path):
        """Test conflicting skiprows and skip_rows parameters raise ValueError"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a\n1\n")

        with pytest.raises(ValueError, match="Conflicting values"):
            next(ar.read_csv_chunked(str(csv_file), skip_rows=2, skiprows=3))

    def test_usecols_filters_columns(self, tmp_path):
        """Test usecols parameter selects specific columns"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b,c\n1,x,10\n2,y,20\n")

        chunks = list(ar.read_csv_chunked(str(csv_file), usecols=["a", "c"]))
        assert chunks[0].shape[1] == 2
        assert chunks[0].shape[0] == 2

    def test_nrows_limits_total_rows(self, tmp_path):
        """Test nrows parameter limits total number of rows across all chunks"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,x\n2,y\n3,z\n4,w\n5,v\n")

        chunks = list(ar.read_csv_chunked(str(csv_file), nrows=3, chunksize=2))
        total = sum(c.shape[0] for c in chunks)
        assert total == 3

    def test_bad_extension_raises_on_consumption(self, tmp_path):
        """Test that unsupported file extension raises ValueError"""
        bad_file = tmp_path / "data.dat"
        bad_file.write_text("a,b\n1,x\n")

        with pytest.raises(ValueError, match="Unsupported file format"):
            next(ar.read_csv_chunked(str(bad_file)))

    def test_empty_file_raises_on_consumption(self, tmp_path):
        """Test that empty file raises CsvReadError"""
        empty_file = tmp_path / "empty.csv"
        empty_file.write_text("")

        with pytest.raises(ar.CsvReadError, match="empty"):
            next(ar.read_csv_chunked(str(empty_file)))

    def test_encoding_not_string_raises(self, tmp_path):
        """Test that non-string encoding raises TypeError on consumption"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a\n1\n")

        with pytest.raises(TypeError, match="encoding must be a string"):
            next(ar.read_csv_chunked(str(csv_file), encoding=123))

    def test_decimal_separator_validation(self, tmp_path):
        """Test decimal_separator must be single non-alphanumeric character"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a\n1\n")

        with pytest.raises(ValueError, match="single character"):
            next(ar.read_csv_chunked(str(csv_file), decimal_separator="ab"))

    def test_mode_validation_on_consumption(self, tmp_path):
        """Test mode parameter must be 'strict' or 'permissive'"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a\n1\n")

        with pytest.raises(ValueError, match="'strict' or 'permissive'"):
            next(ar.read_csv_chunked(str(csv_file), mode="invalid"))

    def test_generator_is_lazy(self, tmp_path):
        """Test that read_csv_chunked returns generator and validates on consumption"""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a\n1\n")

        # Calling with invalid chunksize should not raise
        gen = ar.read_csv_chunked(str(csv_file), chunksize=-1)
        assert hasattr(gen, '__iter__')

        # Error should occur only when consuming
        with pytest.raises(ValueError):
            next(gen)
