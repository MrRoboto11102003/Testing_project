import pytest
import io
import gzip
from pathlib import Path
import arnio as ar


class TestReadCsvChunked:
    """Focused pytest tests for arnio.io.read_csv_chunked"""

    def test_basic_chunking(self, tmp_path):
        """Verify chunks are yielded with correct sizes up to chunksize limit."""
        p = tmp_path / "data.csv"
        p.write_text("a,b\n1,x\n2,y\n3,z\n4,w\n5,v\n")
        
        chunks = list(ar.read_csv_chunked(p, chunksize=2))
        
        assert len(chunks) == 3
        assert [c.shape[0] for c in chunks] == [2, 2, 1]

    def test_nrows_respects_limit(self, tmp_path):
        """Verify nrows parameter limits total rows read across all chunks."""
        p = tmp_path / "data.csv"
        p.write_text("a,b\n1,x\n2,y\n3,z\n4,w\n")
        
        chunks = list(ar.read_csv_chunked(p, chunksize=2, nrows=3))
        
        total_rows = sum(c.shape[0] for c in chunks)
        assert total_rows == 3

    def test_skip_rows_skips_data_rows(self, tmp_path):
        """Verify skip_rows parameter skips data rows after header."""
        p = tmp_path / "data.csv"
        p.write_text("a,b\n1,x\n2,y\n3,z\n4,w\n")
        
        chunks = list(ar.read_csv_chunked(p, skip_rows=2))
        df = ar.to_pandas(chunks[0])
        
        assert df.shape[0] == 2
        assert df["a"].tolist() == [3, 4]

    def test_usecols_selection(self, tmp_path):
        """Verify usecols parameter selects only specified columns."""
        p = tmp_path / "data.csv"
        p.write_text("a,b,c\n1,2,3\n4,5,6\n7,8,9\n")
        
        chunks = list(ar.read_csv_chunked(p, usecols=["a", "c"]))
        
        assert chunks[0].shape[1] == 2
        assert set(chunks[0].columns) == {"a", "c"}

    def test_tsv_auto_delimiter_detection(self, tmp_path):
        """Verify .tsv extension auto-detects tab delimiter when None passed."""
        p = tmp_path / "data.tsv"
        p.write_text("col1\tcol2\n1\t2\n3\t4\n")
        
        chunks = list(ar.read_csv_chunked(p))
        
        assert chunks[0].shape[1] == 2
        assert set(chunks[0].columns) == {"col1", "col2"}

    def test_explicit_delimiter_overrides_extension(self, tmp_path):
        """Verify explicit delimiter parameter overrides file extension detection."""
        p = tmp_path / "data.tsv"
        p.write_text("a,b\n1,x\n2,y\n")
        
        chunks = list(ar.read_csv_chunked(p, delimiter=","))
        
        assert chunks[0].shape[1] == 2

    def test_gzip_decompression(self, tmp_path):
        """Verify .csv.gz files are decompressed and read correctly."""
        p = tmp_path / "data.csv.gz"
        with gzip.open(p, "wt") as f:
            f.write("a,b\n1,x\n2,y\n")
        
        chunks = list(ar.read_csv_chunked(p))
        
        assert len(chunks) == 1
        assert chunks[0].shape == (2, 2)

    def test_pathlib_path_input(self, tmp_path):
        """Verify pathlib.Path objects are accepted as input."""
        p = tmp_path / "data.csv"
        p.write_text("a,b\n1,x\n2,y\n")
        
        chunks = list(ar.read_csv_chunked(Path(p)))
        
        assert len(chunks) == 1
        assert chunks[0].shape[0] == 2

    def test_stringio_file_like_input(self):
        """Verify StringIO and other file-like objects work as input."""
        text = "a,b\n1,x\n2,y\n3,z\n"
        
        chunks = list(ar.read_csv_chunked(io.StringIO(text), chunksize=1))
        
        assert len(chunks) == 3
        assert all(c.shape[0] == 1 for c in chunks)

    def test_chunksize_zero_raises_valueerror(self, tmp_path):
        """Verify chunksize=0 raises ValueError on first iteration."""
        p = tmp_path / "data.csv"
        p.write_text("a\n1\n")
        
        with pytest.raises(ValueError, match="chunksize must be a positive integer"):
            next(ar.read_csv_chunked(p, chunksize=0))

    def test_chunksize_bool_raises_typeerror(self, tmp_path):
        """Verify chunksize rejects boolean values as TypeError."""
        p = tmp_path / "data.csv"
        p.write_text("a\n1\n")
        
        with pytest.raises(TypeError, match="chunksize must be an integer"):
            next(ar.read_csv_chunked(p, chunksize=True))

    def test_unsupported_extension_raises_valueerror(self, tmp_path):
        """Verify unsupported file extensions raise ValueError immediately."""
        p = tmp_path / "data.dat"
        p.write_text("a\n1\n")
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            next(ar.read_csv_chunked(p))

    def test_empty_file_raises_csvreaderror(self, tmp_path):
        """Verify empty CSV file raises CsvReadError with appropriate message."""
        p = tmp_path / "empty.csv"
        p.write_text("")
        
        with pytest.raises(ar.CsvReadError, match="CSV file is empty"):
            next(ar.read_csv_chunked(p))

    def test_skiprows_skiprows_conflict_raises_valueerror(self, tmp_path):
        """Verify conflicting skip_rows and skiprows values raise ValueError."""
        p = tmp_path / "data.csv"
        p.write_text("a\n1\n2\n3\n")
        
        with pytest.raises(ValueError, match="Conflicting values"):
            next(ar.read_csv_chunked(p, skip_rows=1, skiprows=2))

    def test_delimiter_multiple_chars_raises_valueerror(self, tmp_path):
        """Verify multi-character delimiter raises ValueError."""
        p = tmp_path / "data.csv"
        p.write_text("a,b\n1,x\n")
        
        with pytest.raises(ValueError, match="delimiter must be a single character"):
            next(ar.read_csv_chunked(p, delimiter="||"))
