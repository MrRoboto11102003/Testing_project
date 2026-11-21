import pytest
import tempfile
import os
import gzip
from pathlib import Path
import arnio as ar


class TestReadCsvChunked:
    """Tests for arnio.io.read_csv_chunked function."""

    @pytest.fixture
    def simple_csv(self):
        """Create a simple CSV file with 25 rows."""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.csv', delete=False, newline=''
        ) as f:
            f.write("name,age,score\n")
            for i in range(25):
                f.write(f"person_{i},25,{80 + i}\n")
            path = f.name
        yield path
        if os.path.exists(path):
            os.unlink(path)

    @pytest.fixture
    def tsv_file(self):
        """Create a TSV file."""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.tsv', delete=False, newline=''
        ) as f:
            f.write("name\tage\tscore\n")
            for i in range(20):
                f.write(f"person_{i}\t{20+i}\t{80+i}\n")
            path = f.name
        yield path
        if os.path.exists(path):
            os.unlink(path)

    @pytest.fixture
    def gzip_csv(self):
        """Create a gzipped CSV file."""
        with tempfile.NamedTemporaryFile(suffix='.csv.gz', delete=False) as f:
            path = f.name
        with gzip.open(path, 'wt', newline='') as f:
            f.write("id,value\n")
            for i in range(30):
                f.write(f"{i},val_{i}\n")
        yield path
        if os.path.exists(path):
            os.unlink(path)

    @pytest.fixture
    def csv_with_nulls(self):
        """Create a CSV with null values."""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.csv', delete=False, newline=''
        ) as f:
            f.write("col1,col2,col3\n")
            f.write("a,1,x\n")
            f.write("b,NA,y\n")
            f.write("c,3,\n")
            f.write("d,NULL,z\n")
            path = f.name
        yield path
        if os.path.exists(path):
            os.unlink(path)

    @pytest.fixture
    def csv_no_header(self):
        """Create a CSV without header."""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.csv', delete=False, newline=''
        ) as f:
            for i in range(12):
                f.write(f"val1_{i},val2_{i},val3_{i}\n")
            path = f.name
        yield path
        if os.path.exists(path):
            os.unlink(path)

    def test_basic_chunking(self, simple_csv):
        """Test basic chunked reading with proper chunk sizes."""
        chunks = list(ar.read_csv_chunked(simple_csv, chunksize=10))
        assert len(chunks) == 3
        assert chunks[0].shape[0] == 10
        assert chunks[1].shape[0] == 10
        assert chunks[2].shape[0] == 5

    def test_chunksize_respected(self, simple_csv):
        """Test that no chunk exceeds the specified chunksize."""
        for chunk in ar.read_csv_chunked(simple_csv, chunksize=7):
            assert chunk.shape[0] <= 7

    def test_nrows_limit_total_rows(self, simple_csv):
        """Test nrows parameter limits total rows read."""
        chunks = list(ar.read_csv_chunked(simple_csv, chunksize=10, nrows=12))
        total = sum(c.shape[0] for c in chunks)
        assert total == 12

    def test_skip_rows_parameter(self, simple_csv):
        """Test skip_rows parameter reduces total rows."""
        all_chunks = list(ar.read_csv_chunked(simple_csv, chunksize=100))
        skip_chunks = list(ar.read_csv_chunked(simple_csv, chunksize=100, skip_rows=5))
        
        all_total = sum(c.shape[0] for c in all_chunks)
        skip_total = sum(c.shape[0] for c in skip_chunks)
        
        assert skip_total == all_total - 5

    def test_skiprows_alias(self, simple_csv):
        """Test skiprows alias for skip_rows."""
        chunks = list(ar.read_csv_chunked(simple_csv, chunksize=100, skiprows=3))
        total = sum(c.shape[0] for c in chunks)
        assert total == 22

    def test_conflicting_skip_values_raise_error(self, simple_csv):
        """Test that conflicting skip_rows and skiprows raise ValueError."""
        with pytest.raises(ValueError, match="Conflicting values"):
            list(ar.read_csv_chunked(
                simple_csv, skip_rows=2, skiprows=5, chunksize=10
            ))

    def test_tsv_auto_delimiter_detection(self, tsv_file):
        """Test that .tsv files auto-detect tab delimiter."""
        chunks = list(ar.read_csv_chunked(tsv_file, chunksize=10))
        assert chunks[0].shape[1] == 3

    def test_delimiter_explicit_override(self, tsv_file):
        """Test explicit delimiter overrides auto-detection."""
        chunks = list(ar.read_csv_chunked(tsv_file, delimiter=",", chunksize=10))
        assert chunks[0].shape[1] == 1

    def test_gzip_csv_support(self, gzip_csv):
        """Test reading .csv.gz compressed files."""
        chunks = list(ar.read_csv_chunked(gzip_csv, chunksize=10))
        total = sum(c.shape[0] for c in chunks)
        assert total == 30

    def test_usecols_column_selection(self, simple_csv):
        """Test usecols parameter selects specified columns."""
        chunks = list(ar.read_csv_chunked(
            simple_csv, chunksize=10, usecols=['name', 'age']
        ))
        assert chunks[0].shape[1] == 2

    def test_no_header_mode(self, csv_no_header):
        """Test reading CSV without header."""
        chunks = list(ar.read_csv_chunked(
            csv_no_header, has_header=False, chunksize=5
        ))
        assert chunks[0].shape[1] == 3

    def test_null_values_handling(self, csv_with_nulls):
        """Test null_values parameter recognizes specified null strings."""
        chunks = list(ar.read_csv_chunked(
            csv_with_nulls, chunksize=10, null_values=['NA', 'NULL']
        ))
        assert len(chunks) > 0

    def test_file_like_object_input(self, simple_csv):
        """Test reading from file-like object."""
        with open(simple_csv, 'r') as f:
            chunks = list(ar.read_csv_chunked(f, chunksize=10))
            total = sum(c.shape[0] for c in chunks)
            assert total == 25

    def test_pathlib_path_input(self, simple_csv):
        """Test Path object support."""
        path = Path(simple_csv)
        chunks = list(ar.read_csv_chunked(path, chunksize=10))
        assert len(chunks) == 3

    def test_permissive_parser_mode(self, simple_csv):
        """Test permissive parser mode."""
        chunks = list(ar.read_csv_chunked(
            simple_csv, mode='permissive', chunksize=10
        ))
        assert len(chunks) > 0

    def test_unsupported_file_format_error(self, tmp_path):
        """Test that unsupported file formats raise ValueError."""
        bad_file = tmp_path / "test.xlsx"
        bad_file.write_text("data")
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            list(ar.read_csv_chunked(str(bad_file)))
