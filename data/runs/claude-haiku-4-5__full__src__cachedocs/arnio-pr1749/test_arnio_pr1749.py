import pytest
import pandas as pd
import arnio as ar
import os
import tempfile
from pathlib import Path


@pytest.fixture
def csv_3rows():
    """Create a simple 3-row CSV file."""
    content = "id,name,value\n1,Alice,10\n2,Bob,20\n3,Charlie,30\n"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(content)
        f.flush()
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def csv_100rows():
    """Create a 100-row CSV file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("id,name,value\n")
        for i in range(1, 101):
            f.write(f"{i},name_{i},{i*10}\n")
        f.flush()
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def tsv_file():
    """Create a TSV file."""
    content = "id\tname\tvalue\n1\tAlice\t10\n2\tBob\t20\n"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
        f.write(content)
        f.flush()
        yield f.name
    os.unlink(f.name)


class TestReadCsvChunkedBasic:
    """Test core chunked reading behavior."""

    def test_yields_arframe_objects(self, csv_3rows):
        """read_csv_chunked yields ArFrame objects."""
        chunks = list(ar.read_csv_chunked(csv_3rows))
        assert len(chunks) >= 1
        for chunk in chunks:
            assert hasattr(chunk, 'shape')
            assert hasattr(chunk, 'columns')

    def test_chunksize_limits_rows_per_chunk(self, csv_100rows):
        """Chunks respect the chunksize parameter."""
        chunks = list(ar.read_csv_chunked(csv_100rows, chunksize=15))
        assert len(chunks) == 7  # 100 rows / 15 ≈ 7 chunks
        for chunk in chunks[:-1]:
            assert chunk.shape[0] == 15
        assert chunks[-1].shape[0] == 10  # Last chunk: 100 % 15

    def test_multiple_small_chunks(self, csv_3rows):
        """Small chunksize creates multiple chunks."""
        chunks = list(ar.read_csv_chunked(csv_3rows, chunksize=1))
        assert len(chunks) == 3
        assert all(chunk.shape[0] == 1 for chunk in chunks)

    def test_columns_consistent_across_chunks(self, csv_100rows):
        """Column names are consistent across all chunks."""
        chunks = list(ar.read_csv_chunked(csv_100rows, chunksize=20))
        expected_cols = ['id', 'name', 'value']
        for chunk in chunks:
            assert chunk.columns == expected_cols


class TestReadCsvChunkedDelimiter:
    """Test delimiter detection and override."""

    def test_tsv_auto_detects_tab_delimiter(self, tsv_file):
        """TSV files auto-detect tab delimiter."""
        chunks = list(ar.read_csv_chunked(tsv_file))
        df = ar.to_pandas(chunks[0])
        assert set(df.columns) == {'id', 'name', 'value'}
        assert len(df) == 2

    def test_explicit_delimiter_overrides_extension(self):
        """Explicit delimiter parameter overrides auto-detection."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
            f.write("id,name,value\n1,Alice,10\n")
            f.flush()
            fname = f.name
        try:
            chunks = list(ar.read_csv_chunked(fname, delimiter=','))
            df = ar.to_pandas(chunks[0])
            assert set(df.columns) == {'id', 'name', 'value'}
        finally:
            os.unlink(fname)


class TestReadCsvChunkedNrows:
    """Test nrows parameter for row limiting."""

    def test_nrows_limits_total_rows(self, csv_100rows):
        """nrows parameter limits total rows across all chunks."""
        chunks = list(ar.read_csv_chunked(csv_100rows, chunksize=20, nrows=35))
        total_rows = sum(chunk.shape[0] for chunk in chunks)
        assert total_rows == 35

    def test_nrows_smaller_than_chunksize(self, csv_100rows):
        """nrows can be smaller than a single chunk."""
        chunks = list(ar.read_csv_chunked(csv_100rows, chunksize=100, nrows=5))
        total_rows = sum(chunk.shape[0] for chunk in chunks)
        assert total_rows == 5


class TestReadCsvChunkedSkipRows:
    """Test skip_rows and skiprows parameters."""

    def test_skip_rows_parameter(self, csv_3rows):
        """skip_rows parameter skips data rows after header."""
        chunks = list(ar.read_csv_chunked(csv_3rows, skip_rows=1))
        df = ar.to_pandas(chunks[0])
        assert len(df) == 2
        assert df.iloc[0]['id'] == 2  # First row should be Bob

    def test_skiprows_alias_for_skip_rows(self, csv_3rows):
        """skiprows is an alias for skip_rows."""
        chunks = list(ar.read_csv_chunked(csv_3rows, skiprows=1))
        df = ar.to_pandas(chunks[0])
        assert len(df) == 2

    def test_conflicting_skip_and_skiprows_raises(self, csv_3rows):
        """Conflicting skip_rows and skiprows raise ValueError."""
        with pytest.raises(ValueError, match="Conflicting"):
            list(ar.read_csv_chunked(csv_3rows, skip_rows=1, skiprows=2))


class TestReadCsvChunkedUsecols:
    """Test column selection."""

    def test_usecols_selects_subset_of_columns(self, csv_3rows):
        """usecols parameter selects only specified columns."""
        chunks = list(ar.read_csv_chunked(csv_3rows, usecols=['id', 'name']))
        chunk = chunks[0]
        assert chunk.columns == ['id', 'name']
        assert 'value' not in chunk.columns


class TestReadCsvChunkedErrors:
    """Test error handling and validation."""

    def test_unsupported_file_format_raises_error(self):
        """Unsupported file extensions raise ValueError."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            f.write(b'{}')
            fname = f.name
        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                list(ar.read_csv_chunked(fname))
        finally:
            os.unlink(fname)

    def test_missing_file_raises_error(self):
        """Missing files raise an error."""
        with pytest.raises((FileNotFoundError, ar.CsvReadError)):
            list(ar.read_csv_chunked('/nonexistent/path.csv'))

    def test_invalid_chunksize_raises_error(self, csv_3rows):
        """Invalid chunksize (zero or negative) raises ValueError."""
        with pytest.raises(ValueError):
            list(ar.read_csv_chunked(csv_3rows, chunksize=0))
        with pytest.raises(ValueError):
            list(ar.read_csv_chunked(csv_3rows, chunksize=-1))


class TestReadCsvChunkedInputTypes:
    """Test different input path types."""

    def test_string_path_input(self, csv_3rows):
        """String paths work correctly."""
        chunks = list(ar.read_csv_chunked(csv_3rows))
        assert len(chunks) >= 1

    def test_pathlib_path_input(self, csv_3rows):
        """pathlib.Path objects work correctly."""
        path = Path(csv_3rows)
        chunks = list(ar.read_csv_chunked(path))
        assert len(chunks) >= 1


class TestReadCsvChunkedOptions:
    """Test miscellaneous options."""

    def test_trim_headers_removes_whitespace(self):
        """trim_headers removes leading/trailing whitespace from names."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(" id , name , value \n1,Alice,10\n")
            f.flush()
            fname = f.name
        try:
            chunks = list(ar.read_csv_chunked(fname, trim_headers=True))
            assert chunks[0].columns == ['id', 'name', 'value']
        finally:
            os.unlink(fname)

    def test_no_header_option(self):
        """has_header=False reads files without a header row."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("1,Alice,10\n2,Bob,20\n")
            f.flush()
            fname = f.name
        try:
            chunks = list(ar.read_csv_chunked(fname, has_header=False))
            assert len(chunks) >= 1
        finally:
            os.unlink(fname)
