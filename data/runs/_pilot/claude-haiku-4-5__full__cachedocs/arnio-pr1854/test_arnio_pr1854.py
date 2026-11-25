import pytest
import io
import tempfile
import os
from pathlib import Path
import arnio as ar
import pandas as pd


@pytest.fixture
def csv_file():
    """Create a simple CSV file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        f.write("id,name,value\n")
        f.write("1,alice,100\n")
        f.write("2,bob,200\n")
        f.write("3,charlie,300\n")
        f.write("4,david,400\n")
        f.write("5,eve,500\n")
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def tsv_file():
    """Create a TSV file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False, newline='') as f:
        f.write("id\tname\tvalue\n")
        f.write("1\talice\t100\n")
        f.write("2\tbob\t200\n")
        f.write("3\tcharlie\t300\n")
        f.write("4\tdavid\t400\n")
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def csv_with_nulls():
    """Create a CSV with missing values."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        f.write("id,name,score\n")
        f.write("1,alice,\n")
        f.write("2,,90\n")
        f.write("3,charlie,85\n")
        f.write("4,,\n")
        f.write("5,eve,95\n")
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def large_csv():
    """Create a CSV with 100 data rows."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        f.write("id,name,value\n")
        for i in range(1, 101):
            f.write(f"{i},user_{i},{i*10}\n")
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def header_only_csv():
    """Create a CSV with header but no data rows."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        f.write("id,name,value\n")
        path = f.name
    yield path
    os.unlink(path)


class TestReadCsvChunked:
    """Test suite for ar.read_csv_chunked()."""

    def test_basic_chunking(self, csv_file):
        """Test that chunking yields multiple ArFrame objects."""
        chunks = list(ar.read_csv_chunked(csv_file, chunksize=2))
        assert len(chunks) == 3, "5 rows / 2 per chunk should yield 3 chunks"
        assert chunks[0].shape[0] == 2
        assert chunks[1].shape[0] == 2
        assert chunks[2].shape[0] == 1

    def test_columns_consistent_across_chunks(self, csv_file):
        """Test that column names are identical across all chunks."""
        chunks = list(ar.read_csv_chunked(csv_file, chunksize=2))
        expected_cols = ["id", "name", "value"]
        for chunk in chunks:
            assert chunk.columns == expected_cols

    def test_dtypes_consistent_across_chunks(self, csv_file):
        """Test that dtypes inferred from first chunk apply to all chunks."""
        chunks = list(ar.read_csv_chunked(csv_file, chunksize=2))
        first_dtypes = chunks[0].dtypes
        for chunk in chunks[1:]:
            assert chunk.dtypes == first_dtypes

    def test_custom_chunksize(self, large_csv):
        """Test chunksize parameter controls rows per chunk."""
        chunks = list(ar.read_csv_chunked(large_csv, chunksize=25))
        assert len(chunks) == 4, "100 rows / 25 per chunk should yield 4 chunks"
        assert all(chunk.shape[0] == 25 for chunk in chunks)

    def test_usecols_filters_columns(self, csv_file):
        """Test usecols parameter selects subset of columns."""
        chunks = list(ar.read_csv_chunked(csv_file, usecols=["id", "name"], chunksize=2))
        for chunk in chunks:
            assert chunk.columns == ["id", "name"]
            assert len(chunk.columns) == 2

    def test_nrows_limits_total_rows(self, large_csv):
        """Test nrows parameter limits total rows across all chunks."""
        chunks = list(ar.read_csv_chunked(large_csv, chunksize=20, nrows=50))
        total_rows = sum(chunk.shape[0] for chunk in chunks)
        assert total_rows == 50

    def test_nrows_smaller_than_chunksize(self, csv_file):
        """Test nrows when smaller than chunksize."""
        chunks = list(ar.read_csv_chunked(csv_file, chunksize=100, nrows=2))
        assert len(chunks) == 1
        assert chunks[0].shape[0] == 2

    def test_skip_rows_parameter(self, csv_file):
        """Test skip_rows skips data rows after header."""
        chunks = list(ar.read_csv_chunked(csv_file, skip_rows=2, chunksize=10))
        # 5 data rows - 2 skipped = 3 remaining
        total_rows = sum(chunk.shape[0] for chunk in chunks)
        assert total_rows == 3

    def test_tsv_delimiter_auto_detection(self, tsv_file):
        """Test .tsv extension auto-detects tab delimiter."""
        chunks = list(ar.read_csv_chunked(tsv_file, chunksize=2))
        assert len(chunks) == 2, "4 rows / 2 per chunk should yield 2 chunks"
        assert chunks[0].columns == ["id", "name", "value"]
        # Verify data is correctly parsed with tab delimiter
        df = ar.to_pandas(chunks[0])
        assert df.shape[1] == 3

    def test_explicit_delimiter_overrides_extension(self, tsv_file):
        """Test explicit delimiter parameter takes precedence."""
        # Read TSV file with explicit tab delimiter (should work)
        chunks = list(ar.read_csv_chunked(tsv_file, delimiter="\t", chunksize=2))
        df = ar.to_pandas(chunks[0])
        assert df.shape[1] == 3, "Should have 3 columns when read with tab delimiter"

    def test_null_values_parameter(self, csv_with_nulls):
        """Test null_values parameter treats specified strings as nulls."""
        chunks = list(ar.read_csv_chunked(csv_with_nulls, chunksize=10))
        df = ar.to_pandas(chunks[0])
        # Empty cells should be treated as null
        assert df.isna().sum().sum() >= 3, "Should have null values in the data"

    def test_file_like_object(self):
        """Test reading from file-like object (StringIO)."""
        csv_content = "id,name,value\n1,alice,100\n2,bob,200\n3,charlie,300\n"
        f = io.StringIO(csv_content)
        chunks = list(ar.read_csv_chunked(f, chunksize=2))
        assert len(chunks) == 2, "3 rows / 2 per chunk should yield 2 chunks"
        assert chunks[0].shape[0] == 2

    def test_pathlib_path_support(self, csv_file):
        """Test that pathlib.Path objects are supported."""
        path_obj = Path(csv_file)
        chunks = list(ar.read_csv_chunked(path_obj, chunksize=2))
        assert len(chunks) == 3

    def test_has_header_false(self):
        """Test reading CSV without header row."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
            f.write("1,alice,100\n")
            f.write("2,bob,200\n")
            path = f.name
        try:
            chunks = list(ar.read_csv_chunked(path, has_header=False, chunksize=1))
            assert len(chunks) == 2
            # Columns should be auto-generated
            assert len(chunks[0].columns) == 3
        finally:
            os.unlink(path)

    def test_empty_csv_file(self, header_only_csv):
        """Test reading CSV with header but no data rows."""
        chunks = list(ar.read_csv_chunked(header_only_csv, chunksize=10))
        # Iterator should be empty when no data rows
        assert len(chunks) == 0

    def test_total_rows_calculation(self, large_csv):
        """Test that all rows are retrieved across all chunks."""
        chunks = list(ar.read_csv_chunked(large_csv, chunksize=30))
        total_rows = sum(chunk.shape[0] for chunk in chunks)
        assert total_rows == 100, "Total rows across all chunks should equal data rows"

    def test_iterator_returns_arframe_objects(self, csv_file):
        """Test that iterator yields ArFrame objects."""
        for chunk in ar.read_csv_chunked(csv_file, chunksize=2):
            assert isinstance(chunk, ar.ArFrame)
